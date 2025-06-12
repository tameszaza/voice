#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models import Generator, Encoder, Discriminator

class NPYAudioDataset(Dataset):
    """Dataset class for loading preprocessed audio features"""
    def __init__(self, mel_dir=None, mfcc_dir=None):
        self.use_mel = mel_dir is not None
        self.use_mfcc = mfcc_dir is not None
        if not (self.use_mel or self.use_mfcc):
            raise ValueError("At least one of mel_dir or mfcc_dir must be specified")
        self.mel_dir = mel_dir
        self.mfcc_dir = mfcc_dir
        base_dir = mel_dir if self.use_mel else mfcc_dir
        self.files = sorted(f for f in os.listdir(base_dir) if f.endswith('.npy'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        feats = []
        if self.use_mel:
            feats.append(np.load(os.path.join(self.mel_dir, fn)))
        if self.use_mfcc:
            feats.append(np.load(os.path.join(self.mfcc_dir, fn)))
        x = np.stack(feats, axis=0)
        return torch.from_numpy(x).float()

def gradient_penalty(D, real, fake, device):
    B = real.size(0)
    α = torch.rand(B, 1, 1, 1, device=device)
    inter = α * real + (1 - α) * fake
    inter.requires_grad_(True)
    d_inter = D(inter)
    grads = torch.autograd.grad(
        outputs=d_inter.sum(), inputs=inter,
        create_graph=True, retain_graph=True
    )[0]
    gp = ((grads.view(B, -1).norm(2, dim=1) - 1)**2).mean()
    return gp

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = NPYAudioDataset(args.mel_dir, args.mfcc_dir)

    # ── train/val split ─────────────────────────────────────────
    ds_size = len(ds)
    indices = np.arange(ds_size)
    np.random.shuffle(indices)
    split = int(args.validation_split * ds_size)
    val_idx, train_idx = indices[:split], indices[split:]
    train_loader = DataLoader(
        ds, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_idx), drop_last=True
    )
    val_loader = DataLoader(
        ds, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_idx)
    )

    x0 = ds[0]
    n_features, H, W = x0.shape

    # ── models + optimizers ─────────────────────────────────────
    G = Generator(args.z_dim, n_features, args.base_channels, H, args.n_layers).to(device)
    E = Encoder(n_features, args.z_dim, args.base_channels, H, args.n_layers).to(device)
    D = Discriminator(n_features, args.base_channels, args.n_layers).to(device)

    opt_D = torch.optim.Adam(
        D.parameters(), lr=args.lr, betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )
    opt_G = torch.optim.Adam(
        G.parameters(), lr=args.lr, betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )
    opt_E = torch.optim.Adam(
        E.parameters(), lr=args.lr, betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )

    writer = SummaryWriter(args.log_dir)
    best_val = float('inf')
    patience_cnt = 0
    step = 0
    d_steps = 0

    for epoch in range(1, args.epochs + 1):
        G.train(); E.train(); D.train()
        for real in train_loader:
            real = real.to(device)
            batch = real.size(0)

            # — Discriminator update —
            z = torch.randn(batch, args.z_dim, device=device)
            fake = G(z, target_hw=(H, W))
            d_real = D(real)
            d_fake = D(fake.detach())
            gp = gradient_penalty(D, real, fake.detach(), device)
            loss_D = d_fake.mean() - d_real.mean() + args.gamma_gp * gp

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            d_steps += 1

            # — Generator + Encoder updates (every n_critic steps) —
            if d_steps % args.n_critic == 0:
                # Generator loss
                fake = G(z, target_hw=(H, W))
                loss_G = -D(fake).mean()

                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

                # Encoder on fake
                z_hat_fake = E(fake.detach())
                x_recon_fake = G(z_hat_fake, target_hw=(H, W))
                loss_E_fake = F.mse_loss(x_recon_fake, fake.detach())

                # **New**: Encoder+Generator on real
                z_hat_real = E(real)
                x_recon_real = G(z_hat_real, target_hw=(H, W))
                loss_recon_real = F.mse_loss(x_recon_real, real)

                loss_E = loss_E_fake + args.lambda_recon_real * loss_recon_real

                opt_E.zero_grad()
                opt_G.zero_grad()
                loss_E.backward()
                opt_E.step()
                opt_G.step()

                writer.add_scalar('Loss/G', loss_G.item(), step)
                writer.add_scalar('Loss/E_fake', loss_E_fake.item(), step)
                writer.add_scalar('Loss/E_real', loss_recon_real.item(), step)

            writer.add_scalar('Loss/D', loss_D.item(), step)
            step += 1

        # ── end of epoch: validation & early stopping ─────────────
        G.eval(); E.eval()
        val_losses = []
        with torch.no_grad():
            for real_val in val_loader:
                rv = real_val.to(device)
                z_hat = E(rv)
                recon = G(z_hat, target_hw=(H, W))
                val_losses.append(F.mse_loss(recon, rv).item())
        avg_val = float(np.mean(val_losses))
        writer.add_scalar('Loss/ValRecon', avg_val, epoch)

        print(f"Epoch {epoch}/{args.epochs} — "
              f"D={loss_D:.4f}  G={loss_G:.4f}  "
              f"ReFake={loss_E_fake:.4f}  ReReal={loss_recon_real:.4f}  "
              f"ValRecon={avg_val:.4f}")

        # check early stopping
        if avg_val < best_val:
            best_val = avg_val
            patience_cnt = 0
            # save best
            torch.save(G.state_dict(), os.path.join(args.log_dir, "G_best.pt"))
            torch.save(E.state_dict(), os.path.join(args.log_dir, "E_best.pt"))
            torch.save(D.state_dict(), os.path.join(args.log_dir, "D_best.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping: no improvement in {args.patience} epochs")
                break

        # periodic checkpointing
        if epoch % args.save_every == 0:
            suffix = f"epoch_{epoch}"
            torch.save(G.state_dict(), os.path.join(args.log_dir, f"G_{suffix}.pt"))
            torch.save(E.state_dict(), os.path.join(args.log_dir, f"E_{suffix}.pt"))
            torch.save(D.state_dict(), os.path.join(args.log_dir, f"D_{suffix}.pt"))

    # final save
    torch.save(G.state_dict(), os.path.join(args.log_dir, "G_final.pt"))
    torch.save(E.state_dict(), os.path.join(args.log_dir, "E_final.pt"))
    torch.save(D.state_dict(), os.path.join(args.log_dir, "D_final.pt"))
    writer.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mel_dir", default=None)
    p.add_argument("--mfcc_dir", default=None)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma_gp", type=float, default=10.0)
    p.add_argument("--save_every", type=int, default=500)
    # ── new hyperparameters ───────────────────────────────────
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="L2 regularization on G/E/D")
    p.add_argument("--lambda_recon_real", type=float, default=0.8,
                   help="weight of real-data reconstruction loss")
    p.add_argument("--validation_split", type=float, default=0.1,
                   help="fraction of data held out for validation")
    p.add_argument("--patience", type=int, default=1000,
                   help="epochs to wait before early stopping")
    args = p.parse_args()

    if args.mel_dir is None and args.mfcc_dir is None:
        raise ValueError("At least one of --mel_dir or --mfcc_dir must be specified")
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
