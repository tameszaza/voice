#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# import your model definitions
from models import MultiGenerator, MultiEncoder, Discriminator, Classifier, Bandit

class NPYAudioDataset(Dataset):
    """same as in Phase 1"""
    def __init__(self, mel_dir=None, mfcc_dir=None):
        self.use_mel  = mel_dir  is not None
        self.use_mfcc = mfcc_dir is not None
        if not (self.use_mel or self.use_mfcc):
            raise ValueError("At least one of mel_dir or mfcc_dir must be specified")
        self.mel_dir  = mel_dir
        self.mfcc_dir = mfcc_dir
        base = mel_dir if self.use_mel else mfcc_dir
        self.files = sorted(f for f in os.listdir(base) if f.endswith('.npy'))
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

def load_models(args, device, n_features, H, W):
    """Instantiate, load weights, and freeze all but E."""
    # 1) instantiate exactly as in Phase 1
    G = MultiGenerator(args.z_dim, out_channels=n_features,
                       base_channels=32, img_size=H, n_layers=3,
                       n_clusters=args.n_clusters).to(device)
    E = MultiEncoder(in_channels=n_features, z_dim=args.z_dim,
                     base_channels=32, img_size=H, n_layers=3,
                     n_clusters=args.n_clusters).to(device)
    D = Discriminator(in_channels=n_features,
                      base_channels=32, n_layers=3).to(device)
    # need feature‐channel count for C
    with torch.no_grad():
        dummy = torch.zeros(1, n_features, H, W, device=device)
        feat  = D.intermediate(dummy)
        c_in  = feat.shape[1]
    C = Classifier(in_channels=c_in,
                   n_clusters=args.n_clusters).to(device)
    B = Bandit(args.n_clusters).to(device)

    # 2) load their saved checkpoints
    sfx = args.ckpt_suffix
    G.load_state_dict(torch.load(os.path.join(args.log_dir, f"G_{sfx}.pt")))
    E.load_state_dict(torch.load(os.path.join(args.log_dir, f"E_{sfx}.pt")))
    D.load_state_dict(torch.load(os.path.join(args.log_dir, f"D_{sfx}.pt")))
    C.load_state_dict(torch.load(os.path.join(args.log_dir, f"C_{sfx}.pt")))
    B.load_state_dict(torch.load(os.path.join(args.log_dir, f"B_{sfx}.pt")))

    # 3) freeze all except E
    for model in (G, D, C, B):
        for p in model.parameters():
            p.requires_grad = False

    return G, E, D, C, B

def train_xzx(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds   = NPYAudioDataset(args.mel_dir, args.mfcc_dir)
    x0   = ds[0]
    n_features, H, W = x0.shape

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)

    G, E, D, C, B = load_models(args, device, n_features, H, W)

    # optimizer for encoder only
    opt_E = torch.optim.Adam(E.parameters(), lr=args.lr,
                             betas=(0.5, 0.9))

    writer = SummaryWriter(os.path.join(args.log_dir, "xzx"))
    step = 0

    for epoch in range(1, args.epochs+1):
        running_loss = 0.0
        for real in loader:
            real = real.to(device)

            # 1) predict cluster index for each x_i
            with torch.no_grad():
                feat_real = D.intermediate(real)            # D_i(x)
                logits    = C(feat_real)                    # class scores
                k_pred    = logits.argmax(dim=1)            # hard assignment

            # 2) encode x -> z_hat, then reconstruct x_hat = G(z_hat, k_pred)
            z_hat = E(real, k_pred)
            x_hat = G(z_hat, k_pred, target_hw=real.shape[-2:])

            # 3) MSE loss & update encoder
            loss = F.mse_loss(x_hat, real)
            opt_E.zero_grad()
            loss.backward()
            opt_E.step()

            writer.add_scalar("Loss/E_xzx", loss.item(), step)
            running_loss += loss.item()
            step += 1

        avg = running_loss / len(loader)
        print(f"[xzx] Epoch {epoch}/{args.epochs}  Loss_E={avg:.4f}")

        # optionally save intermediate encoder snapshots
        if epoch % args.save_every == 0:
            torch.save(E.state_dict(),
                       os.path.join(args.log_dir, f"E_xzx_{epoch}.pt"))

    # final save
    torch.save(E.state_dict(),
               os.path.join(args.log_dir, "E_xzx_final.pt"))
    writer.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Phase 2 x–z–x Training for D-AnoGAN’s Encoder")
    p.add_argument("--mel_dir",      required=True,
                   help="Directory with Mel .npy files")
    p.add_argument("--mfcc_dir",     default=None,
                   help="Directory with MFCC .npy files (optional)")
    p.add_argument("--log_dir",      required=True,
                   help="Where phase-1 checkpoints live + xzx logs")
    p.add_argument("--ckpt_suffix",  required=True,
                   help="Suffix used in saved first-phase .pt files")
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--epochs",       type=int,   default=1000,
                   help="Number of x–z–x training epochs")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--save_every",   type=int,   default=10,
                   help="Save E snapshot every N epochs")
    p.add_argument("--z_dim",        type=int,   default=128)
    p.add_argument("--n_clusters",   type=int,   default=10)
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    train_xzx(args)
