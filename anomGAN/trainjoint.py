#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import Generator, MultiGenerator, Encoder, MultiEncoder, Discriminator, Classifier, Bandit
DEBUG = False
def save_all_models(G, E, D, C, B, log_dir, suffix):
        torch.save(G.state_dict(), os.path.join(log_dir, f"G_{suffix}.pt"))
        torch.save(E.state_dict(), os.path.join(log_dir, f"E_{suffix}.pt"))
        torch.save(D.state_dict(), os.path.join(log_dir, f"D_{suffix}.pt"))
        torch.save(C.state_dict(), os.path.join(log_dir, f"C_{suffix}.pt"))
        torch.save(B.state_dict(), os.path.join(log_dir, f"B_{suffix}.pt"))

# -----------------------------------------------------------------------------
#  Dataset: load paired .npy files (Mel + MFCC) and stack as 2-channel input
# -----------------------------------------------------------------------------
class NPYAudioDataset(Dataset):
    def __init__(self, mel_dir=None, mfcc_dir=None):
        self.use_mel = mel_dir is not None
        self.use_mfcc = mfcc_dir is not None
        if not (self.use_mel or self.use_mfcc):
            raise ValueError("At least one feature type must be specified")
            
        self.mel_dir = mel_dir
        self.mfcc_dir = mfcc_dir
        
        # Use the directory of the first enabled feature to get file list
        base_dir = mel_dir if self.use_mel else mfcc_dir
        self.files = sorted(f for f in os.listdir(base_dir) if f.endswith('.npy'))
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        fn = self.files[idx]
        features = []
        
        if self.use_mel:
            mel = np.load(os.path.join(self.mel_dir, fn))
            features.append(mel)
            
        if self.use_mfcc:
            mfcc = np.load(os.path.join(self.mfcc_dir, fn))
            features.append(mfcc)
            
        x = np.stack(features, axis=0)
        return torch.from_numpy(x).float()

def save_config(args, n_features):
    config_path = os.path.join(args.log_dir, 'configuration.txt')
    with open(config_path, 'w') as f:
        f.write("Training Configuration:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Features used: {n_features} channels\n")
        f.write(f"Using MEL: {args.mel_dir is not None}\n")
        f.write(f"Using MFCC: {args.mfcc_dir is not None}\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")



# -----------------------------------------------------------------------------
#  Gradient penalty (WGAN-GP)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
#  Training loop: Phase 1 Joint Adversarial & Cluster Prior Learning
# -----------------------------------------------------------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = NPYAudioDataset(args.mel_dir, args.mfcc_dir)
    x0 = ds[0]
    n_features = x0.shape[0]
    save_config(args, n_features)
    _, H, W = x0.shape
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # instantiate models
    G = MultiGenerator(
        args.z_dim, out_channels=n_features, base_channels=32,
        img_size=H, n_layers=3, n_clusters=args.n_clusters
    ).to(device)
    E = MultiEncoder(
        in_channels=n_features, z_dim=args.z_dim, base_channels=32,
        img_size=H, n_layers=3, n_clusters=args.n_clusters
    ).to(device)
    D = Discriminator(in_channels=n_features, base_channels=32, n_layers=3).to(device)

    # Dynamically determine feature channels for Classifier
    with torch.no_grad():
        dummy = torch.zeros(1, n_features, H, W, device=device)
        feat = D.intermediate(dummy)
        classifier_in_channels = feat.shape[1]

    C = Classifier(in_channels=classifier_in_channels, n_clusters=args.n_clusters).to(device)
    B = Bandit(args.n_clusters).to(device)

    # optimizers
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_E = torch.optim.Adam(E.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_B = torch.optim.Adam(B.parameters(), lr=args.lr, betas=(0.5, 0.9))

    writer = SummaryWriter(args.log_dir)
    d_steps = 0
    global_step = 0
    total_steps = args.epochs * len(loader)
    beta_t = args.beta0

    try:
        for epoch in range(args.epochs):
            for real in loader:
                real = real.to(device)
                Bp = B.prior()
                z = torch.randn(args.batch_size, args.z_dim, device=device)
                k = B.sample(args.batch_size).to(device)
                # Pass real.shape[-2:] as target_hw to ensure matching size
                fake = G(z, k, target_hw=real.shape[-2:])
                d_real = D(real)
                d_fake = D(fake.detach())
                gp = gradient_penalty(D, real, fake.detach(), device)
                loss_D = d_fake.mean() - d_real.mean() + args.gamma_gp * gp

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
                d_steps += 1

                # log D loss
                writer.add_scalar('Loss/D', loss_D.item(), global_step)

                # --- every n_critic steps: update G, E, C, B ---
                if d_steps % args.n_critic == 0:
                    # -- Generator + Classifier loss --
                    fake = G(z, k, target_hw=real.shape[-2:])
                    d_fake2 = D(fake)
                    feat_fake = D.intermediate(fake)
                    logits = C(feat_fake)
                    loss_cls = F.cross_entropy(logits, k)
                    loss_G   = -d_fake2.mean() - args.lambda_cls * loss_cls

                    opt_G.zero_grad()
                    loss_G.backward(retain_graph=True)
                    opt_G.step()

                    # log G + C
                    writer.add_scalar('Loss/G', loss_G.item(), global_step)
                    writer.add_scalar('Loss/Cls_on_G', loss_cls.item(), global_step)

                    # -- Encoder loss --
                    # -- Encoder loss --
                    z_hat = E(fake.detach(), k)


                    loss_E = F.mse_loss(z_hat, z.detach())
                    opt_E.zero_grad()
                    loss_E.backward()
                    opt_E.step()
                    writer.add_scalar('Loss/E', loss_E.item(), global_step)

                    # -- Classifier standalone (could be the same as above) --
                    opt_C.zero_grad()
                    # Avoid double backward on the same computation graph by recomputing loss_cls
                    with torch.no_grad():
                        feat_fake_detached = feat_fake.detach()
                    logits_detached = C(feat_fake_detached)
                    loss_cls_detached = F.cross_entropy(logits_detached, k)
                    loss_cls_detached.backward()
                    opt_C.step()

                    # -- Bandit loss --
                    feat_real = D.intermediate(real)
                    q_real = F.softmax(C(feat_real), dim=1)      # (B, K)
                    # first term: E[H(B, q)]
                    loss_B1 = - (Bp.unsqueeze(0) * torch.log(q_real + 1e-8)).sum(dim=1).mean()
                    # entropy regularizer
                    H_q = - (q_real * torch.log(q_real + 1e-8)).sum(dim=1).mean()
                    loss_B = loss_B1 - beta_t * args.nu * H_q

                    opt_B.zero_grad()
                    loss_B.backward()
                    opt_B.step()

                    writer.add_scalar('Loss/B', loss_B.item(), global_step)
                    writer.add_histogram('Bandit/prior', B.prior().detach().cpu().numpy(), global_step)

                    # decay beta
                    beta_t = beta_t * args.beta_decay

                global_step += 1

            print(f"Epoch {epoch+1}/{args.epochs}  D_loss={loss_D.item():.4f}  G_loss={loss_G.item():.4f}")
            # Save models every 20 epochs
            if (epoch + 1) % 20 == 0:
                save_all_models(G, E, D, C, B, args.log_dir, epoch+1)
        # Save at the end of training
        save_all_models(G, E, D, C, B, args.log_dir, args.epochs)
    except KeyboardInterrupt:
        print("Interrupted! Saving models...")
        save_all_models(G, E, D, C, B, args.log_dir, f"interrupt_{epoch+1}")
        raise
    finally:
        writer.close()

# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mel_dir",      default=None, help="Directory containing MEL features (optional)")
    p.add_argument("--mfcc_dir",     default=None, help="Directory containing MFCC features (optional)")
    p.add_argument("--log_dir",      required=True)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--epochs",       type=int, default=1000)
    p.add_argument("--z_dim",        type=int, default=128)
    p.add_argument("--n_clusters",   type=int, default=10)
    p.add_argument("--n_critic",     type=int, default=5)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--lambda_cls",   type=float, default=0.1)
    p.add_argument("--gamma_gp",     type=float, default=10.0)
    p.add_argument("--beta0",        type=float, default=1.0)
    p.add_argument("--beta_decay",   type=float, default=0.999)
    p.add_argument("--nu",           type=float, default=0.1)
    args = p.parse_args()
    
    if args.mel_dir is None and args.mfcc_dir is None:
        raise ValueError("At least one of --mel_dir or --mfcc_dir must be specified")
        
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
