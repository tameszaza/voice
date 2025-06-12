#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import MultiGenerator, MultiEncoder, Discriminator, Classifier, Bandit

# -----------------------------------------------------------------------------
#  Dataset: only benign folders under real_data_root, each with mel/ and/or mfcc/
# -----------------------------------------------------------------------------
class RealOnlyMultiFeatureDataset(Dataset):
    def __init__(self, real_data_root, use_mel=True, use_mfcc=True):
        if not (use_mel or use_mfcc):
            raise ValueError("Need at least one of --use_mel or --use_mfcc")
        self.samples = []

        # scan benign class subfolders
        classes = sorted(
            d for d in os.listdir(real_data_root)
            if os.path.isdir(os.path.join(real_data_root, d))
        )
        if not classes:
            raise ValueError(f"No subfolders in {real_data_root}")

        for cls in classes:
            base = os.path.join(real_data_root, cls)
            mel_dir  = os.path.join(base, "mel")  if use_mel  else None
            mfcc_dir = os.path.join(base, "mfcc") if use_mfcc else None

            if use_mel  and not os.path.isdir(mel_dir):
                raise FileNotFoundError(f"Missing mel/ under {base}")
            if use_mfcc and not os.path.isdir(mfcc_dir):
                raise FileNotFoundError(f"Missing mfcc/ under {base}")

            # list filenames from first available channel
            file_list = sorted(os.listdir(mel_dir if use_mel else mfcc_dir))
            for fn in file_list:
                if not fn.endswith(".npy"):
                    continue
                m = os.path.join(mel_dir, fn)  if use_mel  else None
                f = os.path.join(mfcc_dir, fn) if use_mfcc else None
                if use_mel and use_mfcc and not os.path.exists(f):
                    raise FileNotFoundError(f"{f} missing")
                self.samples.append((m, f))

        if not self.samples:
            raise ValueError("No real samples found!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        m, f = self.samples[idx]
        feats = []
        if m:
            feats.append(np.load(m))
        if f:
            feats.append(np.load(f))
        x = np.stack(feats, axis=0)    # [C, H, W]
        return torch.from_numpy(x).float()

# -----------------------------------------------------------------------------
#  Load & freeze all but E
# -----------------------------------------------------------------------------
def load_models(args, device, n_features, H, W):
    G = MultiGenerator(args.z_dim, out_channels=n_features,
                       base_channels=32, img_size=H, n_layers=3,
                       n_clusters=args.n_clusters).to(device)
    E = MultiEncoder(in_channels=n_features, z_dim=args.z_dim,
                     base_channels=32, img_size=H, n_layers=3,
                     n_clusters=args.n_clusters).to(device)
    D = Discriminator(in_channels=n_features,
                      base_channels=32, n_layers=3).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, n_features, H, W, device=device)
        feat  = D.intermediate(dummy)
        c_in  = feat.shape[1]
    C = Classifier(in_channels=c_in,
                   n_clusters=args.n_clusters).to(device)
    B = Bandit(args.n_clusters).to(device)

    # load checkpoints
    sfx = args.ckpt_suffix
    ckpt_dir = args.log_dir
    G.load_state_dict(torch.load(os.path.join(ckpt_dir, f"G_{sfx}.pt"), map_location=device))
    E.load_state_dict(torch.load(os.path.join(ckpt_dir, f"E_{sfx}.pt"), map_location=device))
    D.load_state_dict(torch.load(os.path.join(ckpt_dir, f"D_{sfx}.pt"), map_location=device))
    C.load_state_dict(torch.load(os.path.join(ckpt_dir, f"C_{sfx}.pt"), map_location=device))
    # B.load_state_dict(torch.load(os.path.join(ckpt_dir, f"B_{sfx}.pt"), map_location=device))

    # freeze all except E
    for net in (G, D, C, B):
        for p in net.parameters():
            p.requires_grad = False

    return G, E, D, C, B

# -----------------------------------------------------------------------------
#  Phase 2: x–z–x training on real data only
# -----------------------------------------------------------------------------
def train_xzx(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset of only real data
    ds = RealOnlyMultiFeatureDataset(
        args.real_data_root,
        use_mel=args.use_mel,
        use_mfcc=args.use_mfcc
    )
    x0 = ds[0]
    n_features, H, W = x0.shape

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)

    G, E, D, C, B = load_models(args, device, n_features, H, W)

    opt_E = torch.optim.Adam(E.parameters(), lr=args.lr,
                             betas=(0.5, 0.9))

    writer = SummaryWriter(os.path.join(args.log_dir, "xzx"))
    step = 0

    for epoch in range(1, args.epochs+1):
        running = 0.0
        for x in loader:
            x = x.to(device)

            # predict cluster per sample
            with torch.no_grad():
                feat_real = D.intermediate(x)
                k_pred    = C(feat_real).argmax(dim=1)

            # encode & reconstruct
            z_hat = E(x, k_pred)
            x_hat = G(z_hat, k_pred, target_hw=(H, W))

            # MSE on real
            loss = F.mse_loss(x_hat, x)
            opt_E.zero_grad()
            loss.backward()
            opt_E.step()

            writer.add_scalar("Loss/E_xzx", loss.item(), step)
            running += loss.item()
            step += 1

        avg = running / len(loader)
        print(f"[xzx] Epoch {epoch}/{args.epochs}  Loss_E={avg:.4f}")

        if epoch % args.save_every == 0:
            torch.save(E.state_dict(),
                       os.path.join(args.log_dir, f"E_xzx_{epoch}.pt"))

    # final encoder
    torch.save(E.state_dict(),
               os.path.join(args.log_dir, "E_xzx_final.pt"))
    writer.close()

# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Phase-2 x–z–x Training on multi-folder real data"
    )
    p.add_argument("--real_data_root", required=True,
                   help="Root with subfolders for each real class (each has mel/, mfcc/)")
    p.add_argument("--log_dir",       required=True,
                   help="Phase-1 checkpoints + where to save xzx snapshots")
    p.add_argument("--ckpt_suffix",   required=True,
                   help="Suffix for G_*.pt, E_*.pt, etc. from phase-1")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--epochs",        type=int, default=5000)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--save_every",    type=int,   default=10)
    p.add_argument("--z_dim",         type=int,   default=128)
    p.add_argument("--n_clusters",    type=int,   default=7)
    p.add_argument("--use_mel",       action="store_true",
                   help="Include mel channel")
    p.add_argument("--use_mfcc",      action="store_true",
                   help="Include mfcc channel")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    train_xzx(args)
