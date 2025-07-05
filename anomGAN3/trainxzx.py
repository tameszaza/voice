#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import MultiGenerator, MultiEncoder, Discriminator, Classifier, Bandit
from helper import MultiNpyFilesDataset

# -----------------------------------------------------------------------------
#  Load & freeze all but E
# -----------------------------------------------------------------------------
def load_models(args, device, n_features, H, W):
    G = MultiGenerator(
        args.z_dim,
        out_channels=n_features,
        base_channels=32,
        img_size=H,
        n_layers=3,
        n_clusters=args.n_clusters
    ).to(device)

    E = MultiEncoder(
        in_channels=n_features,
        z_dim=args.z_dim,
        base_channels=32,
        img_size=H,
        n_layers=3,
        n_clusters=args.n_clusters
    ).to(device)

    D = Discriminator(
        in_channels=n_features,
        base_channels=32,
        n_layers=3
    ).to(device)

    # figure out classifier input channels
    with torch.no_grad():
        dummy = torch.zeros(1, n_features, H, W, device=device)
        feat  = D.intermediate(dummy)
        c_in  = feat.shape[1]

    C = Classifier(
        in_channels=c_in,
        n_clusters=args.n_clusters
    ).to(device)

    B = Bandit(args.n_clusters).to(device)

    # load checkpoints
    sfx     = args.ckpt_suffix
    ckpt_dir = args.log_dir

    G.load_state_dict(torch.load(
        os.path.join(ckpt_dir, f"G_{sfx}.pt"),
        map_location=device
    ))
    E.load_state_dict(torch.load(
        os.path.join(ckpt_dir, f"E_{sfx}.pt"),
        map_location=device
    ))
    D.load_state_dict(torch.load(
        os.path.join(ckpt_dir, f"D_{sfx}.pt"),
        map_location=device
    ))
    C.load_state_dict(torch.load(
        os.path.join(ckpt_dir, f"C_{sfx}.pt"),
        map_location=device
    ))
    # If you ever train Bandit, uncomment the next line
    # B.load_state_dict(torch.load(os.path.join(ckpt_dir, f"B_{sfx}.pt"),
    #                              map_location=device))

    # freeze everything except the encoder
    for net in (G, D, C, B):
        for p in net.parameters():
            p.requires_grad = False

    return G, E, D, C, B

# -----------------------------------------------------------------------------
#  Phase 2: x–z–x training on real data only
# -----------------------------------------------------------------------------
def train_xzx(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── dataset & loader ────────────────────────────────────────────
    # each .npy file under real_data_root becomes one class,
    # but we ignore labels and only use the samples.
    ds = MultiNpyFilesDataset(
        dir_path=args.real_data_root,
        max_samples_per_file=None
    )

    # inspect one sample to get (C, H, W)
    x0, _ = ds[0]
    n_features, H, W = x0.shape

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # ── models & optim ──────────────────────────────────────────────
    G, E, D, C, B = load_models(args, device, n_features, H, W)

    opt_E = torch.optim.Adam(
        E.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9)
    )

    writer = SummaryWriter(os.path.join(args.log_dir, "xzx"))
    step = 0

    # Optionally, load optimizer state if resuming (not implemented here)
    # opt_path = os.path.join(args.log_dir, f"opt_E_xzx_last.pt")
    # if os.path.exists(opt_path):
    #     opt_E.load_state_dict(torch.load(opt_path, map_location=device))

    for epoch in range(1, args.epochs + 1):
        running = 0.0

        for batch in loader:
            # unpack and move to device
            x_batch, _ = batch
            x = x_batch.to(device)

            # infer cluster labels from D + C
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
            torch.save(
                E.state_dict(),
                os.path.join(args.log_dir, f"E_xzx_{epoch}.pt")
            )
            torch.save(
                opt_E.state_dict(),
                os.path.join(args.log_dir, f"opt_E_xzx_{epoch}.pt")
            )

    # final encoder snapshot
    torch.save(
        E.state_dict(),
        os.path.join(args.log_dir, "E_xzx_final.pt")
    )
    torch.save(
        opt_E.state_dict(),
        os.path.join(args.log_dir, "opt_E_xzx_final.pt")
    )

    writer.close()

# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Phase-2 x–z–x Training on real .npy data"
    )
    p.add_argument(
        "--real_data_root",
        required=True,
        help="Folder containing multiple .npy files (each .npy = one class)"
    )
    p.add_argument(
        "--log_dir",
        required=True,
        help="Directory with phase-1 checkpoints and where to save phase-2 snapshots"
    )
    p.add_argument(
        "--ckpt_suffix",
        required=True,
        help="Suffix for G_*.pt, E_*.pt, etc. from phase-1"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5000
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=10
    )
    p.add_argument(
        "--z_dim",
        type=int,
        default=128
    )
    p.add_argument(
        "--n_clusters",
        type=int,
        default=7
    )

    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    train_xzx(args)
