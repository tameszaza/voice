#!/usr/bin/env python3
import os
import sys
import glob
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import MultiGenerator, MultiEncoder, Discriminator, Classifier, Bandit

from torch.utils.data import Dataset
import os
import numpy as np
import torch

class RealOnlyMultiFeatureDataset(Dataset):
    def __init__(self,
                 real_data_root: str,
                 use_mel: bool = True,
                 use_mfcc: bool = True,
                 cache_in_ram: bool = True):
        if not (use_mel or use_mfcc):
            raise ValueError("Need at least one of use_mel or use_mfcc")

        self.use_mel = use_mel
        self.use_mfcc = use_mfcc
        self.cache_in_ram = cache_in_ram
        # each entry is (path_pair, idx, is_stacked)
        # where path_pair is either (mel_path, mfcc_path) or (stacked_mel, stacked_mfcc)
        self.samples = []
        self.cached_data = []

        # find each class subfolder
        classes = sorted(
            d for d in os.listdir(real_data_root)
            if os.path.isdir(os.path.join(real_data_root, d))
        )
        if not classes:
            raise ValueError(f"No subfolders in {real_data_root}")

        for cls in classes:
            base = os.path.join(real_data_root, cls)

            # per-channel stacked paths
            mel_stack  = os.path.join(base, "mel",  "stacked.npy") if use_mel  else None
            mfcc_stack = os.path.join(base, "mfcc", "stacked.npy") if use_mfcc else None

            # if either stacked file exists, load lengths and register slices
            if (mel_stack and os.path.isfile(mel_stack)) or \
               (mfcc_stack and os.path.isfile(mfcc_stack)):

                # memory‐map to avoid loading whole thing at once
                arr_m = np.load(mel_stack,  mmap_mode="r") if mel_stack and os.path.isfile(mel_stack)   else None
                arr_f = np.load(mfcc_stack, mmap_mode="r") if mfcc_stack and os.path.isfile(mfcc_stack) else None

                # sanity check
                if arr_m is not None and arr_m.ndim != 3:
                    raise ValueError(f"{mel_stack} must be 3D, got {arr_m.shape}")
                if arr_f is not None and arr_f.ndim != 3:
                    raise ValueError(f"{mfcc_stack} must be 3D, got {arr_f.shape}")

                # number of samples = first dim
                count = arr_m.shape[0] if arr_m is not None else arr_f.shape[0]
                for i in range(count):
                    self.samples.append(((mel_stack, mfcc_stack), i, True))
                continue

            # fallback to per-file .npy under mel/ and mfcc/
            mel_dir  = os.path.join(base, "mel")  if use_mel  else None
            mfcc_dir = os.path.join(base, "mfcc") if use_mfcc else None

            if use_mel  and not os.path.isdir(mel_dir):
                raise FileNotFoundError(f"Missing mel/ under {base}")
            if use_mfcc and not os.path.isdir(mfcc_dir):
                raise FileNotFoundError(f"Missing mfcc/ under {base}")

            file_list = sorted(os.listdir(mel_dir if use_mel else mfcc_dir))
            for fn in file_list:
                if not fn.endswith(".npy"):
                    continue
                m_path = os.path.join(mel_dir, fn)  if use_mel  else None
                f_path = os.path.join(mfcc_dir, fn) if use_mfcc else None
                if use_mel and use_mfcc and not os.path.exists(f_path):
                    raise FileNotFoundError(f"{f_path} missing")
                # idx ignored for per-file mode
                self.samples.append(((m_path, f_path), None, False))

        if not self.samples:
            raise ValueError("No real samples found!")

        # optionally cache all data in RAM
        if self.cache_in_ram:
            for path_pair, idx, is_stacked in self.samples:
                m_pair, f_pair = path_pair
                feats = []

                if is_stacked:
                    # load stacked arrays and index
                    mel_s, mfcc_s = m_pair, f_pair
                    if mel_s and os.path.isfile(mel_s):
                        arr_m = np.load(mel_s, mmap_mode="r")
                        feats.append(arr_m[idx])
                    if mfcc_s and os.path.isfile(mfcc_s):
                        arr_f = np.load(mfcc_s, mmap_mode="r")
                        feats.append(arr_f[idx])
                else:
                    # per-file
                    m_path, f_path = path_pair
                    if m_path is not None:
                        feats.append(np.load(m_path))
                    if f_path is not None:
                        feats.append(np.load(f_path))

                x = np.stack(feats, axis=0).astype(np.float32)  # C×H×W
                self.cached_data.append(torch.from_numpy(x))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        if self.cache_in_ram:
            return self.cached_data[i]

        path_pair, idx, is_stacked = self.samples[i]
        feats = []

        if is_stacked:
            mel_s, mfcc_s = path_pair
            if mel_s and os.path.isfile(mel_s):
                arr_m = np.load(mel_s)
                feats.append(arr_m[idx])
            if mfcc_s and os.path.isfile(mfcc_s):
                arr_f = np.load(mfcc_s)
                feats.append(arr_f[idx])
        else:
            m_path, f_path = path_pair
            if m_path is not None:
                feats.append(np.load(m_path))
            if f_path is not None:
                feats.append(np.load(f_path))

        x = np.stack(feats, axis=0).astype(np.float32)
        return torch.from_numpy(x)



def load_models(args, device, n_features, H, W):
    G = MultiGenerator(args.z_dim, out_channels=n_features,
                       base_channels=32, img_size=H, n_layers=4,
                       n_clusters=args.n_clusters).to(device)
    E = MultiEncoder(in_channels=n_features, z_dim=args.z_dim,
                     base_channels=32, img_size=H, n_layers=4,
                     n_clusters=args.n_clusters).to(device)
    D = Discriminator(in_channels=n_features,
                      base_channels=32, n_layers=4).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, n_features, H, W, device=device)
        feat  = D.intermediate(dummy)
        c_in  = feat.shape[1]
    C = Classifier(in_channels=c_in,
                   n_clusters=args.n_clusters).to(device)
    B = Bandit(args.n_clusters).to(device)

    sfx = args.ckpt_suffix
    ckpt_dir = args.log_dir
    G.load_state_dict(torch.load(os.path.join(ckpt_dir, f"G_{sfx}.pt"), map_location=device))
    E.load_state_dict(torch.load(os.path.join(ckpt_dir, f"E_{sfx}.pt"), map_location=device))
    D.load_state_dict(torch.load(os.path.join(ckpt_dir, f"D_{sfx}.pt"), map_location=device))
    C.load_state_dict(torch.load(os.path.join(ckpt_dir, f"C_{sfx}.pt"), map_location=device))
    # B.load_state_dict(torch.load(os.path.join(ckpt_dir, f"B_{sfx}.pt"), map_location=device))

    for net in (G, D, C, B):
        for p in net.parameters():
            p.requires_grad = False

    return G, E, D, C, B


def find_latest_interrupt(log_dir):
    pattern = re.compile(r"E_xzx_(\d+)_interrupt\.pt$")
    candidates = glob.glob(os.path.join(log_dir, "E_xzx_*_interrupt.pt"))
    best_epoch = 0
    best_paths = None
    for p in candidates:
        m = pattern.search(os.path.basename(p))
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_paths = (
                    p,
                    os.path.join(log_dir, f"opt_E_xzx_{ep}_interrupt.pt")
                )
    return best_epoch, best_paths


def train_xzx(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RealOnlyMultiFeatureDataset(
        args.real_data_root,
        use_mel=args.use_mel,
        use_mfcc=args.use_mfcc,
        cache_in_ram=args.cache_in_ram
    )
    x0 = ds[0]
    n_features, H, W = x0.shape

    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)
    G, E, D, C, B = load_models(args, device, n_features, H, W)

    opt_Es = [
        torch.optim.Adam(enc.parameters(), lr=args.lr, betas=(0.5, 0.9))
        for enc in E.encoders
    ]


    # check for existing interrupt checkpoint
    start_epoch = 1
    latest_ep, paths = find_latest_interrupt(args.log_dir)
    if latest_ep > 0 and os.path.exists(paths[1]):
        print(f"Resuming from interrupted epoch {latest_ep}")
        E.load_state_dict(torch.load(paths[0], map_location=device))
        # if you saved per-cluster optimizer states you could loop here to restore them
        start_epoch = latest_ep + 1

    writer = SummaryWriter(os.path.join(args.log_dir, "xzx"))
    step = (start_epoch - 1) * len(loader)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            running = 0.0
            for x in loader:
                x = x.to(device)
                with torch.no_grad():
                    feat_real = D.intermediate(x)
                    k_pred    = C(feat_real).argmax(dim=1)

                # forward through all
                z_hat = E(x, k_pred)
                x_hat = G(z_hat, k_pred, target_hw=(H, W))

                # now per-cluster reconstruction losses, but defer backward
                cluster_losses = []
                for i in torch.unique(k_pred):
                    idx      = (k_pred == i).nonzero(as_tuple=False).view(-1)
                    x_i      = x[idx]
                    x_hat_i  = x_hat[idx]
                    cluster_losses.append(
                        F.mse_loss(x_hat_i, x_i)
                    )

                # sum into one scalar
                total_e_loss = sum(cluster_losses)

                # zero grads for every sub-encoder
                for opt in opt_Es:
                    opt.zero_grad()

                # single backward pass
                total_e_loss.backward()

                # step every sub-encoder
                for opt in opt_Es:
                    opt.step()

                # log once per batch
                writer.add_scalar("Loss/E_xzx", total_e_loss.item(), step)
                running += total_e_loss.item()
                step += 1

            avg = running / len(loader)
            print(f"[xzx] Epoch {epoch}/{args.epochs}  Loss={avg:.4f}")

            if epoch % args.save_every == 0:
                torch.save(E.state_dict(),
                           os.path.join(args.log_dir, f"E_xzx_{epoch}.pt"))

    except KeyboardInterrupt:
        print(f"Interrupted at epoch {epoch}. Saving encoder and optimizer.")
        torch.save(E.state_dict(),
                   os.path.join(args.log_dir, f"E_xzx_{epoch}_interrupt.pt"))
        for i, opt in enumerate(opt_Es):
            torch.save(opt.state_dict(),
                       os.path.join(args.log_dir, f"opt_E_{i}_xzx_{epoch}_interrupt.pt"))
        writer.close()
        sys.exit(0)

    # normal finish
    torch.save(E.state_dict(),
               os.path.join(args.log_dir, "E_xzx_final.pt"))
    for i, opt in enumerate(opt_Es):
        torch.save(opt.state_dict(),
                    os.path.join(args.log_dir, f"opt_E_{i}_xzx_final_.pt"))
    writer.close()


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
    p.add_argument("--batch_size",    type=int, default=256)
    p.add_argument("--epochs",        type=int, default=5000)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--save_every",    type=int,   default=10)
    p.add_argument("--z_dim",         type=int,   default=128)
    p.add_argument("--n_clusters",    type=int,   default=7)
    p.add_argument("--use_mel",       action="store_true",
                   help="Include mel channel")
    p.add_argument("--use_mfcc",      action="store_true",
                   help="Include mfcc channel")
    p.add_argument("--cache_in_ram",  action="store_true",
                   help="Cache all data in RAM for faster I/O")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    train_xzx(args)
