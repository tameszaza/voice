import os
import argparse
import numpy as np
torch_import = True
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from RES_GANomaly_model import RES_MGanomaly
# Enables detailed gradient anomaly detection if needed
#torch.autograd.set_detect_anomaly(True)


# ─────────────────────────────────────────────────────────────────────
# Dataset that handles EITHER
#   • a single .npy file            (old behaviour)
#   • a directory full of *.npy     (cluster1.npy, cluster2.npy, …)
#     and returns (X, k) where k = cluster index ∈ {0,1,…}
# ─────────────────────────────────────────────────────────────────────
class ClusteredLogMelDataset(Dataset):
    """
    * If `path` is a file  → behaves exactly like the old loader, returns (x,)
    * If `path` is a dir   → loads every *.npy once, concatenates,
                              and returns (x, k) where k is the file index.
    All arrays must be shaped (N, C, H, W) with square power-of-two H = W.
    """

    def __init__(self, path: str):
        files = [path] if os.path.isfile(path) else \
                sorted(f for f in os.listdir(path) if f.endswith(".npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found in '{path}'")

        self.data   = []
        self.labels = []           # cluster indices, used as branch_idx  k
        for k, fname in enumerate(files):
            npy_path = fname if os.path.isabs(fname) else os.path.join(path, fname)
            arr = np.load(npy_path).astype(np.float32)  # (N,C,H,W)

            if arr.ndim != 4:
                raise ValueError(f"{fname}: expected 4-D array, got {arr.shape}")
            N, C, H, W = arr.shape
            if H != W or H & (H - 1):
                raise ValueError(f"{fname}: H=W must be a power of 2; got {H}×{W}")

            self.data.append(torch.from_numpy(arr))       # share memory
            self.labels.append(torch.full((N,), k, dtype=torch.long))
            print(f"[Dataset] loaded {fname:>15}: {N} samples  →  branch {k}")

        self.data   = torch.cat(self.data,   dim=0)        # (ΣN, C, H, W)
        self.labels = torch.cat(self.labels, dim=0)        # (ΣN,)

    def __len__(self):  return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        k = self.labels[idx] if self.labels.numel() else None
        return (x, k) if k is not None else (x,)

# --------------------------------------------------------------------------
# CLI Parser
# --------------------------------------------------------------------------
def build_option_parser():
    p = argparse.ArgumentParser(
        "RES-GANomaly simple trainer",
        fromfile_prefix_chars='@'
    )
    p.add_argument("--data_path", type=str,
                default="ResData/wavefake128_2048split/train/fake",
                help="Either a .npy file or a directory of cluster*.npy")
    #p.add_argument("--data_root", type=str, default="ResData")
    p.add_argument("--outf", type=str, default="output_ResMGAN")
    p.add_argument("--name", type=str, default="Fi128z100")
    p.add_argument("--n_branches", type=int, default=None,
               help="Number of decoder branches; "
                    "if omitted and data_path is a dir, auto-infers.")
    p.add_argument("--batchsize", type=int, default=45)
    p.add_argument("--isize", type=int, default=128)
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=128)
    p.add_argument("--extra_res", type=int, default=4,
               help="Extra ResidualSEBlocks per down/upsampling stage")
    p.add_argument("--lr_g", type=float, default=1e-3)
    p.add_argument("--lr_d", type=float, default=1e-3)
    p.add_argument("--lr_cls", type=float, default=1e-3)
    p.add_argument("--niter", type=int, default=3000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ngpu",      type=int, default=1, help="Number of GPUs to use (for DataParallel)")
    p.add_argument("--metric", type=str, default="roc")
    p.add_argument("--tb_freq", type=int, default=10)  # batches between TB writes
    p.add_argument("--manualseed", type=int, default=-1)
    p.add_argument("--w_adv", type=float, default=1.0,  help="Weight on adversarial loss (Eq.12)")
    p.add_argument("--w_con", type=float, default=20.0, help="Weight on latent consistency loss")
    p.add_argument("--w_enc", type=float, default=1.0,  help="Weight on reconstruction loss")
    p.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient-penalty coefficient")
    p.add_argument("--n_critic", type=int, default=5,   help="D updates per G update")
    p.add_argument("--netg_ckpt", type=str, default=None,
                   help="Path to a pretrained generator (.pth).")
    p.add_argument("--netd_ckpt", type=str, default=None,
                   help="Path to a pretrained discriminator (.pth).")
    # p.add_argument("--netg_ckpt", type=str, default="output_vanillaResGAN/ResGanNormRerun/checkpoints/netG_epoch100.pth",
    #                help="Path to a pretrained generator (.pth).")
    # p.add_argument("--netd_ckpt", type=str, default="output_vanillaResGAN/ResGanNormRerun/checkpoints/netD_epoch100.pth",
    #                help="Path to a pretrained discriminator (.pth).")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint tag to resume from, e.g. 'latest' or 'epoch20'.")
    return p

def _resolve_ckpt_path(opt, tag: str) -> str:
    """
    Turn 'latest', 'best', 'epoch120', or an explicit *.pth path
    into a concrete path on disk.
    """
    if tag.endswith(".pth") or os.path.sep in tag:
        return tag  # user gave an explicit file/path

    ckpt_dir = os.path.join(opt.outf, opt.name, "checkpoints")
    return os.path.join(ckpt_dir, f"checkpoint_{tag}.pth")

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    opt = build_option_parser().parse_args()

    # --------------- only training loader ----------------
    # NEW
    train_ds = ClusteredLogMelDataset(opt.data_path)
    if opt.n_branches is None:
        max_k = int(train_ds.labels.max().item()) if hasattr(train_ds, "labels") else 0
        opt.n_branches = max_k + 1
        print(f"[Init] Auto-set n_branches = {opt.n_branches}")

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=15,
        pin_memory=True,
    )
    dataloader = {"train": train_loader}

    # --------------- model  ----------------
    model = RES_MGanomaly(opt, dataloader)
    if opt.resume:
        ckpt_path = _resolve_ckpt_path(opt, opt.resume)
        try:
            model.load(ckpt_path, strict=True)
        except FileNotFoundError as e:
            print(f"[!] Resume failed: {e}.  Starting from scratch.")
    print(f"[Params] netG       : {count_params(model.netg):,} parameters")
    print(f"[Params] netD       : {count_params(model.netd):,} parameters")
    print(f"[Params] classifier : {count_params(model.branch_clf):,} parameters")
    model.train_periodic_save()


if __name__ == "__main__":
    main()