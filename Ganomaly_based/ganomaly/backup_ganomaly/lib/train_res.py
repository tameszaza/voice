import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from RES_GANomaly_model import RES_Ganomaly
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class LogMelDataset(Dataset):
    """
    Loads .npy files containing (1,128,128) log-Mel spectrograms.
    If label is None → returns (spec,) for unsupervised training.
    Otherwise returns (spec, label) for evaluation.
    """

    def __init__(self, root_dir: str, label: int | None):
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ]
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx]).astype("float32")  # (1,128,128)
        spec_t = torch.from_numpy(spec)
        if self.label is None:
            return (spec_t,)  # unsupervised batch
        return (spec_t, torch.tensor(self.label, dtype=torch.long))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_option_parser():
    p = argparse.ArgumentParser(
        "RES-GANomaly trainer",
        fromfile_prefix_chars='@'        # ← this line enables @file.txt
    )
    p.add_argument("--data_root", type=str, default="ResData")
    p.add_argument("--outf", type=str, default="./output")
    p.add_argument("--name", type=str, default="RESGAN_run")
    p.add_argument("--batchsize", type=int, default=8)
    p.add_argument("--isize", type=int, default=128)
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--niter", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ngpu",      type=int, default=1, help="Number of GPUs to use (for DataParallel)")
    p.add_argument("--metric", type=str, default="roc")
    p.add_argument("--tb_freq", type=int, default=10)  # batches between TB writes
    p.add_argument("--manualseed", type=int, default=-1)
    p.add_argument("--w_adv", type=float, default=1.0,  help="Weight on adversarial loss (Eq.12)")
    p.add_argument("--w_con", type=float, default=50.0, help="Weight on latent consistency loss")
    p.add_argument("--w_enc", type=float, default=1.0,  help="Weight on reconstruction loss")
    p.add_argument("--lambda_gp", type=float, default=1.0, help="Gradient-penalty coefficient")
    p.add_argument("--n_critic", type=int, default=1,   help="D updates per G update")
    return p


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    opt = build_option_parser().parse_args()

    # ---------------- data loaders ----------------
    train_ds = LogMelDataset(
        os.path.join(opt.data_root, "log-mel-data-train"), label=None
    )
    real_ds = LogMelDataset(
        os.path.join(opt.data_root, "log-mel-eval", "real"), label=0
    )
    fake_ds = LogMelDataset(
        os.path.join(opt.data_root, "log-mel-eval", "fake"), label=1
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        ConcatDataset([real_ds, fake_ds]),
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=2,
    )

    dataloader = {"train": train_loader, "test": test_loader}

    # ---------------- model ----------------
    model = RES_Ganomaly(opt, dataloader)
    model.train()


if __name__ == "__main__":
    main()
