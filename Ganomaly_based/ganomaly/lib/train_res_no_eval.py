import os
import argparse
import numpy as np
torch_import = True
import torch
from torch.utils.data import Dataset, DataLoader

from RES_GANomaly_model import RES_Ganomaly
# Enables detailed gradient anomaly detection if needed
#torch.autograd.set_detect_anomaly(True)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Dataset that reads ONE aggregated .npy file
# --------------------------------------------------------------------------
class AggregatedLogMelDataset(Dataset):
    """
    Loads a single .npy file shaped (N, C, 128, 128) **once** and
    serves individual tensors on demand.
    """
    def __init__(self, npy_path: str):
        arr = np.load(npy_path).astype("float32")          # (N,C,128,128)
        if arr.ndim != 4 or arr.shape[2:] != (128, 128):
            raise ValueError(f"Unexpected array shape {arr.shape}")
        self.data = torch.from_numpy(arr)                  # keeps a view
        del arr                                            # free NumPy copy

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx],)                           # keep tuple form


class LogMelDataset(Dataset):
    """
    Loads .npy files containing (1,128,128) log-Mel spectrograms.
    Returns just the spectrogram for unsupervised training.
    """

    def __init__(self, root_dir: str):
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx]).astype("float32")  # (1,128,128)
        spec_t = torch.from_numpy(spec)
        return (spec_t,)


# --------------------------------------------------------------------------
# CLI Parser
# --------------------------------------------------------------------------
def build_option_parser():
    p = argparse.ArgumentParser(
        "RES-GANomaly simple trainer",
        fromfile_prefix_chars='@'
    )
    p.add_argument("--data_file", type=str,
                   default="ResData/wavefake32_split/train/real.npy",
                   help="Path to aggregated .npy file shaped (N,C,128,128)")
    p.add_argument("--data_root", type=str, default="ResData")
    p.add_argument("--outf", type=str, default="./output_vanillaResGAN")
    p.add_argument("--name", type=str, default="FirstTime32")
    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--isize", type=int, default=32)
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=128)
    p.add_argument("--lr", type=float, default=1*1e-5)
    p.add_argument("--niter", type=int, default=3000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ngpu",      type=int, default=1, help="Number of GPUs to use (for DataParallel)")
    p.add_argument("--metric", type=str, default="roc")
    p.add_argument("--tb_freq", type=int, default=10)  # batches between TB writes
    p.add_argument("--manualseed", type=int, default=-1)
    p.add_argument("--w_adv", type=float, default=1.0,  help="Weight on adversarial loss (Eq.12)")
    p.add_argument("--w_con", type=float, default=50.0, help="Weight on latent consistency loss")
    p.add_argument("--w_enc", type=float, default=1.0,  help="Weight on reconstruction loss")
    p.add_argument("--lambda_gp", type=float, default=1.0, help="Gradient-penalty coefficient")
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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    opt = build_option_parser().parse_args()

    # --------------- only training loader ----------------
    train_ds = AggregatedLogMelDataset(opt.data_file)
    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dataloader = {"train": train_loader}

    # --------------- model  ----------------
    model = RES_Ganomaly(opt, dataloader)
    if opt.resume:
      try:
        model.load(tag=opt.resume)
      except FileNotFoundError as e:
        print(f"[!] Resume failed: {e}.  Starting from scratch.")
    # uses train_periodic_save to skip evaluation and save every 10 epochs
    model.train_periodic_save()


if __name__ == "__main__":
    main()