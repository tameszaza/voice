from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# local imports -----------------------------------------------------------------
from data import AggregatedLogMelDataset, LogMelDataset  # your dataset file
from model import Ganomaly                                # created earlier


#!/usr/bin/env python
"""
train.py — single-decoder GANomaly baseline (BCE-with-logits)

• Accept **either** --data_file (aggregated .npy tensor or dir of such)  
  **or** --data_root (folder of individual (C,isize,isize) .npy specs).  
• Handles TB logging / checkpoints exactly as the updated model expects.
"""
# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
#  CLI helper
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train single-decoder GANomaly")

    # ---------- data ----------
    g_data = p.add_argument_group("data")
    g_data.add_argument("--data_file", type=str, default="ResData/wavefake128_2048split/train/fake",
                        help="aggregated (N,C,H,W) .npy or dir of such")
    g_data.add_argument("--data_root", type=str, default=None,
                        help="folder of individual (C,H,W) .npy files")
    g_data.add_argument("--isize", type=int, default=128, help="image H×W")
    g_data.add_argument("--nc",    type=int, default=1,   help="#channels")

    # ---------- model ----------
    g_model = p.add_argument_group("model")
    g_model.add_argument("--nz",          type=int, default=128, help="latent dim")
    g_model.add_argument("--ngf",         type=int, default=64, help="#gen filters")
    g_model.add_argument("--ndf",         type=int, default=64, help="#disc filters")
    g_model.add_argument("--extralayers", type=int, default=0,
                         help="extra conv layers in G/D (DCGAN style)")
    g_model.add_argument("--ngpu",        type=int, default=1,
                         help="#GPUs for DataParallel (set 1 for single-GPU)")

    # ---------- training ----------
    g_train = p.add_argument_group("training")
    g_train.add_argument("--batchSize", type=int, default=32)
    g_train.add_argument("--niter",     type=int, default=3000, help="#epochs")
    g_train.add_argument("--lr",        type=float, default=2e-3)
    g_train.add_argument("--beta1",     type=float, default=0.5)
    g_train.add_argument("--tb_freq",   type=int, default=200,
                         help="TensorBoard scalar period (iters)")
    g_train.add_argument("--save_freq", type=int, default=10,
                         help="epochs between light ckpts")

    # ---------- misc ----------
    g_misc = p.add_argument_group("misc")
    g_misc.add_argument("--outf",   type=str, default="./output_tradgan_tuned",
                        help="root for logs/ckpts")
    g_misc.add_argument("--name",   type=str, default="128singleFakeWavefake")
    g_misc.add_argument("--device", type=str, default="cuda:0")
    g_misc.add_argument("--manualseed", type=int, default=None)
    g_misc.add_argument("--resume",     type=str, default=None,
                        help="resume from full ckpt")
    g_misc.add_argument("--load_light", type=str, default=None,
                        help="initialise weights from light ckpt")
    
    g_model.add_argument("--w_adv", type=float, default=1.0,
                     help="weight W_adv in Eq. 12")
    g_model.add_argument("--w_con", type=float, default=30.0,
                        help="weight W_con in Eq. 12")
    g_model.add_argument("--w_enc", type=float, default=1.0,
                        help="weight W_enc in Eq. 12")
    g_model.add_argument("--lambdaGP", type=float, default=10.0,)
    g_train.add_argument("--n_critic",  type=int, default=5,
                         help="#discriminator updates per generator update")


    return p


# ─────────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────────


def seed_everything(seed: int | None):
    if seed is None:
        seed = random.randint(1, 10_000)
    print(f"[seed] {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloader(opt):
    if (opt.data_file is None) == (opt.data_root is None):
        raise ValueError("Provide *exactly one* of --data_file or --data_root")

    if opt.data_file is not None:
        # path may be a single file or a directory of aggregated tensors
        if os.path.isdir(opt.data_file):
            train_ds = AggregatedLogMelDataset(opt.data_file, isize=opt.isize)  # ← FIX
        else:
            train_ds = AggregatedLogMelDataset(opt.data_file, isize=opt.isize)  # ← FIX
    else:
        train_ds = LogMelDataset(opt.data_root, isize=opt.isize)                # ← FIX

    return DataLoader(
        train_ds,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )



# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    opt = build_parser().parse_args()
    seed_everything(opt.manualseed)

    # data ----------------------------------------------------------------
    train_loader = make_dataloader(opt)
    dl_dict = {"train": [train_loader]}  # what Ganomaly.fit() expects

    # model ---------------------------------------------------------------
    model = Ganomaly(opt, dl_dict)

    # resume / lightweight init ------------------------------------------
    if opt.resume:
        model.load(opt.resume)
    elif opt.load_light:
        model.load(opt.load_light, strict=False)

    # train ---------------------------------------------------------------
    model.fit()


if __name__ == "__main__":
    main()

