#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import your helper dataset classes; adjust the import path if needed
from helper import (
    SingleNpyDataset,
    RealFakeNpyDataset,
    MultiNpyFilesDataset,
    MultiFeatureDirectoryDataset,
)

def plot_tensor_sample(x: torch.Tensor, label: int, idx: int, out_dir: str = None):
    """
    Plot a single tensor sample x with its label.
    If x is 2D (H, W), treat it as 1 channel.
    If x is 3D (C, H, W), plot each channel in its own subplot.
    """
    x = x.numpy()

    # if 2D, turn into (1, H, W)
    if x.ndim == 2:
        x = x[np.newaxis, ...]

    # now x has shape (C, H, W)
    n_ch = x.shape[0]
    fig, axes = plt.subplots(nrows=n_ch, figsize=(4 * n_ch, 4))

    # ensure axes is a list we can index
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ch, ax in enumerate(axes):
        im = ax.imshow(x[ch], origin='lower', aspect='auto')
        ax.set_title(f"Sample {idx}  Label {label}  Ch {ch}")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"sample_{idx}.png"))
    else:
        plt.show()

    plt.close(fig)


def visualize_dataset(dataset, num_samples: int, batch_size: int, out_dir: str = None, multifile_mode: bool = False):
    """
    Iterate over the dataset (via DataLoader) and plot the first num_samples.
    If multifile_mode is True, plot num_samples random samples per label.
    """
    if not multifile_mode:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        plotted = 0
        for batch in loader:
            # batch can be (x, label) or x only
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_batch, labels = batch
            else:
                x_batch = batch
                labels = [None] * x_batch.size(0)

            for i in range(x_batch.size(0)):
                if plotted >= num_samples:
                    return
                plot_tensor_sample(x_batch[i], labels[i], plotted, out_dir)
                plotted += 1
    else:
        # Group indices by label
        from collections import defaultdict
        import random
        label_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            label_to_indices[label].append(idx)
        for label, indices in label_to_indices.items():
            chosen = random.sample(indices, min(num_samples, len(indices)))
            for i, idx in enumerate(chosen):
                x, _ = dataset[idx]
                plot_tensor_sample(x, label, f"{label}_{i}", out_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Visualize samples from various .npy-based datasets."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="Which dataset type to visualize")

    # Single .npy file
    p1 = subparsers.add_parser("single", help="Visualize a single .npy file")
    p1.add_argument("npy_file", help="Path to the .npy file")
    p1.add_argument("--label", type=int, default=None,
                    help="Optional label for all samples")
    p1.add_argument("--max_samples", type=int, default=None,
                    help="Limit to first N samples in the file")

    # Real vs fake
    p2 = subparsers.add_parser("realfake", help="Visualize real vs fake .npy pair")
    p2.add_argument("real_npy", help="Path to the real .npy file")
    p2.add_argument("fake_npy", help="Path to the fake .npy file")
    p2.add_argument("--max_per_class", type=int, default=None,
                    help="Limit to first N per class")

    # Multi-file folder
    p3 = subparsers.add_parser("multifile", help="Visualize directory of .npy files")
    p3.add_argument("dir_path", help="Directory containing multiple .npy files")
    p3.add_argument("--max_per_file", type=int, default=None,
                    help="Limit to first N samples per file")

    # Multi-feature directory
    p4 = subparsers.add_parser("multifeature", help="Visualize directory with mel/mfcc subfolders")
    p4.add_argument("data_root", help="Root directory with class subfolders")
    p4.add_argument("--no_mel", dest="use_mel", action="store_false",
                    help="Do not load mel features")
    p4.add_argument("--no_mfcc", dest="use_mfcc", action="store_false",
                    help="Do not load mfcc features")

    # Common args
    for sub in (p1, p2, p3, p4):
        sub.add_argument("--num_samples", type=int, default=10,
                         help="Number of samples to visualize (default 5)")
        sub.add_argument("--batch_size", type=int, default=2,
                         help="Batch size for DataLoader (default 2)")
        sub.add_argument("--out_dir", type=str, default=None,
                         help="If set, save plots as PNGs in this directory")

    args = parser.parse_args()

    if args.mode == "single":
        ds = SingleNpyDataset(args.npy_file, label=args.label,
                              max_samples=args.max_samples)
        multifile_mode = False
    elif args.mode == "realfake":
        ds = RealFakeNpyDataset(args.real_npy, args.fake_npy,
                                max_samples_per_class=args.max_per_class)
        multifile_mode = False
    elif args.mode == "multifile":
        ds = MultiNpyFilesDataset(args.dir_path,
                                  max_samples_per_file=args.max_per_file)
        multifile_mode = True
    elif args.mode == "multifeature":
        ds = MultiFeatureDirectoryDataset(
            args.data_root, use_mel=args.use_mel, use_mfcc=args.use_mfcc
        )
        multifile_mode = False
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    visualize_dataset(ds, num_samples=args.num_samples,
                      batch_size=args.batch_size,
                      out_dir=args.out_dir,
                      multifile_mode=multifile_mode)

if __name__ == "__main__":
    main()
