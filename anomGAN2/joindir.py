#!/usr/bin/env python3
"""
merge_features.py

Scan a root directory containing multiple model subdirectories,
each of which may have 'mel' and/or 'mfcc' subfolders, and
merge all their .npy files into two single stacked files:

    <output_dir>/mel/stacked.npy
    <output_dir>/mfcc/stacked.npy

Each of those will be a single 3-D array of shape (N, H, W),
where N is the total number of slices across all models.
"""

import os
import argparse
import numpy as np

def merge_and_stack(root_dir: str, output_dir: str, feature: str):
    """
    Walk every <root_dir>/<model>/<feature> folder, load all .npy data
    (including any pre-existing stacked.npy), concatenate into one array,
    and save as <output_dir>/<feature>/stacked.npy.
    """
    dest_folder = os.path.join(output_dir, feature)
    os.makedirs(dest_folder, exist_ok=True)

    all_arrays = []
    with os.scandir(root_dir) as models:
        for model_entry in models:
            if not model_entry.is_dir():
                continue
            feat_path = os.path.join(model_entry.path, feature)
            if not os.path.isdir(feat_path):
                continue

            # 1) If a pre-stacked file exists, load its entire array
            stacked_file = os.path.join(feat_path, "stacked.npy")
            if os.path.isfile(stacked_file):
                arr = np.load(stacked_file, mmap_mode="r")
                if arr.ndim != 3:
                    raise ValueError(f"{stacked_file} must be 3D, got {arr.shape}")
                all_arrays.append(arr)
                print(f"Loaded stacked from {stacked_file} with shape {arr.shape}")
                continue

            # 2) Otherwise, load every individual .npy as a 2D slice
            with os.scandir(feat_path) as files:
                for file_entry in files:
                    if not file_entry.name.endswith(".npy"):
                        continue
                    arr = np.load(file_entry.path)
                    if arr.ndim != 2:
                        raise ValueError(f"{file_entry.path} must be 2D, got {arr.shape}")
                    # promote to 3D with leading axis
                    all_arrays.append(arr[np.newaxis, ...])
                    print(f"Loaded slice {file_entry.path} with shape {arr.shape}")

    if not all_arrays:
        print(f"No data found under any '{feature}' folder in {root_dir}.")
        return

    # concatenate along first axis
    stacked = np.concatenate(all_arrays, axis=0)  # shape (N, H, W)
    out_path = os.path.join(dest_folder, "stacked.npy")
    np.save(out_path, stacked)
    print(f"\nSaved merged stacked array for '{feature}' with shape {stacked.shape} → {out_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge 'mel' and 'mfcc' features from multiple models "
                    "into single stacked .npy files."
    )
    parser.add_argument(
        "root_dir",
        help="Directory containing model subdirectories"
    )
    parser.add_argument(
        "output_dir",
        help="Directory where you want mel/stacked.npy and mfcc/stacked.npy"
    )
    args = parser.parse_args()

    # Merge & stack mel
    merge_and_stack(args.root_dir, args.output_dir, "mel")

    # Merge & stack mfcc
    merge_and_stack(args.root_dir, args.output_dir, "mfcc")


if __name__ == "__main__":
    main()
