#!/usr/bin/env python3
"""
stack_npy.py

Recursively stack all 2D .npy files in each subdirectory into a single 3D array.
Original .npy files are removed and replaced by stacked.npy.

Usage:
    python stack_npy.py /path/to/root_directory
"""

import os
import numpy as np
import argparse

def process_directory(dir_path):
    # find all .npy files in this directory
    npy_files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
    if not npy_files:
        return

    arrays = []
    for fname in npy_files:
        path = os.path.join(dir_path, fname)
        try:
            arr = np.load(path)
        except Exception as e:
            print(f"  • Could not load {fname}: {e}")
            continue

        if arr.ndim != 2:
            print(f"  • Skipping {fname}: expected 2D array, got {arr.ndim}D")
            continue

        arrays.append(arr)

    if not arrays:
        print(f"[{dir_path}] no valid 2D arrays to stack")
        return

    # check that all shapes match
    first_shape = arrays[0].shape
    if any(arr.shape != first_shape for arr in arrays):
        shapes = [arr.shape for arr in arrays]
        raise ValueError(f"[{dir_path}] mismatched shapes: {shapes}")

    # stack into a 3D array: (count, height, width)
    stacked = np.stack(arrays, axis=0)

    # remove original files
    for fname in npy_files:
        try:
            os.remove(os.path.join(dir_path, fname))
        except Exception as e:
            print(f"  • Warning: failed to delete {fname}: {e}")

    # save the new stacked file
    out_path = os.path.join(dir_path, 'stacked.npy')
    np.save(out_path, stacked)
    print(f"[{dir_path}] saved stacked array with shape {stacked.shape} as {out_path}")

def traverse_and_process(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        process_directory(dirpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stack all 2D .npy files in each folder into a single 3D .npy file.'
    )
    parser.add_argument('root_dir',
                        help='Root directory to start searching for .npy files.')
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        raise NotADirectoryError(f"Root directory not found: {args.root_dir}")

    traverse_and_process(args.root_dir)
