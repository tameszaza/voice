#!/usr/bin/env python3
"""
split_mel_mfcc.py

Scan your source for `*/mel/*.npy` or `*/mel/stacked.npy`,
randomly select a fraction to move into a test directory,
and leave the remaining train data behind in `src`.

Usage:
    python split_mel_mfcc.py \
      --src /path/to/data_256 \
      --dst /path/to/data_256_test \
      --ratio 0.2 \
      --seed 42
"""
import numpy as np
import argparse
import random
import shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Move a fraction of mel/mfcc data into a test folder, "
                    "leaving the rest as train in-place."
    )
    p.add_argument("--src",   "-s", required=True, type=Path,
                   help="Root of your original dataset")
    p.add_argument("--dst",   "-d", required=True, type=Path,
                   help="Where the test subset will be moved")
    p.add_argument("--ratio", "-r", type=float, default=0.2,
                   help="Fraction reserved for test (default 0.2)")
    p.add_argument("--seed",  type=int,   default=42,
                   help="Random seed (default 42)")
    p.add_argument("--ext",   type=str,   default=".npy",
                   help="File extension (default .npy)")
    return p.parse_args()

def main():
    args     = parse_args()
    src      = args.src.resolve()
    dst      = args.dst.resolve()
    ratio    = args.ratio
    random.seed(args.seed)
    ext      = args.ext

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if src == dst:
        raise ValueError("Source and destination must differ")

    # Map each class-folder parent -> list of mel files
    mel_by_parent = {}
    for mel in src.rglob(f"*/mel/*{ext}"):
        parent = mel.parent.parent
        mel_by_parent.setdefault(parent, []).append(mel)

    moved = 0
    total = 0

    for parent, mel_list in mel_by_parent.items():
        rel = parent.relative_to(src)
        mel_stack  = parent / "mel"  / f"stacked{ext}"
        mfcc_stack = parent / "mfcc" / f"stacked{ext}"

        # --- STACKED CASE ---
                # --- STACKED CASE ---
        if mel_stack.exists() and mfcc_stack.exists():
            arr_m = np.load(mel_stack)
            arr_f = np.load(mfcc_stack)
            n     = arr_m.shape[0]
            total += n

            idxs = list(range(n))
            random.shuffle(idxs)
            cut = int(n * ratio)
            test_idxs  = idxs[:cut]
            train_idxs = idxs[cut:]

            # 1) write test stacked.npy
            dst_mel_stack  = dst  / rel / "mel"  / f"stacked{ext}"
            dst_mfcc_stack = dst  / rel / "mfcc" / f"stacked{ext}"
            dst_mel_stack .parent.mkdir(parents=True, exist_ok=True)
            dst_mfcc_stack.parent.mkdir(parents=True, exist_ok=True)

            test_m = arr_m[test_idxs]    # shape (n_test, H, W)
            test_f = arr_f[test_idxs]
            np.save(str(dst_mel_stack),  test_m)
            np.save(str(dst_mfcc_stack), test_f)
            moved += len(test_idxs)

            # 2) overwrite original with train stacked.npy
            train_m = arr_m[train_idxs]
            train_f = arr_f[train_idxs]
            np.save(mel_stack,  train_m)
            np.save(mfcc_stack, train_f)

            continue


        # --- PER-FILE CASE ---
        files = list(mel_list)
        total += len(files)
        random.shuffle(files)
        cut = int(len(files) * ratio)
        test_files = files[:cut]

        for mel_path in test_files:
            # move mel
            rel_m = mel_path.relative_to(src)
            dst_m = dst / rel_m
            dst_m.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(mel_path), str(dst_m))

            # move matching mfcc
            mfcc_path = parent / "mfcc" / mel_path.name
            if mfcc_path.exists():
                rel_f = mfcc_path.relative_to(src)
                dst_f = dst / rel_f
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(mfcc_path), str(dst_f))
            moved += 1

    print(f"→ Total samples       : {total}")
    print(f"→ Moved to test       : {moved}")
    print(f"→ Remaining in train  : {total - moved}")
    print(f"Test set now lives in: {dst}")

if __name__ == "__main__":
    main()
