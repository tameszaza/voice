#!/usr/bin/env python3
"""
split_mel_mfcc.py

Recursively scan your source folder for `*/mel/*.npy`, randomly
select 20% of those filenames, and move both the mel and its
matching mfcc file to a separate test directory—preserving
the entire folder hierarchy.

Usage:
    python split_mel_mfcc.py \
        --src /path/to/data_256 \
        --dst /path/to/data_256_test \
        --ratio 0.2 \
        --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Split mel/mfcc .npy files into train/test sets by filename."
    )
    p.add_argument(
        "--src", "-s",
        required=True,
        type=Path,
        help="Root of your original dataset (e.g. data_256)."
    )
    p.add_argument(
        "--dst", "-d",
        required=True,
        type=Path,
        help="Where your test subset will be moved (e.g. data_256_test)."
    )
    p.add_argument(
        "--ratio", "-r",
        type=float,
        default=0.2,
        help="Fraction to reserve for test (default 0.2 = 20%%)."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default 42)."
    )
    p.add_argument(
        "--ext",
        type=str,
        default=".npy",
        help="File extension (default .npy)."
    )
    return p.parse_args()

def main():
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    ratio = args.ratio
    seed = args.seed
    ext = args.ext

    if not src.exists():
        raise FileNotFoundError(f"Source folder '{src}' not found.")
    if src == dst:
        raise ValueError("Source and destination must be different paths.")

    # 1) Find all mel files
    mel_files = list(src.rglob(f"*/mel/*{ext}"))
    if not mel_files:
        print(f"No '{ext}' files under any 'mel' folder in {src}.")
        return

    # 2) Shuffle & pick test filenames (by file *name*, not full path)
    random.seed(seed)
    random.shuffle(mel_files)
    n_test = int(len(mel_files) * ratio)
    test_mel_files = mel_files[:n_test]

    moved = 0
    for mel_path in test_mel_files:
        # Compute where it sits under src
        rel_mel = mel_path.relative_to(src)
        dst_mel = dst / rel_mel

        # 3) Move mel file
        dst_mel.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mel_path), str(dst_mel))

        # 4) Locate & move the matching mfcc file
        #    same parent.parent but in 'mfcc' subfolder
        model_dir = mel_path.parent.parent
        mfcc_path = model_dir / "mfcc" / mel_path.name
        if mfcc_path.exists():
            rel_mfcc = mfcc_path.relative_to(src)
            dst_mfcc = dst / rel_mfcc
            dst_mfcc.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(mfcc_path), str(dst_mfcc))
        else:
            print(f"[WARN] No mfcc file for '{mel_path.name}' at expected location:\n  {mfcc_path}")

        moved += 1

    total = len(mel_files)
    print(f"→ Total mel files found : {total}")
    print(f"→ Moved to test       : {moved}")
    print(f"→ Remaining in train  : {total - moved}")
    print(f"Test set now lives in: {dst}")

if __name__ == "__main__":
    main()
