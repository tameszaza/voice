#!/usr/bin/env python3
import os
import shutil
import random
import argparse
import sys

def get_paired_samples(class_dir):
    """
    Return a sorted list of filenames (ending in .npy) that exist
    in both class_dir/mel/ and class_dir/mfcc/.
    """
    mel_dir  = os.path.join(class_dir, "mel")
    mfcc_dir = os.path.join(class_dir, "mfcc")

    if not os.path.isdir(mel_dir) or not os.path.isdir(mfcc_dir):
        sys.exit(f"Error: '{class_dir}' must contain 'mel/' and 'mfcc/' subfolders")

    mel_files  = {f for f in os.listdir(mel_dir)  if f.endswith(".npy")}
    mfcc_files = {f for f in os.listdir(mfcc_dir) if f.endswith(".npy")}
    paired     = sorted(mel_files & mfcc_files)

    if not paired:
        sys.exit(f"Error: No matching .npy filenames in '{mel_dir}' and '{mfcc_dir}'")

    if len(paired) < max(len(mel_files), len(mfcc_files)):
        print(f"Warning: only {len(paired)} paired samples will be used from '{class_dir}'")

    return paired

def copy_samples(sample_list, src_dir, dst_dir, label):
    """
    Copy each filename in sample_list from src_dir/{mel,mfcc} into
    dst_dir/label/{mel,mfcc}.
    """
    for fname in sample_list:
        for feat in ("mel", "mfcc"):
            src_path = os.path.join(src_dir, feat, fname)
            dst_sub  = os.path.join(dst_dir, label, feat)
            os.makedirs(dst_sub, exist_ok=True)
            dst_path = os.path.join(dst_sub, fname)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_path} → {dst_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Build a balanced eval set by copying samples and labeling them 'real'/'fake'"
    )
    parser.add_argument("real_dir",
                        help="Folder of real-class samples (must contain mel/ and mfcc/)")
    parser.add_argument("fake_dir",
                        help="Folder of fake-class samples (must contain mel/ and mfcc/)")
    parser.add_argument("eval_dir",
                        help="Destination folder for the balanced eval set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible selection")
    args = parser.parse_args()

    # find paired samples in each class
    real_samples = get_paired_samples(args.real_dir)
    fake_samples = get_paired_samples(args.fake_dir)

    # determine smaller class
    if len(real_samples) <= len(fake_samples):
        small_label, small_list, small_dir = "real", real_samples, args.real_dir
        large_label, large_list, large_dir = "fake", fake_samples, args.fake_dir
    else:
        small_label, small_list, small_dir = "fake", fake_samples, args.fake_dir
        large_label, large_list, large_dir = "real", real_samples, args.real_dir

    n_small = len(small_list)
    print(f"{small_label.capitalize()} class has {n_small} samples; "
          f"{large_label.capitalize()} has {len(large_list)}")

    # downsample the larger class
    random.seed(args.seed)
    selected_large = random.sample(large_list, n_small)
    print(f"Selected {n_small} random samples from the {large_label} class")

    # copy the balanced sets into eval_dir
    copy_samples(small_list, small_dir, args.eval_dir, small_label)
    copy_samples(selected_large, large_dir, args.eval_dir, large_label)

    print("\nDone. '{0}' now contains:".format(args.eval_dir))
    print(f"  {small_label}/mel   ({n_small} files)")
    print(f"  {small_label}/mfcc  ({n_small} files)")
    print(f"  {large_label}/mel   ({n_small} files)")
    print(f"  {large_label}/mfcc  ({n_small} files)")

if __name__ == "__main__":
    main()
