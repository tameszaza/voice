#!/usr/bin/env python3
"""
merge_features.py

Scan a root directory containing multiple model subdirectories,
each of which may have 'mel' and/or 'mfcc' subfolders, and
merge all their files into two consolidated folders:

    <output_dir>/mel
    <output_dir>/mfcc

Files are copied (not moved).  If two models have the same
filename, the script will rename them as
    <model_name>_<original_filename>
so nothing gets silently overwritten.
"""

import os
import shutil
import argparse

def merge_feature_folder(root_dir: str, output_dir: str, feature: str):
    """
    Merge all files from each <root_dir>/<model>/<feature>
    folder into <output_dir>/<feature>.
    """
    dest = os.path.join(output_dir, feature)
    os.makedirs(dest, exist_ok=True)

    for model_name in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_name)
        feat_path = os.path.join(model_path, feature)

        if not os.path.isdir(feat_path):
            # skip models that do not have this feature folder
            continue

        for fname in os.listdir(feat_path):
            src = os.path.join(feat_path, fname)
            if not os.path.isfile(src):
                continue

            # prefix with model name to avoid collisions
            dst_name = f"{model_name}_{fname}"
            dst = os.path.join(dest, dst_name)

            shutil.copy2(src, dst)
            print(f"copied {src} → {dst}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge 'mel' and 'mfcc' folders from multiple models"
    )
    parser.add_argument(
        "root_dir",
        help="Path to the directory that contains your model subdirectories"
    )
    parser.add_argument(
        "output_dir",
        help="Path where merged 'mel' and 'mfcc' folders will be created"
    )
    args = parser.parse_args()

    # merge mel files
    merge_feature_folder(args.root_dir, args.output_dir, "mel")
    # merge mfcc files
    merge_feature_folder(args.root_dir, args.output_dir, "mfcc")

if __name__ == "__main__":
    main()
