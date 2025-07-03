import os
import argparse
import glob
import numpy as np
import pandas as pd
import shutil
import sys

# Import inference and set_seed from inference_r.py
sys.path.insert(0, os.path.dirname(__file__))
from inference_r import inference, set_seed

def find_checkpoints(model_dir, prefix="G_", suffix=".pt"):
    """Return sorted list of checkpoint suffixes (e.g. ['20', '40', ...'])"""
    pattern = os.path.join(model_dir, f"{prefix}*{suffix}")
    files = glob.glob(pattern)
    suffixes = []
    for f in files:
        base = os.path.basename(f)
        # e.g. G_20.pt -> 20
        if base.startswith(prefix) and base.endswith(suffix):
            s = base[len(prefix):-len(suffix)]
            suffixes.append(s)
    # Sort numerically if possible
    try:
        suffixes = sorted(suffixes, key=lambda x: int(x))
    except Exception:
        suffixes = sorted(suffixes)
    return suffixes

def find_encoder_checkpoints(model_dir, prefix="E_xzx_", suffix=".pt"):
    """Return sorted list of encoder checkpoint filenames (e.g. ['E_xzx_100.pt', ...'])"""
    pattern = os.path.join(model_dir, f"{prefix}*{suffix}")
    files = glob.glob(pattern)
    files = [os.path.basename(f) for f in files]
    # Sort numerically if possible
    try:
        files = sorted(files, key=lambda x: int(x.replace(prefix, "").replace(suffix, "")))
    except Exception:
        files = sorted(files)
    return files

def main():
    parser = argparse.ArgumentParser(
        description="Automate inference over all checkpoints and report best AUC"
    )
    # All arguments from inference_r.py except --ckpt and --out_dir
    parser.add_argument("--real_data_root",  required=True)
    parser.add_argument("--anomaly_data_dir",required=True)
    parser.add_argument("--model_dir",       required=True)
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--z_dim",           type=int, default=128)
    parser.add_argument("--n_clusters",      type=int, default=7)
    parser.add_argument("--base_channels",   type=int, default=32)
    parser.add_argument("--n_layers",        type=int, default=3)
    parser.add_argument("--alpha",           type=float, default=0.9)
    parser.add_argument("--n_thresholds",    type=int, default=200)
    parser.add_argument("--use_mel",         action="store_true")
    parser.add_argument("--use_mfcc",        action="store_true")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--max_samples_per_class", type=int, default=3000)
    parser.add_argument("--bypass_classifier", action="store_true")
    parser.add_argument("--anom_noise_std", type=float, default=0.0)
    parser.add_argument("--out_dir", default="results_eval_all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_all", action="store_true", help="Keep all result folders (default: only best)")
    # Add compare_mode and encoder_pre
    parser.add_argument("--compare_mode", action="store_true",
                        help="If set, sweep all encoder checkpoints (E_xzx_*.pt) and compare using fixed G/D/C checkpoint")
    parser.add_argument("--encoder_pre", type=str, default="E_xzx_",
                        help="Prefix for encoder checkpoints to compare in compare_mode")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint suffix for G/D/C/C (required for compare_mode)")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.compare_mode:
        # Check that --ckpt is provided
        if not args.ckpt:
            print("Error: --ckpt must be specified in --compare_mode")
            return
        # Find all encoder checkpoints
        encoder_ckpts = find_encoder_checkpoints(args.model_dir, prefix=args.encoder_pre)
        if not encoder_ckpts:
            print(f"No encoder checkpoints found in {args.model_dir} with prefix {args.encoder_pre}")
            return

        results = []
        best_auc = -1
        best_encoder = None
        best_out_dir = None

        for encoder_ckpt in encoder_ckpts:
            encoder_name = os.path.splitext(encoder_ckpt)[0]
            out_dir_ck = os.path.join(args.out_dir, f"encoder_{encoder_name}")
            if os.path.exists(out_dir_ck):
                print(f"Skipping encoder {encoder_ckpt} (results already exist at {out_dir_ck})")
                # Try to read AUC from report.txt if present
                report_path = os.path.join(out_dir_ck, "report.txt")
                auc = None
                if os.path.exists(report_path):
                    with open(report_path) as f:
                        for line in f:
                            if line.strip().startswith("AUC"):
                                try:
                                    auc = float(line.strip().split(":")[1])
                                except Exception:
                                    pass
                                break
                if auc is not None:
                    results.append({"encoder": encoder_ckpt, "auc": auc, "out_dir": out_dir_ck})
                    print(f"Encoder {encoder_ckpt}: AUC={auc:.4f}")
                    if auc > best_auc:
                        best_auc = auc
                        best_encoder = encoder_ckpt
                        best_out_dir = out_dir_ck
                else:
                    print(f"Could not find AUC in {report_path}")
                continue

            print(f"\n=== Evaluating encoder: {encoder_ckpt} ===")
            os.makedirs(out_dir_ck, exist_ok=True)

            args_ck = argparse.Namespace(**vars(args))
            args_ck.ckpt = args.ckpt
            args_ck.out_dir = out_dir_ck
            args_ck.encoder_xzx = encoder_ckpt

            try:
                inference(args_ck)
            except Exception as e:
                print(f"Error running inference for encoder {encoder_ckpt}: {e}")
                continue

            # Read AUC from report.txt
            report_path = os.path.join(out_dir_ck, "report.txt")
            auc = None
            if os.path.exists(report_path):
                with open(report_path) as f:
                    for line in f:
                        if line.strip().startswith("AUC"):
                            try:
                                auc = float(line.strip().split(":")[1])
                            except Exception:
                                pass
                            break
            if auc is not None:
                results.append({"encoder": encoder_ckpt, "auc": auc, "out_dir": out_dir_ck})
                print(f"Encoder {encoder_ckpt}: AUC={auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_encoder = encoder_ckpt
                    best_out_dir = out_dir_ck
            else:
                print(f"Could not find AUC in {report_path}")

        # Save summary CSV
        df = pd.DataFrame(results)
        summary_csv = os.path.join(args.out_dir, "auc_summary_encoders.csv")
        df.to_csv(summary_csv, index=False)
        print(f"\nAUC summary saved to {summary_csv}")

        if best_encoder is not None:
            print(f"\nBest encoder: {best_encoder} (AUC={best_auc:.4f})")
            print(f"Results in: {best_out_dir}")
            best_dir = os.path.join(args.out_dir, "best_encoder")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(best_out_dir, best_dir)
            print(f"Best encoder results copied to {best_dir}")
            if not args.keep_all:
                for r in results:
                    if r["out_dir"] != best_out_dir:
                        shutil.rmtree(r["out_dir"], ignore_errors=True)
        else:
            print("No valid AUC found for any encoder checkpoint.")
        return

    # Find all checkpoint suffixes
    ckpts = find_checkpoints(args.model_dir)
    if not ckpts:
        print(f"No checkpoints found in {args.model_dir}")
        return

    results = []
    best_auc = -1
    best_ckpt = None
    best_out_dir = None

    for ck in ckpts:
        out_dir_ck = os.path.join(args.out_dir, f"ckpt_{ck}")
        if os.path.exists(out_dir_ck):
            print(f"Skipping checkpoint {ck} (results already exist at {out_dir_ck})")
            # Try to read AUC from report.txt if present
            report_path = os.path.join(out_dir_ck, "report.txt")
            auc = None
            if os.path.exists(report_path):
                with open(report_path) as f:
                    for line in f:
                        if line.strip().startswith("AUC"):
                            try:
                                auc = float(line.strip().split(":")[1])
                            except Exception:
                                pass
                            break
            if auc is not None:
                results.append({"ckpt": ck, "auc": auc, "out_dir": out_dir_ck})
                print(f"Checkpoint {ck}: AUC={auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_ckpt = ck
                    best_out_dir = out_dir_ck
            else:
                print(f"Could not find AUC in {report_path}")
            continue

        print(f"\n=== Evaluating checkpoint: {ck} ===")
        # Prepare output dir for this checkpoint
        os.makedirs(out_dir_ck, exist_ok=True)

        # Prepare args for inference
        args_ck = argparse.Namespace(**vars(args))
        args_ck.ckpt = ck
        args_ck.out_dir = out_dir_ck

        try:
            inference(args_ck)
        except Exception as e:
            print(f"Error running inference for checkpoint {ck}: {e}")
            continue

        # Read AUC from report.txt
        report_path = os.path.join(out_dir_ck, "report.txt")
        auc = None
        if os.path.exists(report_path):
            with open(report_path) as f:
                for line in f:
                    if line.strip().startswith("AUC"):
                        try:
                            auc = float(line.strip().split(":")[1])
                        except Exception:
                            pass
                        break
        if auc is not None:
            results.append({"ckpt": ck, "auc": auc, "out_dir": out_dir_ck})
            print(f"Checkpoint {ck}: AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_ckpt = ck
                best_out_dir = out_dir_ck
        else:
            print(f"Could not find AUC in {report_path}")

    # Save summary CSV
    df = pd.DataFrame(results)
    summary_csv = os.path.join(args.out_dir, "auc_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"\nAUC summary saved to {summary_csv}")

    if best_ckpt is not None:
        print(f"\nBest checkpoint: {best_ckpt} (AUC={best_auc:.4f})")
        print(f"Results in: {best_out_dir}")
        # Optionally, copy best results to a top-level folder
        best_dir = os.path.join(args.out_dir, "best_ckpt")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(best_out_dir, best_dir)
        print(f"Best checkpoint results copied to {best_dir}")
        # Optionally, remove other result folders
        if not args.keep_all:
            for r in results:
                if r["out_dir"] != best_out_dir:
                    shutil.rmtree(r["out_dir"], ignore_errors=True)
    else:
        print("No valid AUC found for any checkpoint.")

if __name__ == "__main__":
    main()
