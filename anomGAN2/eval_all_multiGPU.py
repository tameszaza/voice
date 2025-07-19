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
import torch.multiprocessing as mp
from inference_r import RealFakeDirDataset
import torch

import glob
import os
import numpy as np

def aggregate_results(out_dir, mode):
    best_auc = -1.0
    best_ckpt = None
    lowest_real = float('inf')
    lowest_ckpt = None

    base = os.path.join(out_dir, mode)
    for d in glob.glob(os.path.join(base, f"{mode}_*")):
        ckpt = os.path.basename(d).split("_",1)[1]

        # --- parse AUC from report.txt ---
        rpt = os.path.join(d, "report.txt")
        auc = None
        if os.path.exists(rpt):
            with open(rpt) as f:
                for line in f:
                    if line.startswith("AUC"):
                        try:
                            auc = float(line.split(":")[1])
                        except:
                            auc = None
                        break

        # --- compute avg real score ---
        try:
            scores = np.load(os.path.join(d, "scores.npy"))
            labels = np.load(os.path.join(d, "labels.npy"))
            real_scores = scores[labels == 0]
            avg_real = real_scores.mean() if len(real_scores) else float('nan')
        except:
            avg_real = float('nan')

        # --- pick best / lowest ---
        if auc is not None and auc > best_auc:
            best_auc, best_ckpt = auc, ckpt
        if not np.isnan(avg_real) and avg_real < lowest_real:
            lowest_real, lowest_ckpt = avg_real, ckpt

    print(f"\n>> [{mode}] Best ckpt: {best_ckpt} with AUC={best_auc:.4f}")
    print(f">> [{mode}] Lowest real‐anomaly score: {lowest_ckpt} with avg_real={lowest_real:.6f}")


def gpu_worker(rank, world_size, args, ckpts, mode):
    # pin this process to GPU #rank
    torch.cuda.set_device(rank)
    args.device = f"cuda:{rank}"
    torch.set_num_threads(max(1, args.num_threads // args.num_workers))

    # rebuild dataset & loader here 
    ds = RealFakeDirDataset(
        args.real_data_root, args.anomaly_data_dir,
        args.use_mel, args.use_mfcc,
        args.max_samples_per_class, cache_in_ram=True
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,  
    )


    my_ckpts = ckpts[rank::world_size]
    print(f"[{mode}][GPU {rank}] {len(my_ckpts)} checkpoints to eval")
    for ck in my_ckpts:
        out_dir = os.path.join(args.out_dir, mode, f"{mode}_{ck}")
        os.makedirs(out_dir, exist_ok=True)

        # **skip if all result files already exist**
        if result_files_exist(out_dir):
            print(f"[{mode}][GPU {rank}] skipping ckpt {ck}, results already there")
            continue

        a = argparse.Namespace(**vars(args))
        a.ckpt     = ck
        a.out_dir  = out_dir
        a.cache_in_ram = True

        try:
            inference(a, ds=ds, loader=loader)
        except Exception as e:
            print(f"[{mode}][GPU {rank}] ckpt={ck} failed: {e}")


    print(f"[{mode}][GPU {rank}] done.")

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

def get_avg_real_score(out_dir):
    """Load scores and labels from out_dir and return mean anomaly score for real class (label==0)"""
    try:
        scores = np.load(os.path.join(out_dir, "scores.npy"))
        labels = np.load(os.path.join(out_dir, "labels.npy"))
        real_scores = scores[labels == 0]
        if len(real_scores) == 0:
            return float('nan')
        return float(np.mean(real_scores))
    except Exception as e:
        print(f"Could not compute avg real score in {out_dir}: {e}")
        return float('nan')

def result_files_exist(out_dir):
    """Check if the main result files exist in out_dir"""
    required = ["scores.npy", "labels.npy", "report.txt"]
    return all(os.path.exists(os.path.join(out_dir, f)) for f in required)

def main():
    parser = argparse.ArgumentParser(
        description="Automate inference over all checkpoints and report best AUC"
    )
    # All arguments from inference_r.py except --ckpt and --out_dir
    parser.add_argument("--real_data_root",  required=True)
    parser.add_argument("--anomaly_data_dir",required=True)
    parser.add_argument("--model_dir",       required=True)
    parser.add_argument("--batch_size",      type=int, default=256)
    parser.add_argument("--z_dim",           type=int, default=128)
    parser.add_argument("--n_clusters",      type=int, default=7)
    parser.add_argument("--base_channels",   type=int, default=32)
    parser.add_argument("--n_layers",        type=int, default=3)
    parser.add_argument("--alpha",           type=float, default=0.9)
    parser.add_argument("--n_thresholds",    type=int, default=200)
    parser.add_argument("--use_mel",         action="store_true")
    parser.add_argument("--use_mfcc",        action="store_true")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--max_samples_per_class", type=int, default=100000)
    parser.add_argument("--bypass_classifier", action="store_true")
    parser.add_argument("--anom_noise_std", type=float, default=0.0)
    parser.add_argument("--anom_score_add", type=float, default=0.0,
                        help="If >0, add this scalar to anomaly scores for anomaly class; if <0, add |value| to real class")
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
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader worker processes per GPU")
    parser.add_argument("--num_threads", type=int, default=os.cpu_count(), help="Total CPU threads available for data loading")

    args = parser.parse_args()

    set_seed(args.seed)

    # first: sweep all G_*.pt across GPUs
    G_ckpts = find_checkpoints(args.model_dir, prefix="G_", suffix=".pt")
    if G_ckpts:
        world_size = torch.cuda.device_count()
        print(f"Sweeping {len(G_ckpts)} generators over {world_size} GPUs…")
        mp.spawn(
            gpu_worker,
            args=(world_size, args, G_ckpts, "G"),
            nprocs=world_size,
            join=True
        )
        aggregate_results(args.out_dir, "G")
    # then (if requested) sweep all E_xzx_*.pt
    if args.compare_mode:
        E_ckpts = find_encoder_checkpoints(args.model_dir, prefix=args.encoder_pre, suffix=".pt")
        if E_ckpts:
            world_size = torch.cuda.device_count()
            print(f"Sweeping {len(E_ckpts)} encoders over {world_size} GPUs…")
            mp.spawn(
                gpu_worker,
                args=(world_size, args, E_ckpts, "E"),
                nprocs=world_size,
                join=True
            )
            aggregate_results(args.out_dir, "E")
        else:
            print("No encoder checkpoints to compare.")


if __name__ == "__main__":
    main()
