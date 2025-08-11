#!/usr/bin/env python
# eval_res_ganomaly.py
# ------------------------------------------------------------
# Evaluation script for multi-decoder RES-GANomaly (D-AnoGAN style)
# ------------------------------------------------------------
from __future__ import annotations
import os, json, argparse, random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn import metrics as skm

# ─── project imports ────────────────────────────────────────
from Resnetworks import (
    NetG_MultiDecoder_RES_GANomaly,
    NetD_RES_GANomaly,
    BranchClassifier,
)
# ------------------------------------------------------------


class RealFakeDataset(Dataset):
    """
    • If `root` is a file → behaves like old loader, y = 0 for all samples.
    • If `root` is a directory → expects sub-dirs 'real' & 'fake' (names
      can be swapped via --bona_class).  Loads every *.npy inside.
    All arrays must be (N, C, H, W) with square power-of-two H=W.
    """
    def __init__(self, root: str, bona_name: str = "real"):
        root_path = Path(root)
        if root_path.is_file():
            arr = np.load(root_path).astype(np.float32)
            self.x = torch.from_numpy(arr)
            self.y = torch.zeros(len(arr), dtype=torch.long)
            self.k = torch.zeros(len(arr), dtype=torch.long)
            return

        real_dir = root_path / "real"
        fake_dir = root_path / "fake"
        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError("Expecting sub-dirs 'real' and 'fake' in "
                                    f"{root_path}")

        xs, ys, ks = [], [], []
        cluster_id = 0
        for label_dir, label_name in [(real_dir, "real"), (fake_dir, "fake")]:
            for f in sorted(label_dir.glob("*.npy")):
                arr = np.load(f).astype(np.float32)
                if arr.ndim != 4 or arr.shape[2] != arr.shape[3] or (arr.shape[2] & (arr.shape[2] - 1)):
                    raise ValueError(f"{f}: need (N,C,H,H) power-of-2, got {arr.shape}")
                N, C, H, W = arr.shape
                xs.append(torch.from_numpy(arr))
                ys.append(torch.full((N,), 0 if label_name == bona_name else 1,
                                     dtype=torch.long))
                ks.append(torch.full((N,), cluster_id, dtype=torch.long))
                print(f"[Dataset] {label_name:5s} {f.name:>15}: {N} samples → cluster {cluster_id}")
                cluster_id += 1

        self.x = torch.cat(xs)
        self.y = torch.cat(ys)
        self.k = torch.cat(ks)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.k[idx]


def parse_args():
    p = argparse.ArgumentParser("Evaluate RES-GANomaly checkpoint")
    p.add_argument("--data_path",  required=True,
                   help="dir with cluster*.npy OR single test.npy")
    p.add_argument("--ckpt",       required=True,
                   help="checkpoint_*.pth produced by training")
    p.add_argument("--out_dir",    default="eval_outputs")
    p.add_argument("--batch",      type=int, default=64)
    p.add_argument("--metric",     choices=["roc", "prc", "eer"], default="roc")
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--n_vis",      type=int, default=4,
                   help="How many (good,bad) pairs to visualise")
    p.add_argument("--bona_class", choices=["real", "fake"], default="real",
                   help="'real' or 'fake' is considered bona-fide (label 0)")

    # network hyper-parameters (fallback from ckpt.opt if not in args)
    p.add_argument("--isize",      type=int)
    p.add_argument("--nc",         type=int)
    p.add_argument("--nz",         type=int)
    p.add_argument("--ngf",        type=int)
    p.add_argument("--ndf",        type=int)
    p.add_argument("--n_branches", type=int, default=None,
                   help="(optional) force # of decoder branches")
    p.add_argument("--extra_res",  type=int, default=0,
                   help="Extra ResidualSEBlocks per down/upsampling stage")

    # **new**: how far out (percentile) to plot absolute scores
    p.add_argument("--score_pct",  type=float, default=80.0,
                   help="upper percentile cutoff for absolute-score histogram")

    return p.parse_args()


def build_model(args, ckpt: dict, device: torch.device):
    """
    Build Generator, Discriminator and Branch-Classifier whose sizes match
    the *checkpoint* (falls back to CLI flags only if necessary).
    """
    src = ckpt.get("opt", vars(args))
    required = ["isize", "nc", "nz", "ngf", "ndf", "extra_res"]
    missing = [k for k in required if src.get(k) is None]
    if missing:
        raise ValueError("Missing architecture args: " + ", ".join(missing))

    # ─── Override or infer number of decoder branches ────────────
    if args.n_branches is not None:
        n_br = args.n_branches
        print(f"[Eval] forcing n_branches → {n_br} (from CLI flag)")
    else:
        n_br = _infer_branches_from_ckpt(ckpt["netG"], src.get("n_branches", 1))
        print(f"[Eval] inferring n_branches → {n_br} (from checkpoint)")

    # pack into a simple namespace
    o = SimpleNamespace(
        isize=int(src["isize"]),
        nc=int(src["nc"]),
        nz=int(src["nz"]),
        ngf=int(src["ngf"]),
        ndf=int(src["ndf"]),
        ngpu=0,
        n_branches=n_br,
        extra_res=int(src["extra_res"]),
    )

    netG = NetG_MultiDecoder_RES_GANomaly(o, n_br).to(device)
    netD = NetD_RES_GANomaly(o).to(device)

    # build branch classifier
    if "branchClf" in ckpt:
        first_w = next(iter(ckpt["branchClf"].values()))
        in_ch   = first_w.shape[1]
    else:
        with torch.no_grad():
            _, feat = netD(torch.randn(1, o.nc, o.isize, o.isize, device=device))
            in_ch = feat.size(1)
    branchClf = BranchClassifier(in_ch, n_br).to(device)

    # load weights
    netG.load_state_dict(ckpt["netG"], strict=False)
    netD.load_state_dict(ckpt["netD"], strict=False)
    if "branchClf" in ckpt:
        branchClf.load_state_dict(ckpt["branchClf"], strict=False)

    netG.eval(); netD.eval(); branchClf.eval()
    return netG, netD, branchClf


@torch.no_grad()
def infer_scores(netG, netD, clf, loader, device):
    scores, k_all, y_all = [], [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        z    = netG.encoder(x)
        _, feats = netD(x)
        k_pred   = clf(feats).argmax(dim=1)

        x_hat = torch.empty_like(x)
        z_hat = torch.empty_like(z)
        for k in range(netG.n_branches):
            mask = (k_pred == k)
            if mask.any():
                recon_k    = netG.decoders[k](z[mask])
                x_hat[mask] = recon_k
                z_hat[mask] = netG.encoder(recon_k)

        s = torch.norm((z - z_hat).flatten(1), p=2, dim=1)
        scores.append(s.cpu());   k_all.append(k_pred.cpu());   y_all.append(y.cpu())

    return (torch.cat(scores).numpy(),
            torch.cat(k_all).numpy(),
            torch.cat(y_all).numpy())


def _infer_branches_from_ckpt(sd: dict, fallback: int) -> int:
    dec_ids = [int(k.split('.')[1]) for k in sd.keys()
               if k.startswith("decoders.") and k.split('.')[1].isdigit()]
    return max(dec_ids) + 1 if dec_ids else fallback


def best_f1_acc(y, scores):
    prec, rec, thr = skm.precision_recall_curve(y, scores)
    f1 = 2*prec*rec / (prec + rec + 1e-9)
    idx_f1 = np.nanargmax(f1)
    best_thr_f1, best_f1 = thr[idx_f1], f1[idx_f1]

    fpr, tpr, roc_thr = skm.roc_curve(y, scores)
    acc = [(scores >= t).mean() for t in roc_thr]
    idx_acc = int(np.argmax(acc))
    best_thr_acc, best_acc = roc_thr[idx_acc], acc[idx_acc]
    return best_thr_f1, best_f1, best_thr_acc, best_acc


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    out    = Path(args.out_dir)

    # ─── data loader ───────────────────────────────────────────
    ds     = RealFakeDataset(args.data_path, bona_name=args.bona_class)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=15, pin_memory=True)
    print(f"[Eval] dataset size: {len(ds)}  (bona={args.bona_class})")

    # ─── model build ───────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    dec_keys = [k for k in ckpt["netG"].keys() 
                if k.startswith("decoders.") and k.split(".")[1].isdigit()]
    dec_ids  = sorted({int(k.split(".")[1]) for k in dec_keys})
    print("Found decoder IDs:", dec_ids)
    print("→ number of decoders =", len(dec_ids))

    # 2) Inspect branchClf shape
    clf_state = ckpt.get("branchClf", {})
    for name, w in clf_state.items():
        print(f"branchClf.{name}.shape =", tuple(w.shape))

    # 3) (If full checkpoint) view saved opt
    if "opt" in ckpt:
        print("Saved opt.n_branches =", ckpt["opt"].get("n_branches"))
    else:
        print("No 'opt' in this checkpoint (light save).")
    
    netG, netD, clf = build_model(args, ckpt, device)

    # ─── inference ─────────────────────────────────────────────
    scores, k_used, y_true = infer_scores(netG, netD, clf, loader, device)

    # ─── scalar metrics ---------------------------------------
    auc = skm.roc_auc_score(y_true, scores)
    thr_f1, best_f1, thr_acc, best_acc = best_f1_acc(y_true, scores)
    print(f"[Eval]  AUC={auc:.4f} | best-F1={best_f1:.3f}@{thr_f1:.4f} "
          f"| best-ACC={best_acc:.3f}@{thr_acc:.4f}")

    # ─── 1) absolute-score histogram (0 → score_pct percentile) -------
    cutoff_val = np.percentile(scores, args.score_pct)
    print(f"[Eval] plotting scores from 0 to {args.score_pct}th pct → {cutoff_val:.4f}")
    plt.figure()
    plt.hist(scores[y_true == 0], bins=80, range=(0, cutoff_val),
             density=True, alpha=.6, label="bona-fide")
    plt.hist(scores[y_true == 1], bins=80, range=(0, cutoff_val),
             density=True, alpha=.6, label="anomaly")
    plt.axvline(thr_f1,  color="k", ls="--", label=f"thr F1={thr_f1:.3f}")
    plt.axvline(thr_acc, color="r", ls="--", label=f"thr ACC={thr_acc:.3f}")
    plt.xlim(0, cutoff_val)
    plt.xlabel("anomaly score");  plt.ylabel("density");  plt.legend()
    plt.title(f"Score distribution (absolute, 0–{args.score_pct}th pct)")
    plt.tight_layout()
    plt.savefig(out / "score_hist.png", dpi=150)
    plt.close()

    # ─── 2) decoder-usage bar charts ---------------------------
    def _bar(data, title, fname):
        uniq, cnt = np.unique(data, return_counts=True)
        plt.figure(); plt.bar(uniq, cnt)
        plt.xlabel("decoder id"); plt.ylabel("count"); plt.title(title)
        plt.tight_layout(); plt.savefig(out / fname, dpi=150); plt.close()

    _bar(k_used,               "Decoder usage (all)",       "decoder_usage_all.png")
    _bar(k_used[y_true == 0],  "Decoder usage (bona-fide)", "decoder_usage_bona.png")
    _bar(k_used[y_true == 1],  "Decoder usage (anomaly)",   "decoder_usage_anom.png")

    # ─── 3) reconstruction pairs  -----------------------------
    rng      = np.random.default_rng()
    idx_bona = rng.choice(np.where(y_true == 0)[0], args.n_vis, replace=False)
    idx_anom = rng.choice(np.where(y_true == 1)[0], args.n_vis, replace=False)

    for cnt, idx in enumerate(np.concatenate([idx_bona, idx_anom])):
        lbl   = "bona" if y_true[idx] == 0 else "anom"
        x_vis = ds.x[idx:idx+1].to(device)              # (1, C, H, W)
        k_vis = int(k_used[idx])

        with torch.no_grad():
            z_vis = netG.encoder(x_vis)
            x_hat = netG.decoders[k_vis](z_vis)[0].cpu()   # (C, H, W)

        x_orig = ds.x[idx].cpu()                           # (C, H, W)

        plt.figure(figsize=(4,2))
        plt.subplot(1,2,1)
        plt.imshow(x_orig.squeeze(0), cmap="magma")        # (H, W)
        plt.axis("off"); plt.title("orig")

        plt.subplot(1,2,2)
        plt.imshow(x_hat.squeeze(0),  cmap="magma")        # (H, W)
        plt.axis("off"); plt.title("recon")

        plt.suptitle(f"{lbl.upper()} • decoder {k_vis}")
        plt.tight_layout()
        plt.savefig(out / f"recon_{lbl}_{cnt}.png", dpi=150)
        plt.close()


    # ─── 4) save raw metrics -----------------------------------
    metrics = dict(
        auc=float(auc),
        best_f1=float(best_f1), thr_f1=float(thr_f1),
        best_acc=float(best_acc), thr_acc=float(thr_acc)
    )
    with open(out / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"[Eval] metrics + figures saved to {out.resolve()}")


if __name__ == "__main__":
    main()


#python testVastAI/voice/Ganomaly_based/res_mganomaly/lib/evaluate.py --bona_class fake --data_path ResData/wavefake128_2048split/test/aggregated --ckpt output_ResMGAN/Fi128z100extra4/checkpoints/weights_epoch140.pth --out_dir evalOut/Fi128z100extra4 --isize 128 --nc 1 --nz 100 --ngf 64 --ndf 128 --n_branches 7 --extra_res 4