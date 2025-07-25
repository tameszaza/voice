#!/usr/bin/env python
# eval_res_ganomaly.py
# ------------------------------------------------------------
# Evaluation script for multi-decoder RES-GANomaly (D-AnoGAN style)
# ------------------------------------------------------------
from __future__ import annotations
import os, json, argparse, math, random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn import metrics as skm

# ─── project imports ────────────────────────────────────────          # builds Option-compatible nets
from Resnetworks        import (
    NetG_MultiDecoder_RES_GANomaly,
    NetD_RES_GANomaly, 
    BranchClassifier,
)
# ------------------------------------------------------------


# ─────────────────────────────────────────────────────────────
# 1) Dataset ─ handles   test_dir/real  &  test_dir/fake
#    plus the older “single .npy file” fallback
# ─────────────────────────────────────────────────────────────
class RealFakeDataset(Dataset):
    """
    • If `root` is a file → behaves like old loader, y = 0 for all samples.
    • If `root` is a directory → expects sub-dirs 'real' & 'fake' (names
      can be swapped via --bona_class flag).  Loads every *.npy inside.
    All arrays must be (N, C, H, W) with square power-of-two H=W.
    """
    def __init__(self, root: str, bona_name: str = "real"):
        root_path = Path(root)
        if root_path.is_file():
            arr = np.load(root_path).astype(np.float32)
            self.x   = torch.from_numpy(arr)
            self.y   = torch.zeros(len(arr), dtype=torch.long)   # all bona-fide
            self.k   = torch.full_like(self.y, 0)                # dummy cluster id
            return

        # ---- directory mode ------------------------------------
        real_dir = root_path / "real"
        fake_dir = root_path / "fake"
        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError("Expecting sub-dirs 'real' and 'fake' in "
                                    f"{root_path}")

        xs, ys, ks = [], [], []
        cluster_id = 0
        for label_dir, label_name in [(real_dir, "real"), (fake_dir, "fake")]:
            for f in sorted(label_dir.glob("*.npy")):
                arr = np.load(f).astype(np.float32)   # (N,C,H,W)
                N, C, H, W = arr.shape
                if arr.ndim != 4 or H != W or H & (H - 1):
                    raise ValueError(f"{f}: need (N,C,H,H) power-of-2, got {arr.shape}")
                xs.append(torch.from_numpy(arr))
                ys.append(torch.full((N,), 0 if label_name == bona_name else 1,
                                     dtype=torch.long))
                ks.append(torch.full((N,), cluster_id, dtype=torch.long))
                print(f"[Dataset] {label_name:5s} {f.name:>15}: {N} samples → cluster {cluster_id}")
                cluster_id += 1

        self.x = torch.cat(xs)
        self.y = torch.cat(ys)
        self.k = torch.cat(ks)

    def __len__(self):  return self.x.size(0)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.k[idx]   # x, GT label, (unused) k


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
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

    # ★ Fallback network hyper-parameters (used only if ckpt lacks them)
    p.add_argument("--isize",      type=int)
    p.add_argument("--nc",         type=int)
    p.add_argument("--nz",         type=int)
    p.add_argument("--ngf",        type=int)
    p.add_argument("--ndf",        type=int)
    p.add_argument("--n_branches", type=int,
                   help="number of decoders in the generator")

    return p.parse_args()


def build_model(args, ckpt: dict, device: torch.device):
    """
    Build Generator, Discriminator and Branch-Classifier whose sizes match
    the *checkpoint* (falls back to CLI flags only if necessary).
    """
    # ── retrieve essential hyper-parameters ──────────────────────────
    src = ckpt.get("opt", vars(args))
    need = ["isize", "nc", "nz", "ngf", "ndf"]
    miss = [k for k in need if src.get(k) is None]
    if miss:
        raise ValueError("Missing architecture args: " + ", ".join(miss))

    n_br = _infer_branches_from_ckpt(ckpt["netG"], src.get("n_branches", 1))

    class _O: pass
    o              = _O()
    o.isize        = int(src["isize"]);  o.nc   = int(src["nc"])
    o.nz           = int(src["nz"]);     o.ngf  = int(src["ngf"])
    o.ndf          = int(src["ndf"]);    o.ngpu = 0
    o.n_branches   = n_br

    # ── instantiate nets ────────────────────────────────────────────
    netG = NetG_MultiDecoder_RES_GANomaly(o, n_br).to(device)
    netD = NetD_RES_GANomaly(o).to(device)

    # try to deduce classifier-input channels from ckpt if present
    if "branchClf" in ckpt:
        first_w = next(iter(ckpt["branchClf"].values()))
        in_ch   = first_w.shape[1]
    else:
        with torch.no_grad():
            dummy   = torch.randn(1, o.nc, o.isize, o.isize, device=device)
            _, feat = netD(dummy)
            in_ch   = feat.size(1)

    branchClf = BranchClassifier(in_ch, n_br).to(device)

    # ── load weights (ignore missing / unexpected keys) ─────────────
    netG.load_state_dict(ckpt["netG"],  strict=False)
    netD.load_state_dict(ckpt["netD"],  strict=False)
    if "branchClf" in ckpt:
        branchClf.load_state_dict(ckpt["branchClf"], strict=False)

    netG.eval(); netD.eval(); branchClf.eval()
    return netG, netD, branchClf




# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_scores(netG, netD, clf, loader, device):
    """
    Forward-pass the whole test-set and return:
        scores  – L2 distance in latent space   shape (N,)
        k_used  – decoder index chosen          shape (N,)
        y_true  – ground-truth labels (0/1)     shape (N,)
    """
    scores, k_all, y_all = [], [], []

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)

        # 1) encode once
        z = netG.encoder(x)

        # 2) discriminator features → branch classifier
        _, feats = netD(x)
        k_pred   = clf(feats).argmax(dim=1)          # (B,)

        # 3) per-branch reconstruction
        x_hat = torch.empty_like(x);  z_hat = torch.empty_like(z)
        for k in range(netG.n_branches):
            mask = (k_pred == k)
            if mask.any():
                recon_k   = netG.decoders[k](z[mask])
                x_hat[mask] = recon_k
                z_hat[mask] = netG.encoder(recon_k)

        # 4) anomaly score
        s = torch.norm((z - z_hat).flatten(1), p=2, dim=1)

        scores.append(s.cpu());   k_all.append(k_pred.cpu());   y_all.append(y.cpu())

    return (torch.cat(scores).numpy(),
            torch.cat(k_all).numpy(),
            torch.cat(y_all).numpy())


# ─────────────────────────────────────────────────────────────
def _infer_branches_from_ckpt(sd: dict, fallback: int) -> int:
    dec_ids = [int(k.split('.')[1]) for k in sd.keys()
               if k.startswith("decoders.") and k.split('.')[1].isdigit()]
    return max(dec_ids)+1 if dec_ids else fallback



# ─────────────────────────────────────────────────────────────
def best_f1_acc(y, scores):
    prec, rec, thr = skm.precision_recall_curve(y, scores)
    f1 = 2*prec*rec / (prec+rec + 1e-9)
    idx_f1 = np.nanargmax(f1);  best_thr_f1, best_f1 = thr[idx_f1], f1[idx_f1]

    fpr, tpr, roc_thr = skm.roc_curve(y, scores)
    acc = [(scores >= t).mean() for t in roc_thr]
    idx_acc = int(np.argmax(acc));  best_thr_acc, best_acc = roc_thr[idx_acc], acc[idx_acc]
    return best_thr_f1, best_f1, best_thr_acc, best_acc


# ─────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- data --------------------------------------------------
    ds = RealFakeDataset(args.data_path, bona_name=args.bona_class)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"[Eval] dataset size: {len(ds)}  (bona={args.bona_class})")

    # ---- model -------------------------------------------------
    ckpt   = torch.load(args.ckpt, map_location="cpu")
    netG, netD, clf = build_model(args, ckpt, device)

    # ---- inference --------------------------------------------
    scores, k_used, y_true = infer_scores(netG, netD, clf, loader, device)

    auc = skm.roc_auc_score(y_true, scores)
    thr_f1, best_f1, thr_acc, best_acc = best_f1_acc(y_true, scores)
    print(f"[Eval] AUC={auc:.4f} | best-F1={best_f1:.3f}@{thr_f1:.4f} "
          f"| best-ACC={best_acc:.3f}@{thr_acc:.4f}")

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------
    out = Path(args.out_dir)

    # 1) score histogram
    plt.figure()
    plt.hist(scores[y_true==0], 100, alpha=.6, label="bona-fide")
    plt.hist(scores[y_true==1], 100, alpha=.6, label="anomaly")
    plt.axvline(thr_f1,  color="k", ls="--", label="thr_F1")
    plt.axvline(thr_acc, color="r", ls="--", label="thr_ACC")
    plt.xlabel("anomaly score"); plt.ylabel("count"); plt.legend()
    plt.title("Score distribution")
    plt.savefig(out / "score_hist.png", dpi=150)

    # 2) decoder usage (overall + per class)
    def _bar(data, title, fname):
        uniq, cnt = np.unique(data, return_counts=True)
        plt.figure(); plt.bar(uniq, cnt); plt.xlabel("decoder id"); plt.ylabel("count")
        plt.title(title); plt.savefig(out / fname, dpi=150)
    _bar(k_used,                "Decoder usage (all)",          "decoder_usage_all.png")
    _bar(k_used[y_true==0],     "Decoder usage (bona-fide)",    "decoder_usage_bona.png")
    _bar(k_used[y_true==1],     "Decoder usage (anomaly)",      "decoder_usage_anom.png")

    # 3) recon examples (N couples per class)
    n = args.n_vis
    idx_bona = np.random.choice(np.where(y_true==0)[0], n, replace=False)
    idx_anom = np.random.choice(np.where(y_true==1)[0], n, replace=False)
    vis_idx  = np.concatenate([idx_bona, idx_anom])
    x_vis    = ds.x[vis_idx].to(device)
    k_vis    = k_used[vis_idx]
    with torch.no_grad():
        z_vis = netG.encoder(x_vis)
        x_hat = torch.empty_like(x_vis)
        for i,k in enumerate(k_vis):
            k_int = int(k) % len(netG.decoders)  
            x_hat[i] = netG.decoders[int(k)](z_vis[i:i+1])[0]
    plt.figure(figsize=(4, 2*n))
    for i in range(2*n):
        plt.subplot(2*n,2,2*i+1); plt.imshow(x_vis[i,0].cpu(), cmap="magma"); plt.axis("off")
        plt.subplot(2*n,2,2*i+2); plt.imshow(x_hat[i,0].cpu(), cmap="magma"); plt.axis("off")
    plt.suptitle(f"{n} bona-fide (top) + {n} anomaly (bottom) : original vs recon")
    plt.savefig(out / "recon_pairs.png", dpi=150)

    # 4) raw metrics JSON
    metrics = dict(auc=auc, best_f1=float(best_f1), thr_f1=float(thr_f1),
                   best_acc=float(best_acc), thr_acc=float(thr_acc))
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Eval] plots + metrics saved to  {out.absolute()}")


if __name__ == "__main__":
    main()

#python testVastAI/voice/Ganomaly_based/res_mganomaly/lib/evaluate.py --bona_class fake --data_path ResData/wavefake32_split/test/aggregated --ckpt output_ResMGAN/Fi64z100/checkpoints/weights_epoch1440.pth --out_dir evalOut/Fi64z100 --isize 32 --nc 1 --nz 100 --ngf 64 --ndf 128 --n_branches 7