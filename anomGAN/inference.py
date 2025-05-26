#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, f1_score
)
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  Dataset: paired mel / mfcc .npy files for real vs fake classes
# -----------------------------------------------------------------------------
class RealFakeAudioDataset(Dataset):
    def __init__(self,
                 real_mel_dir=None, real_mfcc_dir=None,
                 fake_mel_dir=None, fake_mfcc_dir=None,
                 n_samples=None, seed=0):
        # must supply at least one feature per class
        if real_mel_dir is None and real_mfcc_dir is None:
            raise ValueError("Need at least one of --real_mel_dir or --real_mfcc_dir")
        if fake_mel_dir is None and fake_mfcc_dir is None:
            raise ValueError("Need at least one of --fake_mel_dir or --fake_mfcc_dir")
        self.real_mel_dir, self.real_mfcc_dir = real_mel_dir, real_mfcc_dir
        self.fake_mel_dir, self.fake_mfcc_dir = fake_mel_dir, fake_mfcc_dir

        # list filenames for each class (use intersection if both dirs provided)
        real_base = real_mel_dir or real_mfcc_dir
        fake_base = fake_mel_dir or fake_mfcc_dir
        real_files = sorted(f for f in os.listdir(real_base) if f.endswith('.npy'))
        fake_files = sorted(f for f in os.listdir(fake_base) if f.endswith('.npy'))

        # optionally subsample
        if n_samples:
            random.seed(seed)
            real_files = random.sample(real_files, min(n_samples, len(real_files)))
            fake_files = random.sample(fake_files, min(n_samples, len(fake_files)))

        # build sample list: (mel_path or None, mfcc_path or None, label)
        self.samples = []
        for fn in real_files:
            m = os.path.join(real_mel_dir, fn) if real_mel_dir else None
            f = os.path.join(real_mfcc_dir, fn) if real_mfcc_dir else None
            self.samples.append((m, f, 0))
        for fn in fake_files:
            m = os.path.join(fake_mel_dir, fn) if fake_mel_dir else None
            f = os.path.join(fake_mfcc_dir, fn) if fake_mfcc_dir else None
            self.samples.append((m, f, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel_path, mfcc_path, label = self.samples[idx]
        feats = []
        if mel_path:
            feats.append(np.load(mel_path))
        if mfcc_path:
            feats.append(np.load(mfcc_path))
        x = np.stack(feats, axis=0)        # (C, H, W)
        return torch.from_numpy(x).float(), label

# -----------------------------------------------------------------------------
#  Inference pipeline following D-AnoGAN §III-D and Eq. (6)
# -----------------------------------------------------------------------------
def inference(args):
    device = torch.device(args.device)

    # build dataset and loader
    ds = RealFakeAudioDataset(
        real_mel_dir=args.real_mel_dir,
        real_mfcc_dir=args.real_mfcc_dir,
        fake_mel_dir=args.fake_mel_dir,
        fake_mfcc_dir=args.fake_mfcc_dir,
        n_samples=args.n_samples,
        seed=args.seed
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # infer channels and spatial size
    x0, _ = ds[0]
    C_in, H, W = x0.shape

    # instantiate models exactly as trained
    from models import MultiGenerator, MultiEncoder, Discriminator, Classifier
    G = MultiGenerator(
        args.z_dim, out_channels=C_in,
        base_channels=args.base_channels,
        img_size=H, n_layers=args.n_layers,
        n_clusters=args.n_clusters
    ).to(device)
    E = MultiEncoder(
        in_channels=C_in, z_dim=args.z_dim,
        base_channels=args.base_channels,
        img_size=H, n_layers=args.n_layers,
        n_clusters=args.n_clusters
    ).to(device)
    D = Discriminator(
        in_channels=C_in, base_channels=args.base_channels,
        n_layers=args.n_layers
    ).to(device)

    # classifier needs D.intermediate channel count
    with torch.no_grad():
        dummy = torch.zeros(1, C_in, H, W, device=device)
        feat = D.intermediate(dummy)
    C = Classifier(in_channels=feat.shape[1], n_clusters=args.n_clusters).to(device)

    # Print model summaries
    try:
        from torchinfo import summary
        print("==== MultiGenerator ====")
        summary(G, input_data=(torch.zeros(1, args.z_dim, device=device), torch.zeros(1, dtype=torch.long, device=device)), 
                kwargs={"target_hw": (H, W)})
        print("==== MultiEncoder ====")
        summary(E, input_data=(torch.zeros(1, C_in, H, W, device=device), torch.zeros(1, dtype=torch.long, device=device)))
        print("==== Discriminator ====")
        summary(D, input_data=torch.zeros(1, C_in, H, W, device=device))
        print("==== Classifier ====")
        summary(C, input_data=torch.zeros(1, feat.shape[1], feat.shape[2], feat.shape[3], device=device))
    except ImportError:
        print("torchinfo not installed, printing model structures instead.")
        print("==== MultiGenerator ====")
        print(G)
        print("==== MultiEncoder ====")
        print(E)
        print("==== Discriminator ====")
        print(D)
        print("==== Classifier ====")
        print(C)

    # Print number of parameters for each model
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MultiGenerator parameters: {count_params(G):,}")
    print(f"MultiEncoder parameters:   {count_params(E):,}")
    print(f"Discriminator parameters:  {count_params(D):,}")
    print(f"Classifier parameters:     {count_params(C):,}")

    # load checkpoints
    ck = args.ckpt
    md = args.model_dir
    G.load_state_dict(torch.load(os.path.join(md, f"G_{ck}.pt"), map_location=device))
    if args.xzx_encoder:
        print(f"Loading encoder from Phase 2: {args.xzx_encoder}")
        E.load_state_dict(torch.load(args.xzx_encoder, map_location=device))
    else:
        print(f"Loading encoder from Phase 1: E_{ck}.pt")
        E.load_state_dict(torch.load(os.path.join(md, f"E_{ck}.pt"), map_location=device))

    D.load_state_dict(torch.load(os.path.join(md, f"D_{ck}.pt"), map_location=device))
    C.load_state_dict(torch.load(os.path.join(md, f"C_{ck}.pt"), map_location=device))

    G.eval(); E.eval(); D.eval(); C.eval()
    # ------------------------------------------------------------
    # DEBUG: visualize first 5 real samples vs. their reconstructions
    import numpy as np
    import matplotlib.pyplot as plt

    # --- DEBUG: visualize first 5 reconstructions and print cluster k ---
    debug_dir = os.path.join(args.out_dir, "debug_first5")
    os.makedirs(debug_dir, exist_ok=True)

    # Visualize first 5 real and first 5 fake samples
    debug_samples = []
    real_count, fake_count = 0, 0
    for i in range(len(ds)):
        x, lbl = ds[i]
        if lbl == 0 and real_count < 5:
            debug_samples.append((i, x, lbl))
            real_count += 1
        elif lbl == 1 and fake_count < 5:
            debug_samples.append((i, x, lbl))
            fake_count += 1
        if real_count >= 5 and fake_count >= 5:
            break

    for i, x, lbl in debug_samples:
        x = x.unsqueeze(0).to(device)   # shape (1, C, H, W)

        # 1) cluster assignment
        feat_x = D.intermediate(x)
        q_x    = F.softmax(C(feat_x), dim=1)
        k_idx  = q_x.argmax(dim=1).item()

        # 2) encode + reconstruct
        with torch.no_grad():
            z_hat = E(x, torch.tensor([k_idx], device=device))
            x_hat = G(z_hat, torch.tensor([k_idx], device=device), target_hw=(H, W))

        # move to CPU/NumPy and ensure a channel dimension
        x_np    = x.squeeze(0).cpu().numpy()
        xhat_np = x_hat.squeeze(0).cpu().numpy()
        if x_np.ndim == 2:
            x_np    = x_np[np.newaxis, ...]
            xhat_np = xhat_np[np.newaxis, ...]

        class_str = "real" if lbl == 0 else "fake"
        print(f"[DEBUG] sample {i:2d}  label={lbl} ({class_str})  cluster k={k_idx}")

        # plot real vs. recon for each channel
        n_ch = x_np.shape[0]
        fig, axes = plt.subplots(2, n_ch, figsize=(4*n_ch, 6))
        if n_ch == 1:
            axes = axes.reshape(2, 1)
        for c in range(n_ch):
            axes[0, c].imshow(x_np[c], aspect="auto", origin="lower")
            axes[0, c].set_title(f"Real   (ch={c})")
            axes[0, c].axis("off")

            axes[1, c].imshow(xhat_np[c], aspect="auto", origin="lower")
            axes[1, c].set_title(f"Recon  (ch={c})\nk={k_idx}")
            axes[1, c].axis("off")

        plt.suptitle(f"Sample {i}  class={class_str}  true={lbl}")
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(os.path.join(debug_dir, f"sample_{i:02d}_class_{class_str}.png"))
        plt.close(fig)

    print("[DEBUG] Finished visualizing first 5 real and first 5 fake samples.")

    all_scores = []
    all_labels = []

    # process each batch
    with torch.no_grad():
        for x, label in loader:
            x = x.to(device)                                   # (B,C,H,W)
            # 1) cluster assignment q = C(D_i(x))
            feat_x = D.intermediate(x)
            q_x = F.softmax(C(feat_x), dim=1)                  # (B,K)
            k_pred = q_x.argmax(dim=1)                         # (B,)

            # 2) reconstruct
            z_hat = E(x, k_pred)
            x_hat = G(z_hat, k_pred, target_hw=(H, W))

            # 3) reconstruction error
            rec_err = torch.norm((x - x_hat).view(x.size(0), -1), dim=1)  # (B,)

            # 4) KL divergence term
            feat_xhat = D.intermediate(x_hat)
            q_xhat = F.softmax(C(feat_xhat), dim=1)
            kl = (q_x * (q_x.log() - q_xhat.log())).sum(dim=1)           # (B,)

            # combined anomaly score A (Eq.6) :contentReference[oaicite:0]{index=0}
            A = (1 - args.alpha) * rec_err + args.alpha * kl

            all_scores.append(A.cpu().numpy())
            all_labels.append(label.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # ROC & AUC
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

        # sweep thresholds and compute metrics
    records = []
    ts = np.linspace(scores.min(), scores.max(), args.n_thresholds)
    for t in ts:
        pred = (scores >= t).astype(int)
        acc  = accuracy_score(labels, pred)
        f1   = f1_score(labels, pred)
        rec  = recall_score(labels, pred)
        prec = precision_score(labels, pred)
        err  = 1 - acc
        records.append({
            "threshold":  t,
            "accuracy":   acc,
            "f1":         f1,
            "recall":     rec,
            "precision":  prec,
            "err_rate":   err
        })
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out_dir, "metrics_vs_threshold.csv"), index=False)

    # find best threshold by F1
    best_idx = df["f1"].idxmax()
    best     = df.iloc[best_idx]
    best_t   = best.threshold

    # Plot all four metrics vs threshold
    plt.figure(figsize=(8,5))
    plt.plot(ts, df["accuracy"],  label="Accuracy")
    plt.plot(ts, df["f1"],        label="F1 score")
    plt.plot(ts, df["recall"],    label="Recall")
    plt.plot(ts, df["precision"], label="Precision")
    plt.axvline(best_t, color="k", linestyle="--", label=f"Best F1 @ {best_t:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Metrics vs Threshold")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "metrics_vs_threshold.png"))
    plt.close()

    # Confusion matrix at best-F1 threshold
    best_pred = (scores >= best_t).astype(int)
    cm = confusion_matrix(labels, best_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix @ F1={best.f1:.3f}")
    plt.colorbar()
    tick_labels = ["real","anom"]
    plt.xticks([0,1], tick_labels)
    plt.yticks([0,1], tick_labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix_bestF1.png"))
    plt.close()

    df = pd.DataFrame(records)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "metrics_vs_threshold.csv"), index=False)

    # best threshold by F1
    best = df.iloc[df["f1"].idxmax()]
    best_t = best.threshold
    best_pred = (scores >= best_t).astype(int)
    cm = confusion_matrix(labels, best_pred)
    pd.DataFrame(cm, index=["real","anom"], columns=["pred_real","pred_anom"])\
      .to_csv(os.path.join(args.out_dir, "confusion_matrix.csv"))

    # ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.scatter([best.fpr if "fpr" in best else (fpr[np.abs(thr-best_t).argmin()])],
                [1-best.err_rate], color="red", label="Best F1")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "roc_curve.png"))
    plt.close()

    # summary
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(f"AUC:         {roc_auc:.4f}\n")
        f.write(f"Best F1:     {best.f1:.4f} @ thr={best.threshold:.4f}\n")
        f.write(f"Accuracy@F1: {best.accuracy:.4f}\n")
        f.write(f"ErrRate@F1:  {best.err_rate:.4f}\n")

    print(f"Inference done. AUC={roc_auc:.4f}, Best F1={best.f1:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="D-AnoGAN Anomaly Inference (mel, mfcc, or both)")
    p.add_argument("--real_mel_dir",  default=None, help="Real-class Mel .npy files")
    p.add_argument("--real_mfcc_dir", default=None, help="Real-class MFCC .npy files")
    p.add_argument("--fake_mel_dir",  default=None, help="Anomalous-class Mel .npy files")
    p.add_argument("--fake_mfcc_dir", default=None, help="Anomalous-class MFCC .npy files")
    p.add_argument("--model_dir",     required=True, help="Directory with G_*.pt, E_*.pt, D_*.pt, C_*.pt")
    p.add_argument("--ckpt",          required=True, help="Checkpoint suffix (e.g. epoch or 'final')")
    p.add_argument("--xzx_encoder", default=None,
               help="Path to encoder fine-tuned in phase 2 (E_xzx_final.pt)")

    p.add_argument("--out_dir",       default="inference_report")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--n_samples",     type=int,   default=None,
                   help="Max samples per class (real/fake)")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--z_dim",         type=int,   default=128)
    p.add_argument("--n_clusters",    type=int,   default=10)
    p.add_argument("--base_channels", type=int,   default=32)
    p.add_argument("--n_layers",      type=int,   default=3)
    p.add_argument("--alpha",         type=float, default=0.9,
                   help="Weighting in anomaly score Eq.(6)")
    p.add_argument("--n_thresholds",  type=int,   default=200)
    p.add_argument("--device",        default="cuda")
    args = p.parse_args()

    inference(args)
