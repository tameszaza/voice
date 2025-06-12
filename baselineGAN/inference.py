#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Import base dataset class from anomGAN
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anomGAN.inference import RealFakeAudioDataset

from models import Generator, Encoder, Discriminator

def inference(args):
    device = torch.device(args.device)

    ds = RealFakeAudioDataset(
        real_mel_dir=args.real_mel_dir,
        real_mfcc_dir=args.real_mfcc_dir,
        fake_mel_dir=args.fake_mel_dir,
        fake_mfcc_dir=args.fake_mfcc_dir,
        n_samples=args.n_samples,
        seed=args.seed
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    x0, _ = ds[0]
    n_features, H, W = x0.shape

    # Initialize models
    G = Generator(args.z_dim, n_features, args.base_channels, H, args.n_layers).to(device)
    E = Encoder(n_features, args.z_dim, args.base_channels, H, args.n_layers).to(device)
    D = Discriminator(n_features, args.base_channels, args.n_layers).to(device)

    # Load checkpoints
    G.load_state_dict(torch.load(os.path.join(args.model_dir, f"G_{args.ckpt}.pt"), map_location=device))
    E.load_state_dict(torch.load(os.path.join(args.model_dir, f"E_{args.ckpt}.pt"), map_location=device))
    D.load_state_dict(torch.load(os.path.join(args.model_dir, f"D_{args.ckpt}.pt"), map_location=device))

    G.eval(); E.eval(); D.eval()

    all_scores = []
    all_labels = []

    # Process batches
    with torch.no_grad():
        for x, label in loader:
            x = x.to(device)
            
            # Get reconstruction
            z_hat = E(x)
            x_hat = G(z_hat, target_hw=(H, W))
            
            # Plot a few reconstructions for visualization
            if args.plot_reconstructions:
                for i in range(min(5, x.size(0))):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.imshow(x[i,0].cpu().numpy())
                    ax1.set_title("Original")
                    ax2.imshow(x_hat[i,0].cpu().numpy())
                    ax2.set_title("Reconstructed")
                    plt.savefig(os.path.join(args.out_dir, f"reconstruction_{i}.png"))
                    plt.close()
            
            # Compute anomaly score as weighted sum of:
            # 1) Reconstruction error
            rec_err = torch.norm((x - x_hat).view(x.size(0), -1), dim=1)
            # 2) Discriminator feature difference 
            feat_x = D.intermediate(x)
            feat_xhat = D.intermediate(x_hat)
            feat_diff = torch.norm((feat_x - feat_xhat).view(x.size(0), -1), dim=1)
            
            # Combined score
            score = (1 - args.alpha) * rec_err + args.alpha * feat_diff
            
            all_scores.append(score.cpu().numpy())
            all_labels.append(label.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # Plot detailed sample visualizations for 5 random samples
    np.random.seed(args.seed)
    viz_indices = np.random.choice(len(scores), size=5, replace=False)
    
    # Re-process selected samples for visualization
    with torch.no_grad():
        for idx in viz_indices:
            x, label = ds[idx]
            x = x.unsqueeze(0).to(device)
            
            # Get reconstruction
            z_hat = E(x)
            x_hat = G(z_hat, target_hw=(H, W))
            
            # Compute individual scores
            rec_err = torch.norm((x - x_hat).view(x.size(0), -1), dim=1)
            feat_x = D.intermediate(x)
            feat_xhat = D.intermediate(x_hat)
            feat_diff = torch.norm((feat_x - feat_xhat).view(x.size(0), -1), dim=1)
            score = (1 - args.alpha) * rec_err + args.alpha * feat_diff
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.imshow(x[0,0].cpu().numpy())
            ax1.set_title(f"Original (Label: {'Fake' if label else 'Real'})")
            ax2.imshow(x_hat[0,0].cpu().numpy())
            ax2.set_title(f"Reconstructed\nScore: {score.item():.4f}")
            plt.suptitle(f"Sample {idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"detailed_sample_{idx}.png"))
            plt.close()

    # Sample 5 random indices for debugging
    np.random.seed(args.seed)
    debug_indices = np.random.choice(len(scores), size=5, replace=False)
    print("\nDebug samples:")
    for idx in debug_indices:
        print(f"Sample {idx}: Score={scores[idx]:.4f}, Label={labels[idx]}")
    print()

    # ROC & AUC
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    # Compute metrics at different thresholds
    records = []
    ts = np.linspace(scores.min(), scores.max(), args.n_thresholds)
    for t in ts:
        pred = (scores >= t).astype(int)
        records.append({
            "threshold": t,
            "accuracy": accuracy_score(labels, pred),
            "f1": f1_score(labels, pred),
            "recall": recall_score(labels, pred),
            "precision": precision_score(labels, pred),
            "err_rate": 1 - accuracy_score(labels, pred)
        })

    df = pd.DataFrame(records)
    best_f1 = df.iloc[df["f1"].idxmax()]
    best_acc = df.iloc[df["accuracy"].idxmax()]
    best_f1_t = best_f1.threshold
    best_acc_t = best_acc.threshold

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "metrics_vs_threshold.csv"), index=False)

    # Plot metrics
    plt.figure(figsize=(8,5))
    for metric in ["accuracy", "f1", "recall", "precision"]:
        plt.plot(ts, df[metric], label=metric.capitalize())
    plt.axvline(best_f1_t, color="k", linestyle="--", label=f"Best F1 @ {best_f1_t:.4f}")
    plt.axvline(best_acc_t, color="r", linestyle="--", label=f"Best Acc @ {best_acc_t:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Metrics vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "metrics_vs_threshold.png"))
    plt.close()

    # Confusion matrices at best thresholds
    pred_f1 = (scores >= best_f1_t).astype(int)
    pred_acc = (scores >= best_acc_t).astype(int)
    
    cm_f1 = confusion_matrix(labels, pred_f1)
    cm_acc = confusion_matrix(labels, pred_acc)
    
    pd.DataFrame(cm_f1, index=["real","anom"], columns=["pred_real","pred_anom"])\
      .to_csv(os.path.join(args.out_dir, "confusion_matrix_f1.csv"))
    pd.DataFrame(cm_acc, index=["real","anom"], columns=["pred_real","pred_anom"])\
      .to_csv(os.path.join(args.out_dir, "confusion_matrix_acc.csv"))

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "roc_curve.png"))
    plt.close()

    # Plot score distributions
    plt.figure(figsize=(10, 6))
    plt.hist(scores[labels == 0], bins=50, alpha=0.5, label='Real', density=True)
    plt.hist(scores[labels == 1], bins=50, alpha=0.5, label='Fake', density=True)
    plt.axvline(best_f1_t, color='k', linestyle='--', label=f'Best F1 thresh')
    plt.axvline(best_acc_t, color='r', linestyle='--', label=f'Best Acc thresh')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "score_distribution.png"))
    plt.close()

    # Update summary report
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(f"AUC:            {roc_auc:.4f}\n")
        f.write("\nBest F1 Metrics:\n")
        f.write(f"F1:             {best_f1.f1:.4f} @ thr={best_f1.threshold:.4f}\n")
        f.write(f"Accuracy:       {best_f1.accuracy:.4f}\n")
        f.write(f"Error Rate:     {best_f1.err_rate:.4f}\n")
        f.write(f"Precision:      {best_f1.precision:.4f}\n")
        f.write(f"Recall:         {best_f1.recall:.4f}\n")
        f.write("\nBest Accuracy Metrics:\n")
        f.write(f"Accuracy:       {best_acc.accuracy:.4f} @ thr={best_acc.threshold:.4f}\n")
        f.write(f"F1:             {best_acc.f1:.4f}\n")
        f.write(f"Error Rate:     {best_acc.err_rate:.4f}\n")
        f.write(f"Precision:      {best_acc.precision:.4f}\n")
        f.write(f"Recall:         {best_acc.recall:.4f}\n")

    print(f"Inference complete.")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Best F1: {best_f1.f1:.4f} @ thr={best_f1.threshold:.4f}")
    print(f"Best Acc: {best_acc.accuracy:.4f} @ thr={best_acc.threshold:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Baseline AnoGAN Inference")
    p.add_argument("--real_mel_dir", default=None)
    p.add_argument("--real_mfcc_dir", default=None)  
    p.add_argument("--fake_mel_dir", default=None)
    p.add_argument("--fake_mfcc_dir", default=None)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="baseline_inference2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--n_thresholds", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--plot_reconstructions", action="store_true", 
                  help="Plot sample reconstructions")
    args = p.parse_args()

    inference(args)
