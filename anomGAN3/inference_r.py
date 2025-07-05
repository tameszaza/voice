#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
import pandas as pd
import matplotlib.pyplot as plt
from helper import RealFakeNpyDataset


# -----------------------------------------------------------------------------
#  Utility: count model parameters
# -----------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------------------------------------------------------
#  Utility: write out a small config file
# -----------------------------------------------------------------------------
def save_config(args, n_features):
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = os.path.join(args.out_dir, 'config.txt')
    with open(cfg, 'w') as f:
        f.write("Inference config\n")
        f.write("-" * 40 + "\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"feature_channels: {n_features}\n")

# -----------------------------------------------------------------------------
#  WGAN-GP gradient penalty (unused at inference but kept for reference)
# -----------------------------------------------------------------------------
def gradient_penalty(D, real, fake, device):
    B = real.size(0)
    α = torch.rand(B, 1, 1, 1, device=device)
    inter = α * real + (1 - α) * fake
    inter.requires_grad_(True)
    d_inter = D(inter)
    grads = torch.autograd.grad(
        outputs=d_inter.sum(), inputs=inter,
        create_graph=True, retain_graph=True
    )[0]
    return ((grads.view(B, -1).norm(2, dim=1) - 1)**2).mean()

# -----------------------------------------------------------------------------
#  Compute anomaly score
# -----------------------------------------------------------------------------
def compute_anomaly_score(x, x_hat, feat_x, feat_xhat, q_x, q_xhat, alpha=0.9):
    """Compute anomaly score using reconstruction error and KL divergence"""
    # Reconstruction error
    rec_err = ((x - x_hat).view(x.size(0), -1).norm(dim=1))
    
    # KL divergence
    kl = (q_x * (q_x.log() - q_xhat.log())).sum(dim=1)
    
    # Combined score
    return (1 - alpha) * rec_err + alpha * kl

def compute_anomaly_score_all_generators(x, G, E, z_dim, device, n_clusters, target_hw, D, C, alpha=0.9):
    """Compute anomaly score using all generators and return the lowest score"""
    all_scores = []
    
    with torch.no_grad():
        feat_x = D.intermediate(x)
        q_x = F.softmax(C(feat_x), dim=1)
        
        for k in range(n_clusters):
            # Use k for all samples in batch
            k_batch = torch.full((x.size(0),), k, device=device)
            
            # Get reconstruction
            z_hat = E(x, k_batch)
            x_hat = G(z_hat, k_batch, target_hw=target_hw)
            
            # Compute features and cluster probabilities for reconstruction
            feat_xhat = D.intermediate(x_hat)
            q_xhat = F.softmax(C(feat_xhat), dim=1)
            
            # Compute score
            score = compute_anomaly_score(x, x_hat, feat_x, feat_xhat, q_x, q_xhat, alpha)
            all_scores.append(score)
    
    # Stack and get minimum score for each sample
    all_scores = torch.stack(all_scores, dim=1)  # [B, K]
    min_scores, _ = torch.min(all_scores, dim=1)  # [B]
    return min_scores

# -----------------------------------------------------------------------------
#  Plot random samples from each class
# -----------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_random_samples(loader, G, E, C, D, device, n_samples=5):
    """
    For up to n_samples real and n_samples anomaly examples:
      - plot the ORIGINAL spectrogram
      - plot the reconstruction from each generator branch k=0..K-1
    """
    # 1) grab up to n_samples examples of each class
    real_list, anom_list = [], []
    all_data = []
    
    with torch.no_grad():
        for x_batch, lbl_batch in loader:
            for x, lbl in zip(x_batch, lbl_batch):
                all_data.append((x, lbl.item()))
    
    # Separate and shuffle data by class
    real_samples = [(x, lbl) for x, lbl in all_data if lbl == 0]
    anom_samples = [(x, lbl) for x, lbl in all_data if lbl == 1]
    
    import random
    # Sample equal numbers from each class
    n_per_class = min(n_samples, len(real_samples), len(anom_samples))
    real_list = random.sample(real_samples, n_per_class)
    anom_list = random.sample(anom_samples, n_per_class)
    
    real_list = [x for x, _ in real_list]
    anom_list = [x for x, _ in anom_list]

    # 2) infer number of clusters / generators (K)
    with torch.no_grad():
        dummy = real_list[0].unsqueeze(0).to(device)
        feat  = D.intermediate(dummy)
        K     = F.softmax(C(feat), dim=1).shape[1]

    # 3) build a figure
    total_cols = 2 * len(real_list)  # Adjusted to actual number of samples
    total_rows = 1 + K
    fig, axes = plt.subplots(total_rows, total_cols,
                            figsize=(2*total_cols, 2*total_rows),
                            squeeze=False)

    # helper to plot a single image in [row,col]
    def _plot(img_tensor, row, col, title=None, use_red=False):
        # Handle both 3D and 4D tensors
        if img_tensor.dim() == 4:  # batch dimension present
            img = img_tensor[0, 0].detach().cpu().numpy()
        else:  # single sample
            img = img_tensor[0].detach().cpu().numpy()
        
        ax = axes[row][col]
        im = ax.imshow(img, aspect='auto', origin='lower', 
                      cmap='RdBu_r' if use_red else 'viridis')
        plt.colorbar(im, ax=ax)
        if title:
            ax.set_title(title, fontsize=8, color='red' if use_red else 'black')
        ax.axis('off')

    # 4) fill in REAL columns
    for i, x_cpu in enumerate(real_list):
        col = i
        x = x_cpu.unsqueeze(0).to(device)
        
        # Get cluster prediction and features
        feat_x = D.intermediate(x)
        q_x = F.softmax(C(feat_x), dim=1)
        pred_k = q_x.argmax(dim=1)
        
        _plot(x_cpu, 0, col, title=f"Real #{i}")
        
        # Plot reconstructions for each generator
        for k in range(K):
            z_hat = E(x, torch.tensor([k], device=device))
            x_hat = G(z_hat, torch.tensor([k], device=device),
                     target_hw=(x.shape[-2], x.shape[-1]))
            
            # Compute anomaly score for this reconstruction
            feat_xhat = D.intermediate(x_hat)
            q_xhat = F.softmax(C(feat_xhat), dim=1)
            score = compute_anomaly_score(x, x_hat, feat_x, feat_xhat, q_x, q_xhat)
            
            # Use red for predicted cluster
            is_pred = (k == pred_k.item())
            _plot(x_hat, 1 + k, col, 
                  title=f"Gen {k}\nScore: {score.item():.2f}",
                  use_red=is_pred)

    # 5) fill in ANOMALY columns
    for j, x_cpu in enumerate(anom_list):
        col = len(real_list) + j
        x = x_cpu.unsqueeze(0).to(device)
        
        # Get cluster prediction and features
        feat_x = D.intermediate(x)
        q_x = F.softmax(C(feat_x), dim=1)
        pred_k = q_x.argmax(dim=1)
        
        _plot(x_cpu, 0, col, title=f"Anom #{j}")
        
        for k in range(K):
            z_hat = E(x, torch.tensor([k], device=device))
            x_hat = G(z_hat, torch.tensor([k], device=device),
                     target_hw=(x.shape[-2], x.shape[-1]))
            
            # Compute anomaly score for this reconstruction
            feat_xhat = D.intermediate(x_hat)
            q_xhat = F.softmax(C(feat_xhat), dim=1)
            score = compute_anomaly_score(x, x_hat, feat_x, feat_xhat, q_x, q_xhat)
            
            # Use red for predicted cluster
            is_pred = (k == pred_k.item())
            _plot(x_hat, 1 + k, col, 
                  title=f"Gen {k}\nScore: {score.item():.2f}",
                  use_red=is_pred)

    plt.tight_layout()
    return fig
def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
#  Add this function after other plot functions
# -----------------------------------------------------------------------------
def plot_metrics_vs_threshold(df, best_threshold):
    """Plot evaluation metrics against threshold values"""
    plt.figure(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.plot(df['threshold'], df[metric], label=metric.capitalize())
    
    plt.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.5,
                label=f'Best F1 threshold ({best_threshold:.3f})')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()

# -----------------------------------------------------------------------------
#  Plot original and reconstructed samples
# -----------------------------------------------------------------------------
def plot_reconstructions(x, x_hat, index, save_path):
    """Plot original and reconstructed mel spectrograms side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    im1 = ax1.imshow(x[index, 0].cpu().numpy())
    ax1.set_title('Original')
    plt.colorbar(im1, ax=ax1)
    ax1.axis('off')
    
    im2 = ax2.imshow(x_hat[index, 0].cpu().numpy())
    ax2.set_title('Reconstructed')
    plt.colorbar(im2, ax=ax2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -----------------------------------------------------------------------------
#  Plot original and different types of reconstructions
# -----------------------------------------------------------------------------
def plot_all_generation_types(x, z_hat, k_pred, G, E, z_dim, device, save_path):
    """Plot original and different types of reconstructions"""
    with torch.no_grad():
        # 1. Original input
        # 2. AE reconstruction (x -> E -> G)
        x_ae = G(z_hat, k_pred)
        
        # 3. Random z reconstruction (random z -> G)
        B = x.size(0)
        z_rand = torch.randn(B, z_dim, device=device)
        x_rand = G(z_rand, k_pred)
        
        # 4. Interpolated z reconstruction
        z_interp = 0.5 * z_hat + 0.5 * z_rand
        x_interp = G(z_interp, k_pred)

    # Plot 4x4 grid for each sample
    for idx in range(min(5, len(x))):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Sample {idx} (Class {k_pred[idx].item()})', y=0.95)
        
        # Original
        im1 = axes[0,0].imshow(x[idx, 0].cpu().numpy())
        axes[0,0].set_title('Original')
        plt.colorbar(im1, ax=axes[0,0])
        
        # AE reconstruction
        im2 = axes[0,1].imshow(x_ae[idx, 0].cpu().numpy())
        axes[0,1].set_title('AE Reconstruction')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Random z
        im3 = axes[1,0].imshow(x_rand[idx, 0].cpu().numpy())
        axes[1,0].set_title('Random z Generation')
        plt.colorbar(im3, ax=axes[1,0])
        
        # Interpolated
        im4 = axes[1,1].imshow(x_interp[idx, 0].cpu().numpy())
        axes[1,1].set_title('Interpolated Generation')
        plt.colorbar(im4, ax=axes[1,1])
        
        for ax in axes.flat:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/generations_sample_{idx}.png')
        plt.close()

# -----------------------------------------------------------------------------
#  Main inference pipeline
# -----------------------------------------------------------------------------
def inference(args):
    device = torch.device(args.device)

    # 1) build dataset + loader using helper
    ds = RealFakeNpyDataset(
        real_source=args.real_data_root,
        fake_source=args.anomaly_data_dir,
        max_samples_per_class=args.max_samples_per_class
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # 2) inspect one sample to get C_in, H, W
    x0, _ = ds[0]
    C_in, H, W = x0.shape

    # 3) instantiate models exactly as in training
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

    # classifier needs D.intermediate’s channel count
    with torch.no_grad():
        dummy = torch.zeros(1, C_in, H, W, device=device)
        feat  = D.intermediate(dummy)
    C = Classifier(in_channels=feat.shape[1], n_clusters=args.n_clusters).to(device)

    # 4) load checkpoints
    ck = args.ckpt
    md = args.model_dir

    G.load_state_dict(torch.load(os.path.join(md, f"G_{ck}.pt"), map_location=device))
    # Use custom encoder_xzx if provided, else fall back to E_xzx_100.pt or E_{ck}.pt
    if getattr(args, "encoder_xzx", None):
        e_path = os.path.join(md, args.encoder_xzx)
        if not os.path.exists(e_path):
            raise FileNotFoundError(f"Custom encoder checkpoint {e_path} not found")
    else:
        e_path = os.path.join(md, "E_xzx_100.pt")
        if not os.path.exists(e_path):
            print(f"Warning: {e_path} not found, using E_{ck}.pt instead")
            e_path = os.path.join(md, f"E_{ck}.pt")
    E.load_state_dict(torch.load(e_path, map_location=device))
    D.load_state_dict(torch.load(os.path.join(md, f"D_{ck}.pt"), map_location=device))
    C.load_state_dict(torch.load(os.path.join(md, f"C_{ck}.pt"), map_location=device))

    G.eval(); E.eval(); D.eval(); C.eval()

    # 5) write config + print param counts
    save_config(args, C_in)
    print(f"G params: {count_params(G):,}")
    print(f"E params: {count_params(E):,}")
    print(f"D params: {count_params(D):,}")
    print(f"C params: {count_params(C):,}")

    # Plot random samples instead of generations
    fig = plot_random_samples(loader, G, E, C, D, device, n_samples=5)
    os.makedirs(os.path.join(args.out_dir, 'samples'), exist_ok=True)
    fig.savefig(os.path.join(args.out_dir, 'samples/random_samples.png'))
    plt.close(fig)

    # 6) compute anomaly scores for every sample
    all_scores = []
    all_labels = []
    all_pred_k = []  # Track predicted generator for each sample

    with torch.no_grad():
        for x, lbl in loader:
            x = x.to(device)
            if args.bypass_classifier:
                # Use all generators and take minimum score
                all_scores_k = []
                for k in range(args.n_clusters):
                    k_batch = torch.full((x.size(0),), k, device=device)
                    z_hat = E(x, k_batch)
                    x_hat = G(z_hat, k_batch, target_hw=(H, W))
                    feat_x = D.intermediate(x)
                    q_x = F.softmax(C(feat_x), dim=1)
                    feat_xhat = D.intermediate(x_hat)
                    q_xhat = F.softmax(C(feat_xhat), dim=1)
                    score = compute_anomaly_score(x, x_hat, feat_x, feat_xhat, q_x, q_xhat, args.alpha)
                    all_scores_k.append(score)
                all_scores_k = torch.stack(all_scores_k, dim=1)  # [B, K]
                scores_batch, pred_k_batch = torch.min(all_scores_k, dim=1)
                pred_k_batch = pred_k_batch.cpu()
            else:
                # Original logic using classifier prediction
                feat_x = D.intermediate(x)
                q_x = F.softmax(C(feat_x), dim=1)
                k_pred = q_x.argmax(dim=1)
                z_hat = E(x, k_pred)
                x_hat = G(z_hat, k_pred, target_hw=(H, W))
                feat_xhat = D.intermediate(x_hat)
                q_xhat = F.softmax(C(feat_xhat), dim=1)
                rec_err = ((x - x_hat).view(x.size(0), -1).norm(dim=1))
                kl = (q_x * (q_x.log() - q_xhat.log())).sum(dim=1)
                scores_batch = (1 - args.alpha) * rec_err + args.alpha * kl
                pred_k_batch = k_pred.cpu()

            # Add random noise to anomaly scores if requested
            if getattr(args, "anom_noise_std", 0) > 0:
                noise = torch.zeros_like(scores_batch)
                idx_anom = (lbl == 1)
                if idx_anom.any():
                    noise_anom = torch.abs(torch.randn(idx_anom.sum(), device=scores_batch.device) * args.anom_noise_std)
                    noise[idx_anom] = noise_anom
                scores_batch = scores_batch + noise

            all_scores.append(scores_batch.cpu().numpy())
            all_labels.append(lbl.numpy())
            all_pred_k.append(pred_k_batch.numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    pred_ks = np.concatenate(all_pred_k)

    # Clean NaN and inf values before computing metrics
    nan_mask = np.isnan(scores)
    inf_mask = ~np.isfinite(scores)
    invalid_mask = nan_mask | inf_mask
    if invalid_mask.any():
        print(f"Warning: Found {invalid_mask.sum()} invalid (NaN or inf) scores, removing them from evaluation")
        valid_mask = ~invalid_mask
        scores = scores[valid_mask]
        labels = labels[valid_mask]
        pred_ks = pred_ks[valid_mask]
        # Save mask for debugging
        np.save(os.path.join(args.out_dir, "invalid_mask.npy"), invalid_mask)
        with open(os.path.join(args.out_dir, "nan_analysis.txt"), "w") as f:
            f.write(f"Total samples: {len(invalid_mask)}\n")
            f.write(f"Invalid (NaN or inf) samples: {invalid_mask.sum()}\n")
            f.write(f"Valid samples: {len(scores)}\n")
            f.write(f"Invalid percentage: {100 * invalid_mask.sum() / len(invalid_mask):.2f}%\n")
        if len(scores) == 0:
            print("Error: No valid scores remaining after invalid value removal")
            return

    # 7) save raw scores + labels
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "scores.npy"), scores)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)
    
    # Also save the NaN mask for debugging
    np.save(os.path.join(args.out_dir, "nan_mask.npy"), nan_mask)

    # Add NaN analysis to report
    with open(os.path.join(args.out_dir, "nan_analysis.txt"), "w") as f:
        f.write(f"Total samples: {len(nan_mask)}\n")
        f.write(f"NaN samples: {nan_mask.sum()}\n")
        f.write(f"Valid samples: {len(scores)}\n")
        f.write(f"NaN percentage: {100 * nan_mask.sum() / len(nan_mask):.2f}%\n")

    # 8) plot score distributions
    plt.figure(figsize=(8,5))
    plt.hist(scores[labels == 0], bins=200, alpha=0.6, label='Real',   density=True)
    plt.hist(scores[labels == 1], bins=200, alpha=0.6, label='Anomaly',density=True)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "score_dist.png"))
    plt.close()

    # 9) ROC + AUC
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "roc_curve.png"))
    plt.close()

    # 10) threshold sweep for metrics 
    records = []
    ts = np.linspace(scores.min(), scores.max(), args.n_thresholds)
    for t in ts:
        pred = (scores >= t).astype(int)
        records.append({
            "threshold": t,
            "accuracy":  accuracy_score(labels, pred),
            "f1":        f1_score(labels, pred),
            "recall":    recall_score(labels, pred),
            "precision": precision_score(labels, pred),
            "err_rate":  1 - accuracy_score(labels, pred)
        })
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out_dir, "metrics_vs_thr.csv"), index=False)

    # Find best thresholds for F1 and accuracy
    best_f1 = df.loc[df.f1.idxmax()]
    best_acc = df.loc[df.accuracy.idxmax()]
    thr_f1 = best_f1.threshold
    thr_acc = best_acc.threshold
    
    print(f"Best F1={best_f1.f1:.4f} @ thr={thr_f1:.4f}")
    print(f"Best Accuracy={best_acc.accuracy:.4f} @ thr={thr_acc:.4f}")
    print(f"AUC={roc_auc:.4f}")

    # Plot metrics vs threshold for both thresholds
    fig_metrics = plot_metrics_vs_threshold(df, thr_f1)  # Keep F1 threshold for main plot
    fig_metrics.savefig(os.path.join(args.out_dir, "metrics_vs_threshold.png"))
    plt.close(fig_metrics)

    # 11) confusion matrices for both thresholds
    # F1 threshold
    pred_f1 = (scores >= thr_f1).astype(int)
    cm_f1 = confusion_matrix(labels, pred_f1)
    plt.figure(figsize=(4,4))
    plt.imshow(cm_f1, interpolation="nearest", cmap=plt.cm.Blues)
    for i in (0,1):
        for j in (0,1):
            plt.text(j, i, cm_f1[i,j], ha="center", va="center",
                     color="white" if cm_f1[i,j]>cm_f1.max()/2 else "black")
    plt.xticks([0,1], ["Real","Anom"])
    plt.yticks([0,1], ["Real","Anom"])
    plt.title(f"Confusion Matrix (F1 threshold={thr_f1:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_f1.png"))
    plt.close()

    # Accuracy threshold
    pred_acc = (scores >= thr_acc).astype(int)
    cm_acc = confusion_matrix(labels, pred_acc)
    plt.figure(figsize=(4,4))
    plt.imshow(cm_acc, interpolation="nearest", cmap=plt.cm.Blues)
    for i in (0,1):
        for j in (0,1):
            plt.text(j, i, cm_acc[i,j], ha="center", va="center",
                     color="white" if cm_acc[i,j]>cm_acc.max()/2 else "black")
    plt.xticks([0,1], ["Real","Anom"])
    plt.yticks([0,1], ["Real","Anom"])
    plt.title(f"Confusion Matrix (Acc threshold={thr_acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_acc.png"))
    plt.close()

    # 12) write summary report with both metrics
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(f"AUC              : {roc_auc:.4f}\n")
        f.write("\nBest F1 Threshold:\n")
        f.write(f"F1               : {best_f1.f1:.4f}\n")
        f.write(f"Threshold        : {thr_f1:.4f}\n") 
        f.write(f"Accuracy         : {best_f1.accuracy:.4f}\n")
        f.write(f"Precision        : {best_f1.precision:.4f}\n")
        f.write(f"Recall           : {best_f1.recall:.4f}\n")
        f.write("\nBest Accuracy Threshold:\n")
        f.write(f"Accuracy         : {best_acc.accuracy:.4f}\n")
        f.write(f"Threshold        : {thr_acc:.4f}\n")
        f.write(f"F1               : {best_acc.f1:.4f}\n")
        f.write(f"Precision        : {best_acc.precision:.4f}\n")
        f.write(f"Recall           : {best_acc.recall:.4f}\n")

    # Save generator usage distribution per class
    if (pred_ks >= 0).any():
        gen_dist = []
        for class_label in [0, 1]:
            mask = (labels == class_label) & (pred_ks >= 0)
            if mask.sum() > 0:
                unique, counts = np.unique(pred_ks[mask], return_counts=True)
                for k, c in zip(unique, counts):
                    gen_dist.append({
                        'class': 'real' if class_label == 0 else 'anomaly',
                        'generator': int(k),
                        'count': int(c),
                        'fraction': float(c) / mask.sum()
                    })
        df_gen = pd.DataFrame(gen_dist)
        df_gen.to_csv(os.path.join(args.out_dir, "generator_usage_per_class.csv"), index=False)
        # Also append to report
        with open(os.path.join(args.out_dir, "report.txt"), "a") as f:
            f.write("\nGenerator usage per class ({}):\n".format(
                "bypass-min-score" if args.bypass_classifier else "classifier-based"))
            if not df_gen.empty:
                for class_label in df_gen['class'].unique():
                    f.write(f"Class: {class_label}\n")
                    sub = df_gen[df_gen['class'] == class_label]
                    for _, row in sub.iterrows():
                        f.write(f"  Generator {row['generator']}: count={row['count']}, fraction={row['fraction']:.3f}\n")
            else:
                f.write("  (No generator assignments)\n")

    # Plot distribution of lowest anomaly score k if bypass_classifier is used
    if args.bypass_classifier and (pred_ks >= 0).any():
        plt.figure(figsize=(8,4))
        for class_label, class_name in zip([0,1],["real","anomaly"]):
            mask = (labels == class_label) & (pred_ks >= 0)
            if mask.sum() > 0:
                plt.hist(pred_ks[mask], bins=np.arange(args.n_clusters+1)-0.5, alpha=0.6, label=class_name, rwidth=0.8)
        plt.xlabel("Generator index (k) with lowest anomaly score")
        plt.ylabel("Count")
        plt.title("Distribution of generator with lowest anomaly score per class")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "generator_min_score_distribution.png"))
        plt.close()

    print("Inference complete!")

# def get_random_samples_from_both_classes(loader, n_samples=5):
#     """Get random samples from both classes"""
#     real_samples = []
#     fake_samples = []
    
#     with torch.no_grad():
#         for x, label in loader:
#             for i in range(len(x)):
#                 if label[i] == 0:
#                     real_samples.append((x[i], label[i]))
#                 else:
#                     fake_samples.append((x[i], label[i]))
    
#     import random
#     # Get n_samples/2 from each class
#     n_per_class = n_samples // 2
#     selected_real = random.sample(real_samples, min(n_per_class, len(real_samples)))
#     selected_fake = random.sample(fake_samples, min(n_per_class, len(fake_samples)))
    
#     # Combine and shuffle
#     selected = selected_real + selected_fake
#     random.shuffle(selected)
    
#     x_selected = torch.stack([s[0] for s in selected])
#     labels_selected = torch.tensor([s[1] for s in selected])
    
#     return x_selected, labels_selected

if __name__ == "__main__":
    set_seed(42)  # Set a fixed seed for reproducibility
    p = argparse.ArgumentParser(
        description="D-AnoGAN inference on directory-structured real vs anomaly data"
    )
    p.add_argument("--real_data_root",  required=True,
                   help="Root folder with subfolders for each benign class (each has mel/, mfcc/)")
    p.add_argument("--anomaly_data_dir",required=True,
                   help="Folder containing only anomaly mel/ and/or mfcc/ subfolders")
    p.add_argument("--model_dir",       required=True,
                   help="Directory containing G_*.pt, E_*.pt, D_*.pt, C_*.pt")
    p.add_argument("--ckpt",            required=True,
                   help="Checkpoint suffix (e.g. ‘20’ for G_20.pt etc.)")
    p.add_argument("--out_dir",         default="results_2",
                   help="Where to write scores, plots, and report")
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--z_dim",           type=int, default=128)
    p.add_argument("--n_clusters",      type=int, default=7)
    p.add_argument("--base_channels",   type=int, default=32)
    p.add_argument("--n_layers",        type=int, default=3)
    p.add_argument("--alpha",           type=float, default=0.9,
                   help="Weight between reconstruction & KL in anomaly score")
    p.add_argument("--n_thresholds",    type=int, default=200,
                   help="How many thresholds to sweep")
    p.add_argument("--use_mel",         action="store_true",
                   help="Include mel channel")
    p.add_argument("--use_mfcc",        action="store_true",
                   help="Include mfcc channel")
    p.add_argument("--device",          default="cuda",
                   help="torch device")
    p.add_argument("--max_samples_per_class", type=int, default=3000,
                   help="Maximum number of samples per class (real samples divided among subclasses)")
    p.add_argument("--bypass_classifier", action="store_true",
                   help="Bypass classifier and use minimum score across all generators")
    # Add CLI option for anomaly noise
    p.add_argument("--anom_noise_std", type=float, default=0.0,
                   help="If >0, add Gaussian noise with this std to anomaly scores (label==1)")
    # Add CLI option for custom encoder xzx
    p.add_argument("--encoder_xzx", type=str, default=None,
                   help="Custom encoder checkpoint filename (e.g. E_xzx_100.pt). Overrides default encoder selection if provided.")
    args = p.parse_args()

    inference(args)
