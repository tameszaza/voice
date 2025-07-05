#!/usr/bin/env python3
"""
Load a single .npy dataset, cluster into 8 balanced groups,
save diagnostic plots, and write out each cluster’s samples
to its own .npy file. Now includes a 3-D PCA scatter plot.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

from helper import SingleNpyDataset

def balanced_kmeans(X: np.ndarray, n_clusters: int, random_state=42) -> np.ndarray:
    N, D = X.shape
    km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    centers = km.cluster_centers_
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    base = N // n_clusters
    rem  = N % n_clusters
    desired = np.array([base+1]*rem + [base]*(n_clusters - rem))

    labels = -1 * np.ones(N, dtype=int)
    counts = np.zeros(n_clusters, dtype=int)

    flat = [(dists[i, j], i, j)
            for i in range(N) for j in range(n_clusters)]
    flat.sort(key=lambda x: x[0])

    for dist, i, j in flat:
        if labels[i] != -1:
            continue
        if counts[j] < desired[j]:
            labels[i] = j
            counts[j] += 1
        if np.all(counts == desired):
            break

    return labels

def main():
    # ── CONFIG ────────────────────────────────────────────────
    npy_path   = '../ResData/wavefake128_2048split/train/real.npy'  # replace with your .npy path
    n_clusters = 8

    # ── LOAD dataset ─────────────────────────────────────────
    ds = SingleNpyDataset(npy_file=npy_path, label=None)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    X_tensor = next(iter(loader))  # shape: (N, C, H, W) or (N, D)

    # flatten to (N, D)
    if X_tensor.ndim > 2:
        X_flat = X_tensor.view(X_tensor.size(0), -1).numpy()
    else:
        X_flat = X_tensor.numpy()

    # ── CLUSTER with balance ──────────────────────────────────
    labels = balanced_kmeans(X_flat, n_clusters)

    # ── REPORT cluster sizes ─────────────────────────────────
    sizes = np.bincount(labels, minlength=n_clusters)
    for k, sz in enumerate(sizes):
        print(f"Cluster {k:2d}: {sz} samples")

    # ── SAVE cluster labels ──────────────────────────────────
    np.save('cluster_labels.npy', labels)
    print("Saved labels to cluster_labels.npy")

    # ── MAKE output directories ───────────────────────────────
    os.makedirs('plots', exist_ok=True)
    os.makedirs('clusters', exist_ok=True)

    # ── 1) Bar chart of cluster sizes ─────────────────────────
    plt.figure(figsize=(8, 4))
    plt.bar(range(n_clusters), sizes, edgecolor='k')
    plt.xticks(range(n_clusters))
    plt.xlabel('Cluster')
    plt.ylabel('Number of samples')
    plt.title('Balanced K-Means: Cluster Sizes')
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'cluster_sizes.png'))
    plt.close()

    # ── 2) PCA scatter colored by cluster (3-D) ───────────────
    # Replace your 2-D code with this block
    pca = PCA(n_components=3, svd_solver='randomized')
    X3 = pca.fit_transform(X_flat)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X3[:, 0], X3[:, 1], X3[:, 2],
        c=labels,
        cmap='tab10',
        s=15,
        alpha=0.6
    )
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_clusters))
    cbar.set_label('Cluster')
    ax.set_title('Balanced K-Means Clusters (PCA 3-D projection)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'cluster_pca_3d.png'))
    plt.close()

    # ── 3) SAVE each cluster’s raw samples to its own .npy ──────
    raw_data = np.load(npy_path, mmap_mode='r')  # shape (N, ...)

    for k in range(n_clusters):
        mask = (labels == k)
        cluster_raw = raw_data[mask]
        out_path = os.path.join('clusters', f'cluster_{k}.npy')
        np.save(out_path, cluster_raw)
        print(f"Saved {cluster_raw.shape[0]} raw samples to {out_path}")

    print("All clusters saved under `clusters/` and plots under `plots/`.")

if __name__ == "__main__":
    main()
