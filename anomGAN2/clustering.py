#!/usr/bin/env python3
"""
Cluster two separate sets of .npy samples (mel and mfcc) into balanced groups,
then copy each file into clusters/cluster{0..K}/[mel|mfcc] folders.

Usage:
    python cluster_mel_mfcc.py
"""

import os
import glob
import shutil

import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 for 3d scatter


def balanced_kmeans(X: np.ndarray, n_clusters: int, random_state=42) -> np.ndarray:
    """
    Perform KMeans but force each cluster to have nearly the same number of points.
    Returns an array of labels (shape N,).
    """
    N, _ = X.shape
    km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    centers = km.cluster_centers_
    # compute distance of every point to every center
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    base = N // n_clusters
    rem  = N % n_clusters
    # how many samples each cluster should get
    desired = np.array([base + 1] * rem + [base] * (n_clusters - rem))

    labels = -1 * np.ones(N, dtype=int)
    counts = np.zeros(n_clusters, dtype=int)

    # sort all (point,cluster) pairs by distance ascending
    pairs = [(dists[i, j], i, j) for i in range(N) for j in range(n_clusters)]
    pairs.sort(key=lambda x: x[0])

    # assign each point to its nearest cluster that still has room
    for dist, i, j in pairs:
        if labels[i] != -1:
            continue
        if counts[j] < desired[j]:
            labels[i] = j
            counts[j] += 1
        if np.all(counts == desired):
            break

    return labels


class NpyFileDataset(Dataset):
    """
    Treat each .npy file in a folder as one sample.
    Returns (array, filepath).
    """
    def __init__(self, dir_path):
        self.files = sorted(glob.glob(os.path.join(dir_path, '*.npy')))
        if not self.files:
            raise ValueError(f"No .npy files in {dir_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr  = np.load(path)
        return arr, path


def collate_fn(batch):
    """
    Batch is list of (array, path). Stack arrays into one big numpy array
    and collect paths in a list.
    """
    arrays, paths = zip(*batch)
    big = np.stack(arrays, axis=0)
    return big, list(paths)


def cluster_folder(input_dir, n_clusters):
    """
    Load all .npy from input_dir, flatten them, run balanced_kmeans,
    and return (labels, list_of_paths, flattened_data) for plotting.
    """
    ds     = NpyFileDataset(input_dir)
    loader = DataLoader(
        ds,
        batch_size = len(ds),
        shuffle    = False,
        collate_fn = collate_fn
    )
    X_all, paths = next(iter(loader))  # X_all shape (N,...)
    N = X_all.shape[0]
    # flatten to (N, D)
    if X_all.ndim > 2:
        X_flat = X_all.reshape(N, -1)
    else:
        X_flat = X_all
    labels = balanced_kmeans(X_flat, n_clusters)
    return labels, paths, X_flat


def main():
    # ------------ USER CONFIGURATION ------------
    mel_dir  = 'data/real/data_256_real/mel'
    mfcc_dir = 'data/real/data_256_real/mfcc'
    n_clusters = 8
    # --------------------------------------------

    # cluster both folders
    print("Clustering mel files...")
    labels_mel,  paths_mel,  X_mel  = cluster_folder(mel_dir,  n_clusters)
    print("Clustering mfcc files...")
    labels_mfcc, paths_mfcc, X_mfcc = cluster_folder(mfcc_dir, n_clusters)

    # report sizes
    sizes_mel  = np.bincount(labels_mel,  minlength=n_clusters)
    sizes_mfcc = np.bincount(labels_mfcc, minlength=n_clusters)
    print("\nCluster sizes (mel):")
    for k, sz in enumerate(sizes_mel):
        print(f"  mel cluster {k}: {sz}")
    print("\nCluster sizes (mfcc):")
    for k, sz in enumerate(sizes_mfcc):
        print(f"  mfcc cluster {k}: {sz}")

    # save label arrays
    np.save('cluster_labels_mel.npy',  labels_mel)
    np.save('cluster_labels_mfcc.npy', labels_mfcc)
    print("\nSaved cluster_labels_mel.npy and cluster_labels_mfcc.npy")

    # prepare output directories
    os.makedirs('plots', exist_ok=True)
    for k in range(n_clusters):
        os.makedirs(f'clusters/cluster_{k}/mel',  exist_ok=True)
        os.makedirs(f'clusters/cluster_{k}/mfcc', exist_ok=True)

    # 1) bar charts
    plt.figure(figsize=(8,4))
    plt.bar(range(n_clusters), sizes_mel,  label='mel',  edgecolor='k', alpha=0.6)
    plt.bar(range(n_clusters), sizes_mfcc, label='mfcc', edgecolor='k', alpha=0.6)
    plt.xlabel('Cluster')
    plt.ylabel('Number of samples')
    plt.title('Cluster sizes for mel and mfcc')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/cluster_sizes_both.png')
    plt.close()

    # 2) PCA 3D scatter for mel
    pca_mel = PCA(n_components=3, svd_solver='randomized')
    mel3   = pca_mel.fit_transform(X_mel)
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        mel3[:,0], mel3[:,1], mel3[:,2],
        c=labels_mel, cmap='tab10', s=15, alpha=0.6
    )
    cb = fig.colorbar(sc, ax=ax, ticks=range(n_clusters))
    cb.set_label('mel cluster')
    ax.set_title('mel clusters (PCA 3D)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    plt.tight_layout()
    plt.savefig('plots/mel_pca3d.png')
    plt.close()

    # 3) PCA 3D scatter for mfcc
    pca_mfcc = PCA(n_components=3, svd_solver='randomized')
    mf3      = pca_mfcc.fit_transform(X_mfcc)
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        mf3[:,0], mf3[:,1], mf3[:,2],
        c=labels_mfcc, cmap='tab10', s=15, alpha=0.6
    )
    cb = fig.colorbar(sc, ax=ax, ticks=range(n_clusters))
    cb.set_label('mfcc cluster')
    ax.set_title('mfcc clusters (PCA 3D)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    plt.tight_layout()
    plt.savefig('plots/mfcc_pca3d.png')
    plt.close()

    # 4) copy files into cluster folders
    for path, lbl in zip(paths_mel, labels_mel):
        dest = f'clusters/cluster_{lbl}/mel/{os.path.basename(path)}'
        shutil.copy(path, dest)
    for path, lbl in zip(paths_mfcc, labels_mfcc):
        dest = f'clusters/cluster_{lbl}/mfcc/{os.path.basename(path)}'
        shutil.copy(path, dest)

    print("\nAll mel and mfcc files copied into clusters/cluster{k}/[mel|mfcc].")
    print("Plots are under plots/")

if __name__ == '__main__':
    main()
