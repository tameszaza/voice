#!/usr/bin/env python3
"""
Cluster two separate sets of .npy samples (mel and mfcc) into balanced groups,
report clustering validity scores, then save each cluster’s samples in
stacked.npy format under output_root/cluster{0..K}/[mel|mfcc].
"""

import os
import glob
import shutil

import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 for 3d scatter


def balanced_kmeans(X: np.ndarray, n_clusters: int, random_state=42) -> np.ndarray:
    N, _ = X.shape
    km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    centers = km.cluster_centers_
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    base = N // n_clusters
    rem  = N % n_clusters
    desired = np.array([base + 1] * rem + [base] * (n_clusters - rem))

    labels = -1 * np.ones(N, dtype=int)
    counts = np.zeros(n_clusters, dtype=int)

    pairs = [(dists[i, j], i, j) for i in range(N) for j in range(n_clusters)]
    pairs.sort(key=lambda x: x[0])

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
    def __init__(self, dir_path):
        stacked_path = os.path.join(dir_path, 'stacked.npy')
        self.is_stacked = False
        self.dir_path = dir_path

        if os.path.isfile(stacked_path):
            arr = np.load(stacked_path, mmap_mode='r')
            if arr.ndim != 3:
                raise ValueError(f"{stacked_path} must be 3D, got {arr.shape}")
            self.is_stacked = True
            self.stacked_arr = arr
            self.n = arr.shape[0]
        else:
            files = sorted(glob.glob(os.path.join(dir_path, '*.npy')))
            if not files:
                raise ValueError(f"No .npy files in {dir_path}")
            self.files = files
            self.n = len(files)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.is_stacked:
            arr2d = self.stacked_arr[idx]
            base = os.path.splitext(os.path.basename(self.dir_path))[0]
            name = f"{base}_{idx}.npy"
            return arr2d, name
        else:
            path = self.files[idx]
            arr2d = np.load(path)
            if arr2d.ndim != 2:
                raise ValueError(f"{path} must be 2D, got {arr2d.shape}")
            return arr2d, path


def collate_fn(batch):
    arrays, paths = zip(*batch)
    big = np.stack(arrays, axis=0)
    return big, list(paths)


def cluster_folder(input_dir, n_clusters):
    ds     = NpyFileDataset(input_dir)
    loader = DataLoader(
        ds,
        batch_size = len(ds),
        shuffle    = False,
        collate_fn = collate_fn
    )
    X_all, paths = next(iter(loader))           # X_all: shape (N, H, W) or (N, features)
    N = X_all.shape[0]
    X_flat = X_all.reshape(N, -1) if X_all.ndim > 2 else X_all
    labels = balanced_kmeans(X_flat, n_clusters)
    return labels, paths, X_flat, X_all


def main():
    # ------------ USER CONFIGURATION ------------
    mel_dir     = '../data/real/data_128/all/mel'
    mfcc_dir    = '../data/real/data_128/all/mfcc'
    n_clusters  = 7
    output_root = '../data/real/clusters_128'
    # --------------------------------------------

    # cluster both folders
    print("Clustering mel files...")
    labels_mel,  paths_mel,  X_mel_flat,  X_mel_all  = cluster_folder(mel_dir,  n_clusters)
    print("Clustering mfcc files...")
    labels_mfcc, paths_mfcc, X_mfcc_flat, X_mfcc_all = cluster_folder(mfcc_dir, n_clusters)

    # report sizes
    sizes_mel  = np.bincount(labels_mel,  minlength=n_clusters)
    sizes_mfcc = np.bincount(labels_mfcc, minlength=n_clusters)
    print("\nCluster sizes (mel):")
    for k, sz in enumerate(sizes_mel):
        print(f"  mel cluster {k}: {sz}")
    print("\nCluster sizes (mfcc):")
    for k, sz in enumerate(sizes_mfcc):
        print(f"  mfcc cluster {k}: {sz}")

    # compute clustering validity scores
    print("\nClustering scores (mel):")
    print(f"  silhouette score        : {silhouette_score(X_mel_flat, labels_mel):.4f}")
    print(f"  Calinski-Harabasz index : {calinski_harabasz_score(X_mel_flat, labels_mel):.4f}")
    print(f"  Davies-Bouldin score    : {davies_bouldin_score(X_mel_flat, labels_mel):.4f}")

    print("\nClustering scores (mfcc):")
    print(f"  silhouette score        : {silhouette_score(X_mfcc_flat, labels_mfcc):.4f}")
    print(f"  Calinski-Harabasz index : {calinski_harabasz_score(X_mfcc_flat, labels_mfcc):.4f}")
    print(f"  Davies-Bouldin score    : {davies_bouldin_score(X_mfcc_flat, labels_mfcc):.4f}")

    # save label arrays
    np.save('cluster_labels_mel.npy',  labels_mel)
    np.save('cluster_labels_mfcc.npy', labels_mfcc)
    print("\nSaved cluster_labels_mel.npy and cluster_labels_mfcc.npy")

    # prepare output folders
    os.makedirs('plots', exist_ok=True)
    for k in range(n_clusters):
        os.makedirs(os.path.join(output_root, f'cluster_{k}', 'mel'),  exist_ok=True)
        os.makedirs(os.path.join(output_root, f'cluster_{k}', 'mfcc'), exist_ok=True)

    # [ plotting code unchanged, omitted for brevity ]

    # replace the old copy/save loop with this stacked-save block:
    for feature, labels, X_all in [
        ('mel',  labels_mel,  X_mel_all),
        ('mfcc', labels_mfcc, X_mfcc_all)
    ]:
        for k in range(n_clusters):
            # select all samples in cluster k
            idxs = np.where(labels == k)[0]
            arrs = X_all[idxs]
            save_dir = os.path.join(output_root, f'cluster_{k}', feature)
            stacked_path = os.path.join(save_dir, 'stacked.npy')
            np.save(stacked_path, arrs)
            print(f"Saved {arrs.shape[0]} samples to {stacked_path}")

    print("\nAll clusters saved in stacked.npy format.")
    print("Plots are under plots/")

if __name__ == '__main__':
    main()
