import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiFeatureDirectoryDataset(Dataset):
    """
    Directory layout:
      data_root/
        class_0/
          mel/   --> each .npy inside has shape (N, H, W)
          mfcc/  --> each .npy inside has shape (N, H, W)
        class_1/
          ...
    Returns (x, label) where x has shape (channels, H, W).
    """
    def __init__(self, data_root, use_mel=True, use_mfcc=True):
        if not (use_mel or use_mfcc):
            raise ValueError("At least one of use_mel or use_mfcc must be True")

        classes = sorted(
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        )
        if not classes:
            raise ValueError(f"No class subfolders found in {data_root}")

        self.samples = []
        for label, cls in enumerate(classes):
            base = os.path.join(data_root, cls)
            mel_dir  = os.path.join(base, 'mel')  if use_mel  else None
            mfcc_dir = os.path.join(base, 'mfcc') if use_mfcc else None

            if use_mel and not os.path.isdir(mel_dir):
                raise FileNotFoundError(f"Missing mel/ under {base}")
            if use_mfcc and not os.path.isdir(mfcc_dir):
                raise FileNotFoundError(f"Missing mfcc/ under {base}")

            listing_dir = mel_dir if use_mel else mfcc_dir
            for fn in sorted(os.listdir(listing_dir)):
                if not fn.endswith('.npy'):
                    continue

                mel_path  = os.path.join(mel_dir, fn)  if use_mel  else None
                mfcc_path = os.path.join(mfcc_dir, fn) if use_mfcc else None
                if use_mel and use_mfcc and not os.path.exists(mfcc_path):
                    raise FileNotFoundError(f"{mfcc_path} missing")

                # load only to read number of samples N
                arr0 = np.load(mel_path if use_mel else mfcc_path,
                               mmap_mode='r')
                N = arr0.shape[0]

                # each slice becomes one sample
                for i in range(N):
                    self.samples.append((mel_path, mfcc_path, label, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel_path, mfcc_path, label, sample_idx = self.samples[idx]
        feats = []

        if mel_path:
            mel = np.load(mel_path, mmap_mode='r')[sample_idx]
            # Min-max normalization
            mel_min = mel.min()
            mel_max = mel.max()
            if mel_max > mel_min:
                mel = (mel - mel_min) / (mel_max - mel_min)
            else:
                mel = mel - mel_min
            feats.append(mel)

        if mfcc_path:
            mfcc = np.load(mfcc_path, mmap_mode='r')[sample_idx]
            # Min-max normalization
            mfcc_min = mfcc.min()
            mfcc_max = mfcc.max()
            if mfcc_max > mfcc_min:
                mfcc = (mfcc - mfcc_min) / (mfcc_max - mfcc_min)
            else:
                mfcc = mfcc - mfcc_min
            feats.append(mfcc)

        x = np.stack(feats, axis=0)   # (channels, H, W)
        return torch.from_numpy(x).float(), label


class SingleNpyDataset(Dataset):
    """
    A single .npy file of shape (N, ...) or (N, C, H, W).
    Returns x or (x, label) if label is given.
    """
    def __init__(self, npy_file, label=None, max_samples=None):
        self.data = np.load(npy_file, mmap_mode='r')
        if max_samples:
            self.data = self.data[:max_samples]
        self.label = label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        # Min-max normalization
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = arr - arr_min
        x = torch.from_numpy(arr).float()
        if self.label is not None:
            return x, self.label
        return x


class RealFakeNpyDataset(Dataset):
    """
    For two sources: real_source (file or dir) and fake_source (file or dir).
    Label 0 = real, 1 = fake.
    Pass --max_samples_per_class to limit each side.
    """
    def __init__(self, real_source, fake_source, max_samples_per_class=None):
        self.samples = []

        def add(src, label):
            if os.path.isdir(src):
                files = sorted(f for f in os.listdir(src) if f.endswith('.npy'))
                for fn in files:
                    path = os.path.join(src, fn)
                    arr = np.load(path, mmap_mode='r')
                    N = arr.shape[0]
                    if max_samples_per_class:
                        N = min(N, max_samples_per_class)
                    for i in range(N):
                        self.samples.append((path, i, label))
            else:
                arr = np.load(src, mmap_mode='r')
                N = arr.shape[0]
                if max_samples_per_class:
                    N = min(N, max_samples_per_class)
                for i in range(N):
                    self.samples.append((src, i, label))

        add(real_source, 0)
        add(fake_source, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sample_idx, label = self.samples[idx]
        arr = np.load(path, mmap_mode='r')[sample_idx]
        arr = arr.copy()   # now it’s a normal, writable ndarray
        # Min-max normalization
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = arr - arr_min
        return torch.from_numpy(arr).float(), label



class MultiNpyFilesDataset(Dataset):
    """
    A folder of many .npy files → each file is its own class.
    Pass --max_samples_per_file to limit each file.
    """
    def __init__(self, dir_path, max_samples_per_file=None):
        self.samples = []
        files = sorted(f for f in os.listdir(dir_path) if f.endswith('.npy'))
        if not files:
            raise ValueError(f"No .npy files found in {dir_path}")

        for label, fn in enumerate(files):
            path = os.path.join(dir_path, fn)
            arr = np.load(path, mmap_mode='r')
            N = arr.shape[0]
            if max_samples_per_file:
                N = min(N, max_samples_per_file)
            for i in range(N):
                self.samples.append((path, i, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sample_idx, label = self.samples[idx]
        arr = np.load(path, mmap_mode='r')[sample_idx]
        arr = arr.copy()   # now it’s a normal, writable ndarray

        # Min-max normalization to [0, 1]
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = arr - arr_min  # all values are the same

        return torch.from_numpy(arr).float(), label
