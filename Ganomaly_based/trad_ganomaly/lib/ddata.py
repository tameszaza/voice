from __future__ import annotations

import os
from glob import glob
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class AggregatedLogMelDataset(Dataset):
    """Handles **either**

    1. A single ``.npy`` file shaped ``(N, C, isize, isize)``, **or**
    2. A *directory* that contains *multiple* such ``.npy`` files
       (e.g. your ``wavefake128_2024split/test/fake`` folder).

    In the directory case, every file is loaded, validated, then concatenated
    along the first (sample) dimension so the final tensor is still
    ``(N_total, C, isize, isize)``.
    """

    def __init__(self, path: str, *, isize: int = 128):
        self.isize = isize
        self.data = self._load(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load(self, path: str) -> torch.Tensor:
        if os.path.isdir(path):
            npy_files: List[str] = sorted(glob(os.path.join(path, "*.npy")))
            if not npy_files:
                raise FileNotFoundError(f"No .npy files found in directory {path}")
            parts = [self._load_single(f) for f in npy_files]
            arr = np.concatenate(parts, axis=0)
        else:
            arr = self._load_single(path)
        return torch.from_numpy(arr.astype("float32"))

    def _load_single(self, file_path: str) -> np.ndarray:
        arr = np.load(file_path)
        expected_shape = (self.isize, self.isize)
        if arr.ndim != 4 or arr.shape[2:] != expected_shape:
            raise ValueError(
                f"{file_path}: shape {arr.shape} – expected (N, C, {self.isize}, {self.isize})"
            )
        return arr

    # ------------------------------------------------------------------
    # Torch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return self.data.shape[0]

    def __getitem__(self, idx):  # type: ignore[override]
        return (self.data[idx],)


class LogMelDataset(Dataset):
    """Loads individual spectrogram files shaped ``(1, isize, isize)`` from a
    directory.  Suitable when *each* sample lives in its own ``.npy``.
    """

    def __init__(self, root_dir: str, *, isize: int = 128):
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ]
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {root_dir}")
        self.isize = isize

    # ------------------------------------------------------------------
    # Torch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.files)

    def __getitem__(self, idx):  # type: ignore[override]
        spec = np.load(self.files[idx]).astype("float32")
        expected_shape = (1, self.isize, self.isize)
        if spec.shape != expected_shape:
            raise ValueError(
                f"{self.files[idx]}: shape {spec.shape} – expected {expected_shape}"
            )
        return (torch.from_numpy(spec),)