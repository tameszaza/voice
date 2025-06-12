# utils/normalise.py  (tiny helper)
import numpy as np, glob

def fit_stats(logmel_dir=""):
    arr = np.stack([np.load(p) for p in glob.glob(f"{logmel_dir}/*.npy")])
    mean = arr.mean(axis=(0,2,3), keepdims=True)
    std  = arr.std (axis=(0,2,3), keepdims=True) + 1e-8
    np.savez(f"{logmel_dir}/stats.npz", mean=mean, std=std)
    print("Saved normalisation constants.")

# def apply(spec, stats_path="voice/log-mel/stats.npz"):
#     stats = np.load(stats_path)
#     return (spec - stats["mean"]) / stats["std"]

if __name__ == "__main__":
    fit_stats(logmel_dir="ResData/log-mel-data")
