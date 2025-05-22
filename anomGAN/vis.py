import os
import numpy as np
import matplotlib.pyplot as plt
import random

def visualize_and_save_preprocessed(mel_dir, mfcc_dir, save_dir, num_examples=5):
    """
    Randomly select and visualize preprocessed Mel-spectrogram and MFCC .npy files side by side.
    Save the visualizations to a given directory.
    
    Parameters:
    - mel_dir: Directory containing Mel-spectrogram .npy files
    - mfcc_dir: Directory containing MFCC .npy files
    - save_dir: Directory where plots will be saved
    - num_examples: Number of examples to visualize and save
    """
    os.makedirs(save_dir, exist_ok=True)
    mel_files = [f for f in os.listdir(mel_dir) if f.endswith('.npy')]
    selected_files = random.sample(mel_files, min(num_examples, len(mel_files)))
    
    for idx, filename in enumerate(selected_files):
        mel_path = os.path.join(mel_dir, filename)
        mfcc_path = os.path.join(mfcc_dir, filename)

        if not os.path.exists(mfcc_path):
            print(f"Skipping {filename}: no matching MFCC file found.")
            continue

        mel = np.load(mel_path)
        mfcc = np.load(mfcc_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.imshow(mel, aspect='auto', origin='lower')
        ax1.set_title(f"Mel-Spectrogram: {filename}")
        ax1.set_xlabel("Time Frames")
        ax1.set_ylabel("Mel Bins")

        ax2.imshow(mfcc, aspect='auto', origin='lower')
        ax2.set_title(f"MFCC: {filename}")
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("MFCC Coeffs")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"visual_{idx}_{filename.replace('.npy', '')}.png")
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize and save preprocessed Mel and MFCC features")
    parser.add_argument("mel_dir", help="Directory with Mel-spectrogram .npy files")
    parser.add_argument("mfcc_dir", help="Directory with MFCC .npy files")
    parser.add_argument("save_dir", help="Directory to save plots")
    parser.add_argument("--num", type=int, default=5, help="Number of random examples to visualize")
    args = parser.parse_args()
    visualize_and_save_preprocessed(args.mel_dir, args.mfcc_dir, args.save_dir, args.num)
