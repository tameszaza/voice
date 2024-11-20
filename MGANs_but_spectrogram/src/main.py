import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from model import MGAN  # Import the MGAN class
from dataset_load import librispeech_to_mel  # Your dataset preprocessing function
import glob


def main():
    # Hyperparameters
    num_z = 100
    beta = 0.5
    num_gens = 10
    batch_size = 16
    z_prior = "gaussian"
    learning_rate = 0.0002
    num_epochs = 50

    # Spectrogram dimensions
    mel_bins = 64
    num_frames = 128
    num_channels = 1
    img_size = (mel_bins, num_frames, num_channels)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset preprocessing
    data_dir = r'data\LibriSpeech\LibriSpeech\dev-clean'
    print("Preprocessing dataset into Mel spectrograms...")
    mel_dataset = librispeech_to_mel(data_dir, target_sr=16000, n_mels=mel_bins, target_length=num_frames)
    print(f"Dataset size: {len(mel_dataset)}")

    # DataLoader
    dataloader = DataLoader(mel_dataset, batch_size=batch_size, shuffle=True)
    if len(mel_dataset) == 0:
        raise ValueError("The dataset is empty. Please check your data directory and preprocessing function.")

    # Initialize MGAN
    mgan_model = MGAN(
        num_z=num_z,
        beta=beta,
        num_gens=num_gens,
        batch_size=batch_size,
        z_prior=z_prior,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        img_size=img_size,
        num_gen_feature_maps=64,
        num_dis_feature_maps=64,
        sample_dir="samples",
        device=device
    )
    if os.path.exists("mgan_model.pth"):
        mgan_model.load_state_dict(torch.load("mgan_model.pth"))
        print("Model loaded from mgan_model.pth")

    # Train the model
    print("Starting training...")
    mgan_model.fit(dataloader)
    print("Training completed.")
    torch.save(mgan_model.state_dict(), "mgan_model.pth")
    print("Model saved to mgan_model.pth")

if __name__ == "__main__":
    main()
