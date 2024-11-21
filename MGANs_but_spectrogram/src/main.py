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


def save_mel_spectrograms(dataset, save_dir, num_to_visualize=5):
    """
    Visualize and save the first 'num_to_visualize' Mel spectrograms in the dataset.
    
    Args:
    - dataset: TensorDataset containing Mel spectrograms.
    - save_dir: Directory where the images will be saved.
    - num_to_visualize: Number of spectrograms to visualize and save.
    """
    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(num_to_visualize, len(dataset))):
        # Extract Mel spectrogram from dataset
        mel_spec = dataset[i][0].squeeze(0).numpy()  # Remove channel dimension
        
        # Plot the Mel spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'Mel Spectrogram {i + 1}')
        plt.xlabel('Time Frames')
        plt.ylabel('Mel Bands')
        plt.colorbar(format='%+2.0f dB')
        
        # Save the plot
        save_path = os.path.join(save_dir, f'mel_spectrogram_{i + 1}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


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
    data_dir = os.path.join("data", "LibriSpeech", "LibriSpeech", "dev-clean")
    print("Preprocessing dataset into Mel spectrograms...")
    try:
        mel_dataset = librispeech_to_mel(data_dir, target_sr=16000, n_mels=mel_bins, target_length=num_frames)
        print(f"Dataset size: {len(mel_dataset)}")
    except Exception as e:
        print(f"Error during dataset preprocessing: {e}")
        return

    # Check if dataset is empty
    if len(mel_dataset) == 0:
        raise ValueError("The dataset is empty. Please check your data directory and preprocessing function.")

    # Save visualizations of the first 5 Mel spectrograms
    visualization_dir = "mel_spectrograms"
    print("Saving Mel spectrogram visualizations...")
    save_mel_spectrograms(mel_dataset, visualization_dir, num_to_visualize=5)

    # DataLoader
    dataloader = DataLoader(mel_dataset, batch_size=batch_size, shuffle=True)

    # Initialize MGAN
    sample_dir = "samples"
    os.makedirs(sample_dir, exist_ok=True)
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
        sample_dir=sample_dir,
        device=device
    )

    # Load pre-trained model if available
    model_path = "mgan_model.pth"
    if os.path.exists(model_path):
        try:
            mgan_model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    # Train the model
    print("Starting training...")
    try:
        mgan_model.fit(dataloader)
        print("Training completed.")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Save the trained model
    try:
        torch.save(mgan_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


if __name__ == "__main__":
    main()
