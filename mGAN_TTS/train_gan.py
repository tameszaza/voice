# Filename: train_gan.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import matplotlib.pyplot as plt
import librosa.display
import glob
# ------------------ Dataset Class ------------------


# ------------------ Dataset Class ------------------


class FlacDataset(Dataset):
    def __init__(self, data_dir, target_sr=16000, fragment_duration=4.0):
        """
        Dataset to handle audio files and return fixed-length audio waveforms.
        Args:
            data_dir (str): Directory containing audio files (.flac or .wav).
            target_sr (int): Target sampling rate for audio.
            fragment_duration (float): Duration (in seconds) for each audio fragment.
        """
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.fragment_duration = fragment_duration
        self.fragment_samples = int(self.fragment_duration * self.target_sr)
        self.audio_fragments = []
        self._load_files()

    def _load_files(self):
        # Gather all audio files (both .flac and .wav)
        audio_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.flac') or file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))

        if not audio_files:
            raise ValueError(f"No audio files found in directory: {self.data_dir}")

        print(f"Found {len(audio_files)} audio files.")

        fragment_samples = self.fragment_samples
        current_audio = []  # Buffer to hold concatenated audio

        for file_path in audio_files:
            # Load each audio file
            y, sr = librosa.load(file_path, sr=self.target_sr)
            total_samples = len(y)
            start = 0

            while start < total_samples:
                remaining_samples = total_samples - start
                needed_samples = fragment_samples - len(current_audio)
                take_samples = min(remaining_samples, needed_samples)
                current_audio.extend(y[start:start + take_samples])
                start += take_samples

                if len(current_audio) == fragment_samples:
                    # Save the current fragment
                    self.audio_fragments.append(np.array(current_audio))
                    current_audio = []

        # Handle leftover buffer
        if len(current_audio) > 0:
            # Pad to make it a complete fragment if necessary
            if len(current_audio) < fragment_samples:
                current_audio.extend([0] * (fragment_samples - len(current_audio)))
            self.audio_fragments.append(np.array(current_audio))

        print(f"Generated {len(self.audio_fragments)} audio fragments.")

    def __len__(self):
        return len(self.audio_fragments)

    def __getitem__(self, idx):
        # Retrieve the audio fragment
        audio = self.audio_fragments[idx]

        # Normalize audio to [-1, 1]
        audio = audio / np.max(np.abs(audio) + 1e-9)  # Added epsilon to prevent division by zero

        # Convert to PyTorch tensor and add channel dimension
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # Shape [1, fragment_samples]

        return audio_tensor


# ------------------ Rest of the Code Remains the Same ------------------

# The rest of your code (model definitions, training functions, etc.) remains unchanged.
# Ensure you import any additional necessary libraries if they are used in your code.

# ------------------ Main Execution ------------------

# ------------------ Model Architecture ------------------

class Conv1d(nn.Conv1d):
    """Custom Conv1d with weight initialization."""

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)

class Generator(nn.Module):
    def __init__(self, z_channels=128, out_channels=1, seq_length=64000):
        super(Generator, self).__init__()

        self.z_channels = z_channels
        self.seq_length = seq_length

        # Calculate initial sequence length after the fully connected layer
        initial_seq_length = seq_length // (2 ** 4)  # Assuming 4 GBlocks with stride=2
        self.fc = nn.Linear(z_channels, 768 * initial_seq_length)
        self.gblocks = nn.Sequential(
            GBlock(768, 768),
            GBlock(768, 384),
            GBlock(384, 192),
            GBlock(192, 96)
        )
        self.postprocess = nn.Sequential(
            Conv1d(96, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, 768, self.seq_length // (2 ** 4))
        x = self.gblocks(x)
        x = self.postprocess(x)
        return x


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, inputs):
        outputs = self.model(inputs)
        outputs = outputs.view(-1, 1)
        return outputs

class Encoder(nn.Module):
    def __init__(self, in_channels=1, z_dim=128):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, z_dim)

    def forward(self, inputs):
        outputs = self.model(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.fc(outputs)
        return outputs

# ------------------ Training Functions ------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_noise(batch_size, z_dim, device):
    return torch.randn(batch_size, z_dim, device=device)

def orthogonal_loss(feature1, feature2):
    inner_product = torch.sum(feature1 * feature2, dim=1)
    norm1 = torch.norm(feature1, dim=1)
    norm2 = torch.norm(feature2, dim=1)
    cosine_similarity = inner_product / (norm1 * norm2 + 1e-8)
    return torch.mean(cosine_similarity**2)

def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_pred * y_true)

def compute_gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    epsilon = epsilon.expand_as(real_data)

    interpolates = epsilon * real_data + (1 - epsilon) * fake_data
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def visualize_generated_mel_spectrograms(generators, z_dim, num_samples, device, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, gen in enumerate(generators):
        gen.eval()
        with torch.no_grad():
            noise = generate_noise(num_samples, z_dim, device)
            fake = gen(z=noise).cpu().numpy()
            for i in range(num_samples):
                mel_spec = fake[i][0]  # Assuming shape [batch_size, channels, n_mels, time_steps]
                mel_spec_db = mel_spec * 80 - 80  # Convert back to dB scale

                plt.figure(figsize=(10, 4))
                librosa.display.specshow(
                    mel_spec_db, sr=16000, hop_length=256,
                    x_axis='time', y_axis='mel'
                )
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Epoch {epoch} Generator {idx+1} Sample {i+1}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'epoch{epoch}_gen{idx+1}_sample{i+1}.png'))
                plt.close()
        gen.train()
def visualize_generated_audio(generators, num_samples, device, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, gen in enumerate(generators):
        gen.eval()
        with torch.no_grad():
            noise = generate_noise(num_samples, gen.z_channels, device)
            fake_audio = gen(z=noise).cpu().numpy()
            for i in range(num_samples):
                audio = fake_audio[i][0]  # Shape [length]

                # Generate Mel spectrogram for visualization
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                plt.figure(figsize=(10, 4))
                librosa.display.specshow(
                    mel_spec_db, sr=16000, hop_length=512,
                    x_axis='time', y_axis='mel'
                )
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Epoch {epoch} Generator {idx+1} Sample {i+1}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'epoch{epoch}_gen{idx+1}_sample{i+1}.png'))
                plt.close()
        gen.train()


def pretrain_single_generator(num_epochs, z_dim, lr_gen, lr_disc, batch_size, seed, dataset, output_dir):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    generator = Generator(z_channels=z_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_gen = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.9))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.9))

    # Use the Mel spectrogram dataset
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    lambda_gp = 10
    num_critic = 5

    # Training loop
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        loss_disc_epoch = 0
        loss_gen_epoch = 0

        for batch_idx, real in enumerate(train_loader):
            real = real.to(device)
            batch_size = real.size(0)
            real_label = -torch.ones(batch_size, 1, device=device)
            fake_label = torch.ones(batch_size, 1, device=device)

            # Generate noise and fake images
            noise = generate_noise(batch_size, z_dim, device)
            fake = generator(z=noise)

            # Optional: Add noise to inputs
            real = real + 0.001 * torch.randn_like(real)
            fake = fake + 0.001 * torch.randn_like(fake)

            # Train Discriminator multiple times
            for _ in range(num_critic):
                optimizer_disc.zero_grad()
                disc_real = discriminator(real)
                disc_fake = discriminator(fake.detach())

                loss_disc = wasserstein_loss(disc_real, real_label) + wasserstein_loss(disc_fake, fake_label)
                gradient_penalty = compute_gradient_penalty(discriminator, real, fake.detach(), device)
                loss_disc += lambda_gp * gradient_penalty

                loss_disc.backward()
                optimizer_disc.step()

            loss_disc_epoch += loss_disc.item()

            # Train Generator
            optimizer_gen.zero_grad()
            fake = generator(z=noise)
            disc_fake = discriminator(fake)
            loss_gen = wasserstein_loss(disc_fake, real_label)
            loss_gen.backward()
            optimizer_gen.step()

            loss_gen_epoch += loss_gen.item()

        avg_loss_disc = loss_disc_epoch / len(train_loader)
        avg_loss_gen = loss_gen_epoch / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {avg_loss_disc:.4f}, Loss G: {avg_loss_gen:.4f}")

        # Visualize generated images
        visualize_generated_mel_spectrograms(
            [generator], z_dim, num_samples=5, device=device,
            epoch=epoch+1, output_dir=output_dir
        )

    torch.save(generator.state_dict(), os.path.join(output_dir, "pretrained_generator.pth"))
    print(f"Pretrained generator model saved to {os.path.join(output_dir, 'pretrained_generator.pth')}")

    return generator

def initialize_multiple_generators(pretrained_generator, num_generators, z_dim):
    device = next(pretrained_generator.parameters()).device
    generators = []
    for _ in range(num_generators):
        new_generator = Generator(z_channels=z_dim).to(device)
        new_generator.load_state_dict(pretrained_generator.state_dict())
        generators.append(new_generator)
    return generators

def train_gan_with_pretrained_generators(
    pretrained_generator, num_epochs, z_dim, lr_gen, lr_disc, batch_size,
    num_generators, seed, dataset, output_dir, lambda_ortho=0.1
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize multiple generators from the pretrained generator
    generators = initialize_multiple_generators(pretrained_generator, num_generators, z_dim)

    # Initialize Discriminator and Encoder
    discriminator = Discriminator().to(device)
    encoder = Encoder(z_dim=z_dim).to(device)

    optimizer_gens = [optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.5, 0.9)) for gen in generators]
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.9))
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=lr_disc, betas=(0.5, 0.9))

    # Use the Mel spectrogram dataset
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    lambda_gp = 10  # Gradient penalty coefficient
    num_critic = 5  # Number of discriminator updates per generator update

    # Training loop
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        loss_disc_epoch = 0
        loss_gens_epoch = [0] * num_generators

        for batch_idx, real in enumerate(train_loader):
            real = real.to(device)
            batch_size = real.size(0)
            real_label = -torch.ones(batch_size, 1, device=device)
            fake_label = torch.ones(batch_size, 1, device=device)

            # Optional: Add noise to real inputs
            real = real + 0.001 * torch.randn_like(real)

            # Train Discriminator multiple times
            for _ in range(num_critic):
                optimizer_disc.zero_grad()
                disc_real = discriminator(real)

                noises = [generate_noise(batch_size, z_dim, device) for _ in range(num_generators)]
                fakes = [gen(z=noises[idx]).detach() for idx, gen in enumerate(generators)]

                # Optional: Add noise to fake inputs
                for idx in range(num_generators):
                    fakes[idx] = fakes[idx] + 0.001 * torch.randn_like(fakes[idx])

                disc_fakes = [discriminator(fake) for fake in fakes]

                loss_disc = wasserstein_loss(disc_real, real_label)
                for disc_fake in disc_fakes:
                    loss_disc += wasserstein_loss(disc_fake, fake_label)

                # Compute gradient penalties
                gradient_penalty = 0
                for fake in fakes:
                    gradient_penalty += compute_gradient_penalty(discriminator, real, fake, device)

                loss_disc += lambda_gp * gradient_penalty / num_generators
                loss_disc.backward()
                optimizer_disc.step()

            loss_disc_epoch += loss_disc.item()

            # Train Generators with Orthogonal Loss
            for idx, gen in enumerate(generators):
                optimizer_gens[idx].zero_grad()
                optimizer_encoder.zero_grad()

                noise = generate_noise(batch_size, z_dim, device)
                fake = gen(z=noise)
                disc_fake = discriminator(fake)

                # Wasserstein loss for generator
                loss_gen = wasserstein_loss(disc_fake, real_label)

                # Compute orthogonal loss with other generators
                gen_feature = encoder(fake)
                ortho_loss_total = 0
                for other_idx, other_gen in enumerate(generators):
                    if idx != other_idx:
                        other_noise = generate_noise(batch_size, z_dim, device)
                        other_fake = other_gen(z=other_noise)
                        other_feature = encoder(other_fake)
                        ortho_loss = orthogonal_loss(gen_feature, other_feature)
                        ortho_loss_total += ortho_loss

                # Average orthogonal loss
                if num_generators > 1:
                    ortho_loss_total = ortho_loss_total / (num_generators - 1)
                else:
                    ortho_loss_total = 0

                # Total generator loss with scaled orthogonal loss
                total_loss_gen = loss_gen + lambda_ortho * ortho_loss_total

                total_loss_gen.backward()
                optimizer_gens[idx].step()
                optimizer_encoder.step()

                loss_gens_epoch[idx] += total_loss_gen.item()

        # Calculate average losses
        avg_loss_disc = loss_disc_epoch / len(train_loader)
        avg_loss_gens = [loss / len(train_loader) for loss in loss_gens_epoch]

        # Print losses
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {avg_loss_disc:.4f}")
        for idx in range(num_generators):
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss G{idx+1}: {avg_loss_gens[idx]:.4f}")
        print('-' * 50)

        # Visualize and save generated Mel spectrograms
        visualize_generated_mel_spectrograms(
            generators, z_dim, num_samples=5, device=device,
            epoch=epoch+1, output_dir=output_dir
        )

    # Return the trained generators
    return generators

# ------------------ Main Execution ------------------

if __name__ == '__main__':
    # Define your data directory
    data_dir = 'data/LibriSpeech/LibriSpeech/dev-clean'  # Replace with your actual data path

    # Initialize your dataset
    fragment_duration = 4.0  # Duration in seconds
    target_sr = 16000
    dataset = FlacDataset(
        data_dir=data_dir,
        target_sr=target_sr,
        fragment_duration=fragment_duration
    )

    # Calculate sequence length based on fragment duration and sampling rate
    seq_length = int(fragment_duration * target_sr)

    # Pretrain the generator
    pretrained_generator = pretrain_single_generator(
        num_epochs=20,        # Adjust the number of epochs as needed
        z_dim=128,
        lr_gen=0.0001,
        lr_disc=0.0004,
        batch_size=16,        # Adjust batch size based on your system's capability
        seed=1234,
        dataset=dataset,
        output_dir='pretrain_outputsCNN',
        seq_length=seq_length  # Pass the sequence length
    )

    # Train multiple generators
    trained_generators = train_gan_with_pretrained_generators(
        pretrained_generator=pretrained_generator,
        num_epochs=80,        # Adjust the number of epochs as needed
        z_dim=128,
        lr_gen=0.0001,
        lr_disc=0.0004,
        batch_size=8,         # Adjust batch size based on your system's capability
        num_generators=2,
        seed=1234,
        dataset=dataset,
        output_dir='multi_gen_outputs_CNN',
        lambda_ortho=0.05,
        seq_length=seq_length  # Pass the sequence length
    )
