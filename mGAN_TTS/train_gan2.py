# Filename: train_gan_with_preprocessing.py

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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import pickle
import glob

# ------------------ Data Preprocessing ------------------

def find_files(path, pattern="*.wav"):
    filenames = []
    for filename in glob.iglob(f'{path}/**/{pattern}', recursive=True):
        filenames.append(filename)
    return filenames

def convert_audio(wav_file, target_sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    y, sr = librosa.load(wav_file, sr=target_sr)
    y = y / np.max(np.abs(y) + 1e-9)  # Normalize audio
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # Normalize Mel spectrogram to [0, 1]
    mel_spectrogram = (mel_spectrogram + 80) / 80
    return mel_spectrogram, y

def data_prepare(audio_path, mel_path, wav_file):
    mel, audio = convert_audio(wav_file)
    np.save(audio_path, audio, allow_pickle=False)
    np.save(mel_path, mel, allow_pickle=False)
    return audio_path, mel_path, mel.shape[1]  # Use mel.shape[1] for the number of frames

def process_data(output_dir, wav_files, train_dir, test_dir, num_workers, train_rate=0.95):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    results = []
    names = []

    random.shuffle(wav_files)
    train_num = int(len(wav_files) * train_rate)

    # Process training files
    for wav_file in wav_files[:train_num]:
        fid = os.path.basename(wav_file).replace('.wav', '.npy')
        names.append(fid)
        results.append(executor.submit(partial(
            data_prepare,
            os.path.join(train_dir, "audio", fid),
            os.path.join(train_dir, "mel", fid),
            wav_file
        )))

    with open(os.path.join(output_dir, "train", 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)

    names = []
    # Process testing files
    for wav_file in wav_files[train_num:]:
        fid = os.path.basename(wav_file).replace('.wav', '.npy')
        names.append(fid)
        results.append(executor.submit(partial(
            data_prepare,
            os.path.join(test_dir, "audio", fid),
            os.path.join(test_dir, "mel", fid),
            wav_file
        )))

    with open(os.path.join(output_dir, "test", 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)

    return [result.result() for result in tqdm(results)]

def write_metadata(metadata, out_dir):
    frames = sum([m[2] for m in metadata])
    hours = frames * 256 / 16000 / 3600  # Assuming hop_length=256 and sr=16000
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))

def preprocess_data(wav_dir, output_dir, num_workers=4):
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "mel"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "mel"), exist_ok=True)

    wav_files = find_files(wav_dir, pattern="*.wav")
    metadata = process_data(output_dir, wav_files, train_dir, test_dir, num_workers)
    write_metadata(metadata, output_dir)

# ------------------ Dataset Class ------------------

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir, data_type='mel', seq_len=128):
        """
        Args:
            data_dir (str): Path to the directory containing preprocessed data.
            data_type (str): 'mel' or 'audio' to specify which data to load.
            seq_len (int): Sequence length for data truncation or padding.
        """
        self.data_paths = glob.glob(os.path.join(data_dir, data_type, '*.npy'))
        self.data_paths.sort()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx])
        if len(data.shape) == 1:
            # Audio waveform
            if len(data) < self.seq_len:
                padding = np.zeros(self.seq_len - len(data))
                data = np.concatenate((data, padding))
            else:
                data = data[:self.seq_len]
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        else:
            # Mel spectrogram
            if data.shape[1] < self.seq_len:
                pad_width = self.seq_len - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            else:
                data = data[:, :self.seq_len]
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return data_tensor

# ------------------ Model Architecture ------------------

class Conv1d(nn.Conv1d):
    """Custom Conv1d with weight initialization."""

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)

class Generator(nn.Module):
    def __init__(self, z_channels=128, out_channels=1, seq_length=128):
        super(Generator, self).__init__()

        self.z_channels = z_channels
        self.seq_length = seq_length

        self.fc = nn.Linear(z_channels, 256 * (seq_length // 8))
        self.gblocks = nn.Sequential(
            GBlock(256, 128),
            GBlock(128, 64),
            GBlock(64, 32)
        )
        self.postprocess = nn.Sequential(
            Conv1d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        batch_size = z.size(0)
        x = self.fc(z)
        x = x.view(batch_size, 256, self.seq_length // 8)
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
            nn.Conv1d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, inputs):
        outputs = self.model(inputs)
        outputs = outputs.view(-1, 1)
        return outputs

class Encoder(nn.Module):
    def __init__(self, in_channels=1, z_dim=128):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, z_dim)

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

def visualize_generated_audio(generators, z_dim, num_samples, device, epoch, output_dir, target_sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    for idx, gen in enumerate(generators):
        gen.eval()
        with torch.no_grad():
            noise = generate_noise(num_samples, z_dim, device)
            fake = gen(z=noise).cpu().numpy()
            for i in range(num_samples):
                audio = fake[i][0]  # Assuming shape [batch_size, 1, seq_length]
                # Save the waveform as an audio file
                audio_path = os.path.join(output_dir, f'epoch{epoch}_gen{idx+1}_sample{i+1}.wav')
                librosa.output.write_wav(audio_path, audio, sr=target_sr)

                # Plot the waveform
                plt.figure(figsize=(10, 4))
                plt.plot(audio)
                plt.title(f'Epoch {epoch} Generator {idx+1} Sample {i+1}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'epoch{epoch}_gen{idx+1}_sample{i+1}.png'))
                plt.close()
        gen.train()

def pretrain_single_generator(num_epochs, z_dim, lr_gen, lr_disc, batch_size, seed, dataset, output_dir):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    generator = Generator(z_channels=z_dim, seq_length=128).to(device)
    discriminator = Discriminator().to(device)

    optimizer_gen = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.9))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.9))

    # Use the preprocessed dataset
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

            # Generate noise and fake data
            noise = generate_noise(batch_size, z_dim, device)
            fake = generator(z=noise)

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

        # Visualize generated audio
        visualize_generated_audio(
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
        new_generator = Generator(z_channels=z_dim, seq_length=128).to(device)
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

    # Use the preprocessed dataset
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

            # Train Discriminator multiple times
            for _ in range(num_critic):
                optimizer_disc.zero_grad()
                disc_real = discriminator(real)

                noises = [generate_noise(batch_size, z_dim, device) for _ in range(num_generators)]
                fakes = [gen(z=noises[idx]).detach() for idx, gen in enumerate(generators)]

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

        # Visualize and save generated audio
        visualize_generated_audio(
            generators, z_dim, num_samples=5, device=device,
            epoch=epoch+1, output_dir=output_dir
        )

    # Return the trained generators
    return generators

# ------------------ Main Execution ------------------

if __name__ == '__main__':
    # Step 1: Preprocess Data
    wav_dir = r'path/to/your/audio/files'  # Replace with your actual data path
    output_dir = r'path/to/output/directory'  # Directory to save preprocessed data

    # Preprocess the data
    preprocess_data(wav_dir, output_dir, num_workers=4)

    # Step 2: Initialize Dataset
    # Use 'audio' if training on waveforms, 'mel' if training on spectrograms
    data_type = 'audio'  # Choose 'audio' or 'mel' based on your preference
    seq_len = 16384  # Adjust sequence length based on your data (e.g., 1 sec at 16kHz)

    dataset = PreprocessedDataset(
        data_dir=os.path.join(output_dir, 'train'),
        data_type=data_type,
        seq_len=seq_len
    )

    # Step 3: Pretrain the generator
    pretrained_generator = pretrain_single_generator(
        num_epochs=10,        # Adjust the number of epochs as needed
        z_dim=128,
        lr_gen=0.0001,
        lr_disc=0.0004,
        batch_size=16,        # Adjust batch size based on your system's capability
        seed=1234,
        dataset=dataset,
        output_dir='pretrain_outputs'
    )

    # Step 4: Train multiple generators
    trained_generators = train_gan_with_pretrained_generators(
        pretrained_generator=pretrained_generator,
        num_epochs=20,        # Adjust the number of epochs as needed
        z_dim=128,
        lr_gen=0.0001,
        lr_disc=0.0004,
        batch_size=8,         # Adjust batch size based on your system's capability
        num_generators=2,
        seed=1234,
        dataset=dataset,
        output_dir='multi_gen_outputs',
        lambda_ortho=0.05
    )
