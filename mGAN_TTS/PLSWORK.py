# Filename: train_gan.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import necessary modules from your model architecture code
# Assuming they are in a file named 'model_architecture.py'
# from model_architecture import Generator, Multiple_Random_Window_Discriminators

# For the purpose of this example, I'll include the adjusted classes here

# ------------------ Model Architecture ------------------

class Conv1d(nn.Conv1d):
    """Custom Conv1d with weight initialization."""

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)

class Generator(nn.Module):
    def __init__(self, in_channels=567, z_channels=128):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.z_channels = z_channels

        self.preprocess = Conv1d(in_channels, 768, kernel_size=3, padding=1)
        self.gblocks = nn.ModuleList([
            GBlock(768, 768, z_channels, 1),
            GBlock(768, 768, z_channels, 1),
            GBlock(768, 384, z_channels, 2),
            GBlock(384, 384, z_channels, 2),
            GBlock(384, 384, z_channels, 2),
            GBlock(384, 192, z_channels, 3),
            GBlock(192, 96, z_channels, 5)
        ])
        self.postprocess = nn.Sequential(
            Conv1d(96, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, z):
        inputs = self.preprocess(inputs)
        outputs = inputs
        for (i, layer) in enumerate(self.gblocks):
            outputs = layer(outputs, z)
        outputs = self.postprocess(outputs)
        return outputs

class GBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_channels, upsample_factor):
        super(GBlock, self).__init__()

        self.condition_batchnorm1 = ConditionalBatchNorm1d(in_channels, z_channels)
        self.first_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            UpsampleNet(in_channels, in_channels, upsample_factor),
            Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        )

        self.condition_batchnorm2 = ConditionalBatchNorm1d(hidden_channels, z_channels)
        self.second_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2, padding=2)
        )

        self.residual1 = nn.Sequential(
            UpsampleNet(in_channels, in_channels, upsample_factor),
            Conv1d(in_channels, hidden_channels, kernel_size=1)
        )

        self.condition_batchnorm3 = ConditionalBatchNorm1d(hidden_channels, z_channels)
        self.third_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=4, padding=4)
        )

        self.condition_batchnorm4 = ConditionalBatchNorm1d(hidden_channels, z_channels)
        self.fourth_stack = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=8, padding=8)
        )

    def forward(self, condition, z):
        inputs = condition

        outputs = self.condition_batchnorm1(inputs, z)
        outputs = self.first_stack(outputs)
        outputs = self.condition_batchnorm2(outputs, z)
        outputs = self.second_stack(outputs)

        residual_outputs = self.residual1(inputs) + outputs

        outputs = self.condition_batchnorm3(residual_outputs, z)
        outputs = self.third_stack(outputs)
        outputs = self.condition_batchnorm4(outputs, z)
        outputs = self.fourth_stack(outputs)

        outputs = outputs + residual_outputs

        return outputs

class UpsampleNet(nn.Module):
    def __init__(self, input_size, output_size, upsample_factor):
        super(UpsampleNet, self).__init__()
        self.layer = nn.ConvTranspose1d(
            input_size, output_size, upsample_factor * 2,
            upsample_factor, padding=upsample_factor // 2
        )
        nn.init.orthogonal_(self.layer.weight)
        self.layer = nn.utils.spectral_norm(self.layer)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.layer.stride[0]]
        return outputs

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, z_channels=128):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, affine=False)
        self.layer = nn.utils.spectral_norm(nn.Linear(z_channels, num_features * 2))
        self.layer.weight.data.normal_(1, 0.02)  # Initialize scale at N(1, 0.02)
        self.layer.bias.data.zero_()             # Initialize bias at 0

    def forward(self, inputs, noise):
        outputs = self.batch_norm(inputs)
        gamma, beta = self.layer(noise).chunk(2, 1)
        gamma = gamma.view(-1, inputs.size(1), 1)
        beta = beta.view(-1, inputs.size(1), 1)
        outputs = gamma * outputs + beta
        return outputs

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        # Simple CNN discriminator for example purposes
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

# Define Encoder if needed
class Encoder(nn.Module):
    def __init__(self, in_channels=1, z_dim=100):
        super(Encoder, self).__init__()
        # Simple CNN encoder
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

# ------------------ Training Code ------------------

# Function to set random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to generate noise
def generate_noise(batch_size, z_dim, device):
    return torch.randn(batch_size, z_dim, device=device)

# Orthogonal loss function
def orthogonal_loss(feature1, feature2):
    inner_product = torch.sum(feature1 * feature2, dim=1)
    norm1 = torch.norm(feature1, dim=1)
    norm2 = torch.norm(feature2, dim=1)
    cosine_similarity = inner_product / (norm1 * norm2 + 1e-8)
    return torch.mean(cosine_similarity**2)

# Wasserstein loss
def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_pred * y_true)

# Gradient penalty
def compute_gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)  # Adjust dimensions
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

# Function to visualize generated mel spectrograms
def visualize_generated_mel_spectrograms(generators, z_dim, num_samples, device, epoch, output_dir):
    # This function depends on your specific data and processing
    # For the purpose of this example, we'll skip implementation
    pass

# Training function for multiple generators
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
                fakes = [gen(noises[idx], z=noises[idx]).detach() for idx, gen in enumerate(generators)]

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
                fake = gen(real, z=noise)
                disc_fake = discriminator(fake)

                # Wasserstein loss for generator
                loss_gen = wasserstein_loss(disc_fake, real_label)

                # Compute orthogonal loss with other generators
                gen_feature = encoder(fake)
                ortho_loss_total = 0
                for other_idx, other_gen in enumerate(generators):
                    if idx != other_idx:
                        other_noise = generate_noise(batch_size, z_dim, device)
                        other_fake = other_gen(real, z=other_noise)
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

# Pretraining function
def pretrain_single_generator(num_epochs, z_dim, lr_gen, lr_disc, batch_size, seed, dataset, output_dir):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    generator = Generator(in_channels=567, z_channels=z_dim).to(device)
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
            fake = generator(real, z=noise)

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
            fake = generator(real, z=noise)
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
        new_generator = Generator(in_channels=567, z_channels=z_dim).to(device)
        new_generator.load_state_dict(pretrained_generator.state_dict())
        generators.append(new_generator)
    return generators



# ------------------ Main Execution ------------------

if __name__ == '__main__':
    # Define your dataset
    # Assuming you have a PyTorch dataset named 'MelSpectrogramDataset'
    # from dataset import MelSpectrogramDataset

    # For this example, we'll use random data
    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, in_channels, seq_length):
            self.num_samples = num_samples
            self.data = torch.randn(num_samples, in_channels, seq_length)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = RandomDataset(num_samples=1000, in_channels=567, seq_length=128)

    # Pretrain the generator
    pretrained_generator = pretrain_single_generator(
        num_epochs=5,
        z_dim=128,
        lr_gen=0.0001,
        lr_disc=0.0004,
        batch_size=16,
        seed=1234,
        dataset=dataset,
        output_dir='pretrain_outputsCNN'
    )

    # Train multiple generators
    trained_generators = train_gan_with_pretrained_generators(
        pretrained_generator=pretrained_generator,
        num_epochs=5,
        z_dim=128,
        lr_gen=0.0001,
        lr_disc=0.0004,
        batch_size=8,
        num_generators=2,
        seed=1234,
        dataset=dataset,
        output_dir='multi_gen_outputs_CNN',
        lambda_ortho=0.05
    )
