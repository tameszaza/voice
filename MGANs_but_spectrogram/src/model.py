import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import glob

# Set random seed for reproducibility
manualSeed = 6789
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(manualSeed)

# Generator shared layers
class GeneratorSharedLayers(nn.Module):
    def __init__(self, ngf, nc, mel_bins, time_frames):
        super(GeneratorSharedLayers, self).__init__()
        self.main = nn.Sequential(
            # Input: (ngf * 8, mel_bins // 8, time_frames // 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf * 4, mel_bins // 4, time_frames // 4)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf * 2, mel_bins // 2, time_frames // 2)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf, mel_bins, time_frames)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, input):
        return self.main(input)

# Generator with unique input layer and shared layers
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, shared_layers, mel_bins, time_frames):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.mel_bins = mel_bins
        self.time_frames = time_frames
        self.input_layer = nn.Sequential(
            nn.Linear(nz, ngf * 8 * (mel_bins // 8) * (time_frames // 8)),
            nn.BatchNorm1d(ngf * 8 * (mel_bins // 8) * (time_frames // 8)),
            nn.ReLU(True)
        )
        self.shared_layers = shared_layers


    def forward(self, input):
        x = self.input_layer(input)
        x = x.view(-1, self.ngf * 8, self.mel_bins // 8, self.time_frames // 8)
        x = self.shared_layers(x)
        return x


# Discriminator shared layers
class DiscriminatorSharedLayers(nn.Module):
    def __init__(self, ndf, nc):
        super(DiscriminatorSharedLayers, self).__init__()
        self.main = nn.Sequential(
            # Input: (nc, mel_bins, time_frames)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf, mel_bins // 2, time_frames // 2)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf * 2, mel_bins // 4, time_frames // 4)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf * 4, mel_bins // 8, time_frames // 8)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf * 8, mel_bins // 16, time_frames // 16)
        )

    def forward(self, input):
        return self.main(input)

# Discriminator with shared layers and unique output layers
class Discriminator(nn.Module):
    def __init__(self, ndf, nc, shared_layers, num_gens, mel_bins, time_frames):
        super(Discriminator, self).__init__()
        self.shared_layers = shared_layers
        self.output_bin = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.output_mul = nn.Sequential(
            nn.Conv2d(ndf * 8, num_gens, 4, 1, 0, bias=False)
        )


    def forward(self, input):
        x = self.shared_layers(input)
        output_bin = self.output_bin(x).view(-1, 1).squeeze(1)
        output_mul = self.output_mul(x).squeeze()
        return output_bin, output_mul

# MGAN class encapsulating the training loop
class MGAN:
    def __init__(self, num_z, beta, num_gens, batch_size, z_prior, learning_rate,
                 num_epochs, img_size, num_gen_feature_maps, num_dis_feature_maps,
                 sample_dir, device):
        self.num_z = num_z
        self.beta = beta
        self.num_gens = num_gens
        self.batch_size = batch_size
        self.z_prior = z_prior
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.img_size = img_size
        self.ngf = num_gen_feature_maps
        self.ndf = num_dis_feature_maps
        self.sample_dir = sample_dir
        self.device = device

        self.mel_bins, self.num_frames, self.num_channels = img_size
        self.history = {'d_loss': [], 'g_loss': []}

        self._build_model()

    def _build_model(self):
        mel_bins, time_frames, num_channels = self.img_size

        self.shared_gen_layers = GeneratorSharedLayers(self.ngf, num_channels,mel_bins,time_frames).to(self.device)
        self.generators = nn.ModuleList([
            Generator(self.num_z, self.ngf, num_channels, self.shared_gen_layers, mel_bins, time_frames).to(self.device)
            for _ in range(self.num_gens)
        ])
        self.shared_dis_layers = DiscriminatorSharedLayers(self.ndf, num_channels).to(self.device)
        self.discriminator = Discriminator(self.ndf, num_channels, self.shared_dis_layers, self.num_gens, mel_bins, time_frames).to(self.device)

        self.optimizerD = optim.Adam(
            list(self.discriminator.parameters()) + list(self.shared_dis_layers.parameters()),
            lr=self.learning_rate, betas=(0.5, 0.999)
        )
        gen_params = [param for gen in self.generators for param in gen.input_layer.parameters()]
        gen_params += list(self.shared_gen_layers.parameters())
        self.optimizerG = optim.Adam(gen_params, lr=self.learning_rate, betas=(0.5, 0.999))

        self.criterion_bin = nn.BCELoss()
        self.criterion_mul = nn.CrossEntropyLoss()

    def fit(self, trainloader):
        fixed_noise = self._sample_z(self.num_gens * 16).to(self.device)

        real_label = 1.0
        fake_label = 0.0

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            for i, data in enumerate(trainloader):
                print(f"Batch {i+1}")

                ############################
                # (1) Update Discriminator #
                ############################
                real_images = data[0].to(self.device)
                b_size = real_images.size(0)
                print(f"Real images shape: {real_images.shape}")
                label_real = torch.full((b_size,), real_label, device=self.device)

                # Forward pass real images through the discriminator
                output_bin_real, _ = self.discriminator(real_images)
                print(f"Output binary shape (real): {output_bin_real.shape}")
                d_bin_real_loss = self.criterion_bin(output_bin_real, label_real)

                # Generate fake images and corresponding labels
                fake_images = []
                gen_labels = []

                # Ensure batch_per_gen is at least 2
                batch_per_gen = max(2, b_size // self.num_gens)  

                if batch_per_gen < b_size // self.num_gens:
                    print(f"Warning: Batch size {b_size} is too small for {self.num_gens} generators. Adjusting batch_per_gen to {batch_per_gen}.")

                for idx, gen in enumerate(self.generators):
                    if idx * batch_per_gen >= b_size:
                        break  # Prevent overflow when batch size is smaller than expected
                    # Adjust z sampling to match batch_per_gen
                    z = self._sample_z(batch_per_gen).to(self.device)
                    fake_imgs = gen(z)
                    fake_images.append(fake_imgs)
                    gen_labels.append(torch.full((fake_imgs.size(0),), idx, dtype=torch.long, device=self.device))

                # Concatenate fake images and labels
                fake_images = torch.cat(fake_images, 0)  # Shape: [total_fake_batch_size, channels, mel_bins, time_frames]
                gen_labels = torch.cat(gen_labels, 0)    # Shape: [total_fake_batch_size]

                print(f"Fake images shape: {fake_images.shape}")
                print(f"Gen labels shape: {gen_labels.shape}")

                # Forward pass fake images through the discriminator
                output_bin_fake, output_mul_fake = self.discriminator(fake_images.detach())
                print(f"Output binary shape (fake): {output_bin_fake.shape}")
                print(f"Output multi-class shape: {output_mul_fake.shape}")

                # Ensure shapes are compatible for loss computation
                output_mul_fake = output_mul_fake.view(-1, self.num_gens)  # Shape: [total_fake_batch_size, num_gens]
                gen_labels = gen_labels.view(-1)                          # Shape: [total_fake_batch_size]
                d_mul_loss = self.criterion_mul(output_mul_fake, gen_labels)


                if output_mul_fake.shape[0] != gen_labels.shape[0]:
                    print(f"Shape mismatch! output_mul_fake: {output_mul_fake.shape}, gen_labels: {gen_labels.shape}")
                    raise ValueError("Mismatch in shapes for discriminator multi-class output and labels.")

                # Compute discriminator losses
                label_fake = torch.full((output_bin_fake.size(0),), fake_label, device=self.device)
                d_bin_fake_loss = self.criterion_bin(output_bin_fake, label_fake)
                d_mul_loss = self.criterion_mul(output_mul_fake, gen_labels)

                d_loss = d_bin_real_loss + d_bin_fake_loss + d_mul_loss * self.beta
                d_loss.backward()
                self.optimizerD.step()

                ##########################
                # (2) Update Generators #
                ##########################
                for gen in self.generators:
                    gen.zero_grad()
                self.shared_gen_layers.zero_grad()

                label_real = torch.full((output_bin_fake.size(0),), real_label, device=self.device)
                output_bin_fake, output_mul_fake = self.discriminator(fake_images)
                g_bin_loss = self.criterion_bin(output_bin_fake, label_real)
                g_mul_loss = self.criterion_mul(output_mul_fake.view(-1, self.num_gens), gen_labels) * self.beta

                g_loss = g_bin_loss + g_mul_loss
                g_loss.backward()
                self.optimizerG.step()

                # Save losses for plotting
                self.history['d_loss'].append(d_loss.item())
                self.history['g_loss'].append(g_loss.item())

            print(f"[{epoch+1}/{self.num_epochs}] d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}")

            # Save samples every few epochs
            if (epoch + 1) % 5 == 0:
                self._save_samples(epoch + 1, fixed_noise)

        # Plot loss history after training
        self._plot_history()



    def _sample_z(self, size):
        if self.z_prior == "uniform":
            return torch.rand(size, self.num_z) * 2 - 1
        return torch.randn(size, self.num_z)

    def _save_samples(self, epoch, fixed_noise):
        with torch.no_grad():
            fake_images = []
            for idx, gen in enumerate(self.generators):
                noise = fixed_noise[idx * 16:(idx + 1) * 16].to(self.device)
                gen.eval()
                fake_imgs = gen(noise)
                gen.train()
                fake_images.append(fake_imgs)

            fake_images = torch.cat(fake_images, 0)
            fake_images = (fake_images + 1) / 2.0
            os.makedirs(self.sample_dir, exist_ok=True)
            sample_path = os.path.join(self.sample_dir, f"epoch_{epoch:04d}.png")
            vutils.save_image(fake_images, sample_path, nrow=16, padding=2, normalize=True)
            print(f"Saved samples to {sample_path}")

    def _plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['d_loss'], label="D Loss")
        plt.plot(self.history['g_loss'], label="G Loss")
        plt.title("Loss During Training")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
