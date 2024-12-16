import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from models.generator import Generator
from models.v2_discriminator import Discriminator
from utils.optimizer import Optimizer
from utils.loss import MultiResolutionSTFTLoss
from utils.dataset import CustomerDataset, CustomerCollate
from models.encoder import Encoder

def create_models(args):
    generators = [Generator(args.local_condition_dim, args.z_dim) for _ in range(args.num_generators)]
    discriminator = Discriminator()
    encoder = Encoder(input_channels=1, feature_dim=128)

    return generators, discriminator, encoder

def calculate_orthogonal_loss(encoder, generator_outputs):
    # Pass each generator output through the encoder to get feature representations
    features = [encoder(output) for output in generator_outputs]
    orthogonal_loss = 0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            # Calculate inner product between feature vectors and normalize
            f_i, f_j = features[i], features[j]
            inner_product = (f_i * f_j).sum(dim=1)
            norm_product = (f_i.norm(dim=1) * f_j.norm(dim=1))
            orthogonal_loss += (inner_product / norm_product).mean()
    return orthogonal_loss

def train(args):
    train_dataset = CustomerDataset(
        args.input, upsample_factor=args.hop_length, local_condition=True, global_condition=False
    )
    collate_fn = CustomerCollate(
        upsample_factor=args.hop_length,
        condition_window=args.condition_window,
        local_condition=True,
        global_condition=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn
    )

    device = torch.device("cuda" if args.use_cuda else "cpu")

    generators, discriminator, encoder = create_models(args)
    for g in generators:
        g.to(device)
    discriminator.to(device)
    encoder.to(device)

    g_optimizers = [optim.Adam(g.parameters(), lr=args.g_learning_rate) for g in generators]
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    criterion = nn.MSELoss()
    stft_criterion = MultiResolutionSTFTLoss().to(device)

    global_step = 0
    for epoch in range(args.epochs):
        for batch, (samples, conditions) in enumerate(train_loader):
            samples = samples.to(device)
            conditions = conditions.to(device)
            z = torch.randn(samples.size(0), args.z_dim).to(device)

            # Forward pass for all generators
            generator_outputs = [g(conditions, z) for g in generators]

            # Train discriminator
            real_output = discriminator(samples)
            fake_outputs = [discriminator(output.detach()) for output in generator_outputs]

            real_loss = criterion(real_output, torch.ones_like(real_output))
            fake_loss = sum(criterion(fake, torch.zeros_like(fake)) for fake in fake_outputs) / len(fake_outputs)
            d_loss = real_loss + fake_loss

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generators
            orthogonal_loss = calculate_orthogonal_loss(encoder, generator_outputs)
            g_losses = []
            for g_output, g_optimizer in zip(generator_outputs, g_optimizers):
                adv_loss = criterion(discriminator(g_output), torch.ones_like(real_output))
                sc_loss, mag_loss = stft_criterion(g_output.squeeze(1), samples.squeeze(1))
                g_loss = adv_loss * args.lambda_adv + sc_loss + mag_loss + args.lambda_orth * orthogonal_loss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                g_losses.append(g_loss.item())

            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"Step {global_step}: D Loss: {d_loss.item():.4f}, G Loss: {sum(g_losses)/len(g_losses):.4f}, Orthogonal Loss: {orthogonal_loss.item():.4f}")

    print("Training Complete.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--g_learning_rate', type=float, default=0.0001)
    parser.add_argument('--d_learning_rate', type=float, default=0.0001)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--num_generators', type=int, default=2)
    parser.add_argument('--lambda_adv', type=float, default=1.0)
    parser.add_argument('--lambda_orth', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--condition_window', type=int, default=100)

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
