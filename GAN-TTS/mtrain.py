import torch
from utils.dataset import CustomerDataset, CustomerCollate
from torch.utils.data import DataLoader
import torch.nn.parallel.data_parallel as parallel
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import time
from models.generator import Generator
from models.v2_discriminator import Discriminator
from models.encoder import Encoder
from tensorboardX import SummaryWriter
from utils.optimizer import Optimizer
from utils.audio import hop_length
from utils.loss import MultiResolutionSTFTLoss


def load_checkpoint(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)

def create_models_and_restore(args, device):
    generators = [Generator(args.local_condition_dim, args.z_dim) for _ in range(args.num_generators)]
    discriminator = Discriminator()
    encoder = Encoder(input_channels=1, feature_dim=128)
    
    global_step = 0

    if args.mgan_checkpoint and os.path.exists(args.mgan_checkpoint):
        checkpoint = load_checkpoint(args.mgan_checkpoint, device)
        for idx, g in enumerate(generators):
            g.load_state_dict(checkpoint[f"generator_{idx}"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        encoder.load_state_dict(checkpoint["encoder"])
        global_step = checkpoint["global_step"]
        print(f"Resumed from MGAN checkpoint at step {global_step}")
    elif args.single_checkpoint and os.path.exists(args.single_checkpoint):
        checkpoint = load_checkpoint(args.single_checkpoint, device)
        for idx, g in enumerate(generators):
            g.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        print("Initialized from single generator training checkpoint")
    else:
        print("No checkpoint found, initializing all models from scratch.")

    return generators, discriminator, encoder, global_step

def save_checkpoint(args, generators, discriminator, encoder, g_optimizers, d_optimizer, step):
    checkpoint = {
        f"generator_{idx}": g.state_dict() for idx, g in enumerate(generators)
    }
    checkpoint["discriminator"] = discriminator.state_dict()
    checkpoint["encoder"] = encoder.state_dict()
    checkpoint["g_optimizers"] = [opt.state_dict() for opt in g_optimizers]
    checkpoint["d_optimizer"] = d_optimizer.state_dict()
    checkpoint["global_step"] = step

    checkpoint_path = os.path.join(args.checkpoint_dir, f"mgan_checkpoint_step_{step}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved MGAN checkpoint at step {step}: {checkpoint_path}")

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset = CustomerDataset(
        args.input, upsample_factor=hop_length, local_condition=True, global_condition=False)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    generators, discriminator, encoder, global_step = create_models_and_restore(args, device)

    for g in generators:
        g.to(device)
    discriminator.to(device)
    encoder.to(device)

    g_optimizers = [optim.Adam(g.parameters(), lr=args.g_learning_rate) for g in generators]
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    stft_criterion = MultiResolutionSTFTLoss().to(device)
    criterion = nn.MSELoss().to(device)

    for epoch in range(args.epochs):
        collate = CustomerCollate(
            upsample_factor=hop_length,
            condition_window=args.condition_window,
            local_condition=True,
            global_condition=False)

        train_loader = DataLoader(
            train_dataset, collate_fn=collate, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

        for batch, (samples, conditions) in enumerate(train_loader):
            batch_size = int(conditions.shape[0])
            samples = samples.to(device)
            conditions = conditions.to(device)
            z = torch.randn(batch_size, args.z_dim).to(device)

            generator_outputs = [g(conditions, z) for g in generators]

            real_output = discriminator(samples)
            fake_outputs = [discriminator(output.detach()) for output in generator_outputs]

            real_loss = criterion(real_output, torch.ones_like(real_output))
            fake_loss = sum(criterion(fake, torch.zeros_like(fake)) for fake in fake_outputs) / len(fake_outputs)
            d_loss = real_loss + fake_loss

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            orthogonal_loss = sum(
                torch.norm(encoder(output) - encoder(other_output))
                for i, output in enumerate(generator_outputs)
                for other_output in generator_outputs[i+1:]
            )

            for g_output, g_optimizer in zip(generator_outputs, g_optimizers):
                adv_loss = criterion(discriminator(g_output), torch.ones_like(real_output))
                sc_loss, mag_loss = stft_criterion(g_output.squeeze(1), samples.squeeze(1))
                g_loss = adv_loss * args.lambda_adv + sc_loss + mag_loss + args.lambda_orth * orthogonal_loss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            global_step += 1

            if global_step % args.checkpoint_step == 0:
                save_checkpoint(args, generators, discriminator, encoder, g_optimizers, d_optimizer, global_step)

            if global_step % args.summary_step == 0:
                print(f"Step {global_step}: D Loss: {d_loss.item():.4f}, Orthogonal Loss: {orthogonal_loss.item():.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--single_checkpoint', type=str, default=None, help="Path to single-generator checkpoint")
    parser.add_argument('--mgan_checkpoint', type=str, default=None, help="Path to MGAN checkpoint")
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
    parser.add_argument('--checkpoint_step', type=int, default=5000)
    parser.add_argument('--summary_step', type=int, default=100)

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
