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
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

def save_checkpoint(args, generators, discriminator, encoder, g_optimizers, d_optimizer, step):
    checkpoint = {}
    # Collect all generator states
    for idx, gen in enumerate(generators):
        checkpoint[f"generator_{idx}"] = gen.state_dict()
    # Discriminator, encoder, and optimizer states
    checkpoint["discriminator"] = discriminator.state_dict()
    checkpoint["encoder"] = encoder.state_dict()
    checkpoint["d_optimizer"] = d_optimizer.state_dict()
    for idx, opt in enumerate(g_optimizers):
        checkpoint[f"g_optimizer_{idx}"] = opt.state_dict()
    checkpoint["global_step"] = step

    checkpoint_path = os.path.join(args.checkpoint_dir, f"mgan_step_{step}.pth")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {step}: {checkpoint_path}")

def load_checkpoint(checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        return torch.load(checkpoint_path, map_location=device, weights_only=False)

    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

def calculate_orthogonal_loss(encoder, generator_outputs):
    """
    Calculate orthogonal loss between generator outputs.
    Each generator output is shape [B, 1, T].
    We pass them through `encoder` (which might produce [B, feature_dim]) and
    enforce orthogonality between their embeddings.
    """
    with torch.no_grad():  # <-- Typically we freeze the encoder or do not backprop from here
        features = [encoder(output.detach()) for output in generator_outputs]
    orth_loss = 0.0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f_i, f_j = features[i], features[j]
            # Normalized inner product
            inner_product = (f_i * f_j).sum(dim=1)
            norm_product = (f_i.norm(dim=1) * f_j.norm(dim=1) + 1e-8)
            orth_loss += (inner_product / norm_product).mean()
    return orth_loss

def restore_optimizers(g_optimizers, d_optimizer, checkpoint):
    for idx, opt in enumerate(g_optimizers):
        opt.load_state_dict(checkpoint[f"g_optimizer_{idx}"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    print("Optimizers restored.")

def initialize_models(args, device):
    """Initialize or restore MGAN models."""
    generators = [Generator(args.local_condition_dim, args.z_dim).to(device)
                  for _ in range(args.num_generators)]
    discriminator = Discriminator().to(device)
    encoder = Encoder(input_channels=1, feature_dim=128).to(device)

    global_step = 0
    # If MGAN checkpoint exists, load it
    if args.mgan_checkpoint and os.path.exists(args.mgan_checkpoint):
        checkpoint = load_checkpoint(args.mgan_checkpoint, device)
        for idx, g in enumerate(generators):
            g.load_state_dict(checkpoint[f"generator_{idx}"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        encoder.load_state_dict(checkpoint["encoder"])
        global_step = checkpoint["global_step"]
        print(f"Resumed from MGAN checkpoint at step {global_step}")
    # If no MGAN checkpoint, maybe load single‐generator weights
    elif args.single_checkpoint and os.path.exists(args.single_checkpoint):
        checkpoint = load_checkpoint(args.single_checkpoint, device)
        # load single generator weights into each generator
        for idx, g in enumerate(generators):
            g.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        print("Initialized multi‐generators + disc from single‐generator checkpoint.")
    else:
        print("No checkpoint found; training from scratch.")
    return generators, discriminator, encoder, global_step

def train(args):
    device = torch.device("cuda" if args.use_cuda else "cpu")
    writer = SummaryWriter(log_dir=args.checkpoint_dir)
    cudnn.benchmark = True  # Optimize GPU kernel selection
    scaler = torch.amp.GradScaler(device='cuda')

    # TorchMetrics
    accuracy_fn = BinaryAccuracy().to(device)
    precision_fn = BinaryPrecision().to(device)
    recall_fn = BinaryRecall().to(device)
    f1_fn = BinaryF1Score().to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create datasets once
    train_dataset = CustomerDataset(
        args.input, upsample_factor=hop_length,
        local_condition=True, global_condition=False
    )
    external_dataset = CustomerDataset(
        args.external_generator_data, upsample_factor=hop_length,
        local_condition=True, global_condition=False
    )

    # Initialize MGAN models
    generators, discriminator, encoder, global_step = initialize_models(args, device)

    # Optimizers
    g_optimizers = [
        optim.Adam(gen.parameters(), lr=args.g_learning_rate) for gen in generators
    ]
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    # Optionally restore optimizer states if MGAN checkpoint is present
    if args.mgan_checkpoint and os.path.exists(args.mgan_checkpoint):
        checkpoint = load_checkpoint(args.mgan_checkpoint, device)
        restore_optimizers(g_optimizers, d_optimizer, checkpoint)

    # Losses
    stft_criterion = MultiResolutionSTFTLoss().to(device)
    criterion = nn.MSELoss().to(device)

    # Create the collate function and DataLoader objects once (moved outside the epoch loop)
    collate = CustomerCollate(
        upsample_factor=hop_length,
        condition_window=args.condition_window,
        local_condition=True,
        global_condition=False
    )
    train_loader = DataLoader(
        train_dataset, collate_fn=collate,
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=True, persistent_workers=True
    )
    external_loader = DataLoader(
        external_dataset, collate_fn=collate,
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=True, persistent_workers=True
    )
    external_iter = iter(external_loader)

    # Epoch loop
    for epoch in range(args.epochs):
        for batch_idx, (samples, conditions) in enumerate(train_loader):
            samples = samples.to(device, non_blocking=True)
            conditions = conditions.to(device, non_blocking=True)
            batch_size = samples.size(0)

            # Generate noise
            z = torch.randn(batch_size, args.z_dim, device=device)

            # Forward all generators
            generator_outputs = [gen(conditions, z) for gen in generators]

            # Real data
            real_output = discriminator(samples)
            # Fake data from internal generators
            fake_outputs = [discriminator(g_out.detach()) for g_out in generator_outputs]

            # External data
            try:
                external_samples, _ = next(external_iter)
            except StopIteration:
                external_iter = iter(external_loader)
                external_samples, _ = next(external_iter)
            external_samples = external_samples.to(device, non_blocking=True)
            external_output = discriminator(external_samples)

            # Discriminator loss
            real_loss = criterion(real_output, torch.ones_like(real_output))
            fake_loss = sum(criterion(f_out, torch.zeros_like(f_out)) for f_out in fake_outputs) / len(fake_outputs)
            external_loss = criterion(external_output, torch.zeros_like(external_output))
            d_loss = real_loss + args.lambda_fake * fake_loss + args.lambda_ex * external_loss

            # Update Discriminator
            d_optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                d_loss = real_loss + args.lambda_fake * fake_loss + args.lambda_ex * external_loss
            scaler.scale(d_loss).backward()  # Scales loss to avoid underflow
            scaler.step(d_optimizer)
            scaler.update()  # Updates the scaler for next iteration

            #################################
            #   Compute Generator Loss (MGAN)
            #################################
            total_g_loss = 0.0

            # Calculate orthogonal loss once if multiple generators
            orth_loss_val = 0.0
            if len(generators) > 1 and args.lambda_orth > 0:
                orth_loss_val = calculate_orthogonal_loss(encoder, generator_outputs)

            # For each generator
            for g_out in generator_outputs:
                fake_out = discriminator(g_out)
                adv_loss = criterion(fake_out, torch.ones_like(fake_out))
                sc_loss, mag_loss = stft_criterion(g_out.squeeze(1), samples.squeeze(1))
                g_loss = adv_loss * args.lambda_adv + sc_loss + args.lambda_mag * mag_loss

                if len(generators) > 1 and args.lambda_orth > 0:
                    g_loss += args.lambda_orth * orth_loss_val

                total_g_loss += g_loss

            # Single backward pass for the sum (update all generators at once)
            for g_optim in g_optimizers:
                g_optim.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                total_g_loss = 0.
                for g_out in generator_outputs:
                    fake_out = discriminator(g_out)
                    adv_loss = criterion(fake_out, torch.ones_like(fake_out))
                    sc_loss, mag_loss = stft_criterion(g_out.squeeze(1), samples.squeeze(1))
                    g_loss = adv_loss * args.lambda_adv + sc_loss + args.lambda_mag * mag_loss
                    total_g_loss += g_loss

            scaler.scale(total_g_loss).backward()  # Scales loss before backpropagation
            for g_optim in g_optimizers:
                scaler.step(g_optim)
            scaler.update()  # Updates scaler

            ##############################
            #   Logging / TensorBoard
            ##############################
            d_loss_val = d_loss.item()
            real_loss_val = real_loss.item()
            fake_loss_val = fake_loss.item()
            external_loss_val = external_loss.item()
            total_g_loss_val = total_g_loss.item()

            if global_step % 100 == 0:
                with torch.no_grad():
                    real_preds = (real_output > 0.5).float().view(-1)
                    fake_preds = torch.cat([(f_out > 0.5).float().view(-1) for f_out in fake_outputs], dim=0)
                    external_preds = (external_output > 0.5).float().view(-1)

                    real_targets = torch.ones_like(real_preds)
                    fake_targets = torch.zeros_like(fake_preds)
                    external_targets = torch.zeros_like(external_preds)

                    all_preds = torch.cat([real_preds, fake_preds, external_preds], dim=0)
                    all_targets = torch.cat([real_targets, fake_targets, external_targets], dim=0)

                    total_precision = precision_fn(all_preds, all_targets)
                    total_recall = recall_fn(all_preds, all_targets)
                    total_f1 = f1_fn(all_preds, all_targets)
                    total_accuracy = accuracy_fn(all_preds, all_targets)

                writer.add_scalar('Loss/Discriminator_Total', d_loss_val, global_step)
                writer.add_scalar('Loss/Discriminator_Real', real_loss_val, global_step)
                writer.add_scalar('Loss/Discriminator_Fake', fake_loss_val, global_step)
                writer.add_scalar('Loss/Discriminator_External', external_loss_val, global_step)
                writer.add_scalar('Metrics/Discriminator_Precision', total_precision, global_step)
                writer.add_scalar('Metrics/Discriminator_Recall', total_recall, global_step)
                writer.add_scalar('Metrics/Discriminator_F1', total_f1, global_step)
                writer.add_scalar('Metrics/Discriminator_Accuracy', total_accuracy, global_step)
                writer.add_scalar('Loss/Generators_Total', total_g_loss_val, global_step)
                if len(generators) > 1:
                    writer.add_scalar('Loss/Orthogonal', orth_loss_val, global_step)
                for idx, g_out in enumerate(generator_outputs):
                    sc_loss, mag_loss = stft_criterion(g_out.squeeze(1), samples.squeeze(1))
                    writer.add_scalar(f'Loss/Generator_{idx}_SpectralConvergence', sc_loss.item(), global_step)
                    writer.add_scalar(f'Loss/Generator_{idx}_Magnitude', mag_loss.item(), global_step)

            print(f"Step {global_step}: "
                  f"D_Loss={d_loss_val:.4f}, "
                  f"Orth={orth_loss_val:.4f}, "
                  f"G_Loss={total_g_loss_val:.4f}")

            global_step += 1

            if global_step % args.checkpoint_step == 0:
                save_checkpoint(args, generators, discriminator, encoder, g_optimizers, d_optimizer, global_step)

            # Cleanup references
            del samples, conditions, z
            del real_output, fake_outputs, external_samples, external_output
            torch.cuda.empty_cache()  # Optional debugging measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train', help='Directory of training data')
    parser.add_argument('--checkpoint_dir', type=str, default="logdir2", help="Directory to save model")
    parser.add_argument('--single_checkpoint', type=str, default=None, help="Path to single-generator checkpoint")
    parser.add_argument('--mgan_checkpoint', type=str, default=None, help="Path to MGAN checkpoint")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--g_learning_rate', type=float, default=0.0001)
    parser.add_argument('--d_learning_rate', type=float, default=0.0001)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--num_generators', type=int, default=2)
    parser.add_argument('--lambda_adv', type=float, default=0.02)
    parser.add_argument('--lambda_orth', type=float, default=10)
    parser.add_argument('--lambda_fake', type=float, default=1.0)
    parser.add_argument('--lambda_mag', type=float, default=2.0)
    parser.add_argument('--lambda_ex', type=float, default=1.0)
    parser.add_argument('--checkpoint_step', type=int, default=10000)
    parser.add_argument('--summary_step', type=int, default=100)
    parser.add_argument('--condition_window', type=int, default=100, help="Conditioning window size")
    parser.add_argument('--external_generator_data', type=str, required=True,
                        help="Path to external generator .wav data")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
