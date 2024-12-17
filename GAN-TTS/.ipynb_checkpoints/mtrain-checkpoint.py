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
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score



def save_checkpoint(args, generators, discriminator, encoder, g_optimizers, d_optimizer, step):
    """Save the current state of the MGAN training."""
    checkpoint = {
        f"generator_{idx}": g.state_dict() for idx, g in enumerate(generators)
    }
    checkpoint["discriminator"] = discriminator.state_dict()
    checkpoint["encoder"] = encoder.state_dict()
    checkpoint["d_optimizer"] = d_optimizer.state_dict()
    checkpoint.update({f"g_optimizer_{idx}": opt.state_dict() for idx, opt in enumerate(g_optimizers)})
    checkpoint["global_step"] = step

    checkpoint_path = os.path.join(args.checkpoint_dir, f"mgan_step_{step}.pth")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {step}: {checkpoint_path}")

def load_checkpoint(checkpoint_path, device):
    """Load a checkpoint from the given path."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        return torch.load(checkpoint_path, map_location=device)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
def calculate_orthogonal_loss(encoder, generator_outputs):
    """Calculate orthogonal loss between generator outputs."""
    features = [encoder(output.detach()) for output in generator_outputs]
    orthogonal_loss = 0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f_i, f_j = features[i], features[j]
            inner_product = (f_i * f_j).sum(dim=1)
            norm_product = (f_i.norm(dim=1) * f_j.norm(dim=1))
            orthogonal_loss += (inner_product / norm_product).mean()
    return orthogonal_loss

def restore_optimizers(g_optimizers, d_optimizer, checkpoint):
    """Restore the optimizer states from a checkpoint."""
    for idx, opt in enumerate(g_optimizers):
        opt.load_state_dict(checkpoint[f"g_optimizer_{idx}"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    print("Optimizers restored.")


def initialize_models(args, device):
    """Initialize models and restore weights if available."""
    generators = [Generator(args.local_condition_dim, args.z_dim) for _ in range(args.num_generators)]
    discriminator = Discriminator()
    encoder = Encoder(input_channels=1, feature_dim=128)

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

    # If MGAN checkpoint doesn't exist, load single generator/discriminator checkpoint
    elif args.single_checkpoint and os.path.exists(args.single_checkpoint):
        checkpoint = load_checkpoint(args.single_checkpoint, device)
        for idx, g in enumerate(generators):
            g.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        print("Initialized generators and discriminator from single-generator checkpoint")

    else:
        print("No checkpoint found. Initializing all models from scratch.")

    return generators, discriminator, encoder, global_step

def train(args):
    device = torch.device("cuda" if args.use_cuda else "cpu")
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=args.checkpoint_dir)
    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1_score = BinaryF1Score().to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize Dataset and DataLoader
    train_dataset = CustomerDataset(
        args.input, upsample_factor=hop_length, local_condition=True, global_condition=False
    )
    external_dataset = CustomerDataset(
        args.external_generator_data, upsample_factor=hop_length, local_condition=True, global_condition=False
    )

    
    
    generators, discriminator, encoder, global_step = initialize_models(args, device)

    for g in generators:
        g.to(device)
    discriminator.to(device)
    encoder.to(device)

    # Optimizers
    g_optimizers = [optim.Adam(g.parameters(), lr=args.g_learning_rate) for g in generators]
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    # Loss Functions
    stft_criterion = MultiResolutionSTFTLoss().to(device)
    criterion = nn.MSELoss().to(device)

    for epoch in range(args.epochs):
        # Create the CustomerCollate object
        collate = CustomerCollate(
            upsample_factor=hop_length,
            condition_window=args.condition_window,
            local_condition=True,
            global_condition=False
        )
    
        # Initialize DataLoaders with CustomerCollate
        train_loader = DataLoader(
            train_dataset, collate_fn=collate, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
        )
        external_loader = DataLoader(
            external_dataset, collate_fn=collate, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
        )


        external_iter = iter(external_loader)

        
        for batch, (samples, conditions) in enumerate(train_loader):
            batch_size = conditions.size(0)
            samples = samples.to(device)
            conditions = conditions.to(device)
            z = torch.randn(batch_size, args.z_dim).to(device)

            # Generate outputs
            generator_outputs = [g(conditions, z) for g in generators]

            # Real data
            real_output = discriminator(samples)

            # Fake data from internal generators
            fake_outputs = [discriminator(output.detach()) for output in generator_outputs]

            # External generator data
            try:
                external_samples, _ = next(external_iter)
            except StopIteration:
                external_iter = iter(external_generator_loader)
                external_samples, _ = next(external_iter)
            external_samples = external_samples.to(device)
            external_output = discriminator(external_samples)

            # Discriminator loss
            real_loss = criterion(real_output, torch.ones_like(real_output))
            fake_loss = sum(criterion(fake, torch.zeros_like(fake)) for fake in fake_outputs) / len(fake_outputs)
            external_loss = criterion(external_output, torch.zeros_like(external_output))

            d_loss = real_loss + fake_loss + external_loss

            # Update discriminator
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


            orthogonal_loss = calculate_orthogonal_loss(encoder, generator_outputs) if len(generators) > 1 else 0.0

            for g_output, g_optimizer in zip(generator_outputs, g_optimizers):
                fake_output = discriminator(g_output)
                adv_loss = criterion(fake_output, torch.ones_like(fake_output))
                sc_loss, mag_loss = stft_criterion(g_output.squeeze(1), samples.squeeze(1))
            
                # Compute the total loss for the current generator
                g_loss = adv_loss * args.lambda_adv + sc_loss + mag_loss
            
                # Calculate orthogonal loss if multiple generators are present
                if len(generators) > 1:
                    orthogonal_loss = calculate_orthogonal_loss(encoder, generator_outputs)
                    g_loss += args.lambda_orth * orthogonal_loss
            
                # Backpropagation and optimization
                g_optimizer.zero_grad()
                g_loss.backward(retain_graph=True if len(generators) > 1 else False)  # Retain graph for orthogonal loss
                g_optimizer.step()

           

            # Log metrics
            if global_step % 100 == 0:
                # Discriminator Metrics
                real_acc = accuracy(real_output, torch.ones_like(real_output))
                fake_acc = accuracy(fake_outputs[0], torch.zeros_like(fake_outputs[0]))
                ext_acc = accuracy(external_output, torch.zeros_like(external_output))
                
                real_prec = precision(real_output, torch.ones_like(real_output))
                real_rec = recall(real_output, torch.ones_like(real_output))
                real_f1 = f1_score(real_output, torch.ones_like(real_output))
    
                fake_prec = precision(fake_outputs[0], torch.zeros_like(fake_outputs[0]))
                fake_rec = recall(fake_outputs[0], torch.zeros_like(fake_outputs[0]))
                fake_f1 = f1_score(fake_outputs[0], torch.zeros_like(fake_outputs[0]))
                writer.add_scalar('Loss/Discriminator_Total', d_loss.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Real', real_loss.item(), global_step)
                writer.add_scalar('Loss/Discriminator_Fake', fake_loss.item(), global_step)
                writer.add_scalar('Loss/Discriminator_External', external_loss.item(), global_step)
                
                writer.add_scalar('Metrics/Accuracy_Real', real_acc, global_step)
                writer.add_scalar('Metrics/Accuracy_Fake', fake_acc, global_step)
                writer.add_scalar('Metrics/Accuracy_External', ext_acc, global_step)
                
                writer.add_scalar('Metrics/Precision_Real', real_prec, global_step)
                writer.add_scalar('Metrics/Recall_Real', real_rec, global_step)
                writer.add_scalar('Metrics/F1_Real', real_f1, global_step)

                writer.add_scalar('Metrics/Precision_Fake', fake_prec, global_step)
                writer.add_scalar('Metrics/Recall_Fake', fake_rec, global_step)
                writer.add_scalar('Metrics/F1_Fake', fake_f1, global_step)
                # Log Discriminator Loss
                writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                
                # Log Generator Losses
                writer.add_scalar('Loss/Generator_Adv', adv_loss.item(), global_step)
                writer.add_scalar('Loss/Generator_SC', sc_loss.item(), global_step)
                writer.add_scalar('Loss/Generator_Mag', mag_loss.item(), global_step)
                
                # Log Orthogonal Loss (only if applicable)
                if len(generators) > 1:
                    writer.add_scalar('Loss/Orthogonal', orthogonal_loss, global_step)
                
                # Optionally log other metrics like Learning Rate or Gradients if needed
                for i, g_optimizer in enumerate(g_optimizers):
                    writer.add_scalar(f'LR/Generator_{i}', g_optimizer.param_groups[0]['lr'], global_step)
                
                writer.add_scalar('LR/Discriminator', d_optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

            # Save checkpoint
            if global_step % args.checkpoint_step == 0:
                save_checkpoint(args, generators, discriminator, encoder, g_optimizers, d_optimizer, global_step)

            
            if global_step % args.summary_step == 0:
                print(f"Step {global_step}: D Loss: {d_loss.item():.4f}, "
                      f"Orthogonal Loss: {orthogonal_loss:.4f} "
                      f"Adv Loss: {adv_loss.item():.4f}, "
                      f"SC Loss: {sc_loss.item():.4f}, "
                      f"Mag Loss: {mag_loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train', help='Directory of training data')
    parser.add_argument('--checkpoint_dir', type=str, default="logdir2", help="Directory to save model")
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
    parser.add_argument('--lambda_adv', type=float, default=0.01)
    parser.add_argument('--lambda_orth', type=float, default=0.1)
    parser.add_argument('--checkpoint_step', type=int, default=5000)
    parser.add_argument('--summary_step', type=int, default=1)
    parser.add_argument('--condition_window', type=int, default=100, help="Conditioning window size")
    parser.add_argument('--external_generator_data', type=str, required=True, help="Path to external generator .wav data")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()

