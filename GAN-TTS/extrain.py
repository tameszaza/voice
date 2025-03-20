import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

# ------------------------------------------------------------------
# Import your existing modules
# ------------------------------------------------------------------
from utils.dataset import CustomerDataset, CustomerCollate
from models.v2_discriminator import Discriminator
from utils.audio import hop_length
from utils.loss import MultiResolutionSTFTLoss
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

# ------------------------------------------------------------------
# Checkpoint functions for the Discriminator
# ------------------------------------------------------------------
def save_discriminator_checkpoint(args, discriminator, d_optimizer, step, suffix=""):
    """
    Save a checkpoint. If suffix == "_best", then save the best model
    in a fixed filename (replacing the old best).
    """
    checkpoint = {
        "discriminator": discriminator.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "global_step": step
    }
    if suffix == "_best":
        filename = "disc_only_best.pth"
    else:
        filename = f"disc_only_step_{step}{suffix}.pth"
    checkpoint_path = os.path.join(args.checkpoint_dir, filename)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Discriminator-only checkpoint saved at step {step}{', best model' if suffix=='_best' else ''}: {checkpoint_path}")
    return checkpoint_path

def load_discriminator_checkpoint(path, discriminator, d_optimizer, device):
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)
        discriminator.load_state_dict(checkpoint["discriminator"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        step = checkpoint["global_step"]
        print(f"Resumed from step {step}")
        return step
    else:
        raise FileNotFoundError(f"No checkpoint found at {path}")

# ------------------------------------------------------------------
# Main training loop: ONLY trains Discriminator
# ------------------------------------------------------------------
def train_discriminator_only(args):
    device = torch.device("cuda" if args.use_cuda else "cpu")
    writer = SummaryWriter(log_dir=args.checkpoint_dir)
    cudnn.benchmark = True  # Optimize GPU kernel selection

    # Create your dataset and loaders
    train_dataset = CustomerDataset(
        args.input, upsample_factor=hop_length,
        local_condition=True, global_condition=False
    )
    external_dataset = CustomerDataset(
        args.external_generator_data, upsample_factor=hop_length,
        local_condition=True, global_condition=False
    )

    collate_fn = CustomerCollate(
        upsample_factor=hop_length,
        condition_window=args.condition_window,
        local_condition=True,
        global_condition=False
    )
    train_loader = DataLoader(
        train_dataset, collate_fn=collate_fn,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    external_loader = DataLoader(
        external_dataset, collate_fn=collate_fn,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    external_iter = iter(external_loader)

    # Initialize Discriminator
    discriminator = Discriminator().to(device)

    # Loss
    criterion = nn.MSELoss().to(device)

    # Optimizer
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)
    scaler = GradScaler()

    # Learning Rate Scheduler: Reduce LR on plateau (monitors avg epoch loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        d_optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True, min_lr=args.min_lr
    )

    # Load from optional checkpoint if provided
    global_step = 0
    if args.disc_checkpoint and os.path.exists(args.disc_checkpoint):
        global_step = load_discriminator_checkpoint(args.disc_checkpoint, discriminator, d_optimizer, device)
    else:
        print("No discriminator checkpoint found; training from scratch.")

    # TorchMetrics (optional)
    accuracy_fn = BinaryAccuracy().to(device)
    precision_fn = BinaryPrecision().to(device)
    recall_fn = BinaryRecall().to(device)
    f1_fn = BinaryF1Score().to(device)

    # For best model restoration (replace old best)
    global_best_loss = float('inf')
    best_state = None
    patience_counter = 0

    try:
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_idx, (samples, conditions) in enumerate(train_loader):
                samples = samples.to(device, non_blocking=True)
                batch_size = samples.size(0)

                # Real data: label=1
                real_output = discriminator(samples)
                real_loss = criterion(real_output, torch.ones_like(real_output))

                # External data acts as 'fake'
                try:
                    external_samples, _ = next(external_iter)
                except StopIteration:
                    external_iter = iter(external_loader)
                    external_samples, _ = next(external_iter)
                external_samples = external_samples.to(device, non_blocking=True)
                external_output = discriminator(external_samples)
                external_loss = criterion(external_output, torch.zeros_like(external_output))

                total_d_loss = real_loss + external_loss

                d_optimizer.zero_grad()
                with autocast():
                    loss = real_loss + external_loss
                scaler.scale(loss).backward()
                scaler.step(d_optimizer)
                scaler.update()

                epoch_loss += total_d_loss.item()
                batch_count += 1
                global_step += 1

                # Log metrics per batch (optional)
                if global_step % 10 == 0:
                    d_loss_val = total_d_loss.item()
                    with torch.no_grad():
                        # Evaluate metrics at multiple thresholds
                        real_output_flat = real_output.view(-1)
                        external_output_flat = external_output.view(-1)
                        real_targets = torch.ones_like(real_output_flat)
                        external_targets = torch.zeros_like(external_output_flat)

                        thresholds = torch.linspace(0.1, 0.9, 9).to(device)
                        best_acc = 0
                        best_f1 = 0

                        for thr in thresholds:
                            preds_real = (real_output_flat > thr).float()
                            preds_ext = (external_output_flat > thr).float()
                            all_preds = torch.cat([preds_real, preds_ext], dim=0)
                            all_targs = torch.cat([real_targets, external_targets], dim=0)
                            f1_val = f1_fn(all_preds, all_targs)
                            acc_val = accuracy_fn(all_preds, all_targs)
                            if f1_val > best_f1:
                                best_f1 = f1_val
                            if acc_val > best_acc:
                                best_acc = acc_val

                        writer.add_scalar('Loss/Discriminator', d_loss_val, global_step)
                        writer.add_scalar('Metrics/BestF1', best_f1, global_step)
                        writer.add_scalar('Metrics/BestAcc', best_acc, global_step)

                    print(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx}/{len(train_loader)}], D_Loss={d_loss_val:.4f}")

                # Save periodic checkpoint if needed
                if (global_step % args.checkpoint_step == 0) and (global_step > 0):
                    save_discriminator_checkpoint(args, discriminator, d_optimizer, global_step)

                # Cleanup
                del samples, external_samples
                torch.cuda.empty_cache()

            # End of epoch: compute average loss
            avg_epoch_loss = epoch_loss / batch_count
            print(f"Epoch {epoch} average loss: {avg_epoch_loss:.6f}")

            # Step the LR scheduler
            scheduler.step(avg_epoch_loss)

            # Log current learning rate
            current_lr = d_optimizer.param_groups[0]['lr']
            writer.add_scalar('LearningRate', current_lr, epoch)
            print(f"Epoch {epoch} current LR: {current_lr}")

            # Check if we have an improvement; use avg_epoch_loss as criteria
            if avg_epoch_loss < global_best_loss:
                global_best_loss = avg_epoch_loss
                best_state = {
                    "discriminator": discriminator.state_dict(),
                    "d_optimizer": d_optimizer.state_dict(),
                    "global_step": global_step
                }
                patience_counter = 0
                # Save the best checkpoint in-place (replacing old best)
                save_discriminator_checkpoint(args, discriminator, d_optimizer, global_step, suffix="_best")
                print(f"New best model at epoch {epoch} with loss {avg_epoch_loss:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epoch(s).")

            # If patience is exceeded, restore best model and reset patience counter
            if patience_counter >= args.patience:
                if best_state is not None:
                    print(f"Restoring best model weights after {patience_counter} epochs with no improvement.")
                    discriminator.load_state_dict(best_state["discriminator"])
                    d_optimizer.load_state_dict(best_state["d_optimizer"])
                    patience_counter = 0

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected, saving latest checkpoint...")
        save_discriminator_checkpoint(args, discriminator, d_optimizer, global_step)
        print("Checkpoint saved. Exiting now.")
        raise

    print("Discriminator-only training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data_train/real/train',
                        help='Directory of training data (real samples)')
    parser.add_argument('--external_generator_data', type=str, default='data_train/fake/train',
                        help="Directory for negative examples or external data")
    parser.add_argument('--disc_checkpoint', type=str, default=None,
                        help="Load an existing discriminator checkpoint")
    parser.add_argument('--checkpoint_dir', type=str, default="logdir_dv2",
                        help="Where to save new checkpoints")
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_learning_rate', type=float, default=0.001)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--condition_window', type=int, default=100,
                        help="Conditioning window size (if needed by collate)")
    parser.add_argument('--checkpoint_step', type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument('--patience', type=int, default=10,
                        help="Number of epochs with no improvement to wait before reducing LR and restoring best weights")
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help="Minimum learning rate for the scheduler")

    args = parser.parse_args()
    train_discriminator_only(args)

if __name__ == "__main__":
    main()
