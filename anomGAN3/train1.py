#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


from models import MultiGenerator, MultiEncoder, Discriminator, Classifier
from helper import MultiNpyFilesDataset

# -----------------------------------------------------------------------------
#  Save models (no Bandit)
# -----------------------------------------------------------------------------
def save_all_models(G, E, D, C, log_dir, suffix, opt_G=None, opt_E=None, opt_D=None, opt_C=None):
    torch.save(G.state_dict(), os.path.join(log_dir, f"G_{suffix}.pt"))
    torch.save(E.state_dict(), os.path.join(log_dir, f"E_{suffix}.pt"))
    torch.save(D.state_dict(), os.path.join(log_dir, f"D_{suffix}.pt"))
    torch.save(C.state_dict(), os.path.join(log_dir, f"C_{suffix}.pt"))
    # Save optimizers if provided
    if opt_G is not None:
        torch.save(opt_G.state_dict(), os.path.join(log_dir, f"opt_G_{suffix}.pt"))
    if opt_E is not None:
        torch.save(opt_E.state_dict(), os.path.join(log_dir, f"opt_E_{suffix}.pt"))
    if opt_D is not None:
        torch.save(opt_D.state_dict(), os.path.join(log_dir, f"opt_D_{suffix}.pt"))
    if opt_C is not None:
        torch.save(opt_C.state_dict(), os.path.join(log_dir, f"opt_C_{suffix}.pt"))

def load_saved_models(G, E, D, C, weights_dir, epoch, opt_G=None, opt_E=None, opt_D=None, opt_C=None):
    G.load_state_dict(torch.load(os.path.join(weights_dir, f"G_{epoch}.pt")))
    E.load_state_dict(torch.load(os.path.join(weights_dir, f"E_{epoch}.pt")))
    D.load_state_dict(torch.load(os.path.join(weights_dir, f"D_{epoch}.pt")))
    C.load_state_dict(torch.load(os.path.join(weights_dir, f"C_{epoch}.pt")))
    # Load optimizers if provided and files exist
    if opt_G is not None:
        opt_g_path = os.path.join(weights_dir, f"opt_G_{epoch}.pt")
        if os.path.exists(opt_g_path):
            opt_G.load_state_dict(torch.load(opt_g_path))
    if opt_E is not None:
        opt_e_path = os.path.join(weights_dir, f"opt_E_{epoch}.pt")
        if os.path.exists(opt_e_path):
            opt_E.load_state_dict(torch.load(opt_e_path))
    if opt_D is not None:
        opt_d_path = os.path.join(weights_dir, f"opt_D_{epoch}.pt")
        if os.path.exists(opt_d_path):
            opt_D.load_state_dict(torch.load(opt_d_path))
    if opt_C is not None:
        opt_c_path = os.path.join(weights_dir, f"opt_C_{epoch}.pt")
        if os.path.exists(opt_c_path):
            opt_C.load_state_dict(torch.load(opt_c_path))

# -----------------------------------------------------------------------------
#  Save training config
# -----------------------------------------------------------------------------
def save_config(args, n_features):
    cfg = os.path.join(args.log_dir, 'configuration.txt')
    with open(cfg, 'w') as f:
        f.write("Training Configuration:\n")
        f.write("-" * 50 + "\n")
        f.write(f"data_root : {args.data_root}\n")
        f.write(f"n_clusters: {args.n_clusters}\n")
        # f.write(f"use_mel   : {args.use_mel}\n")
        # f.write(f"use_mfcc  : {args.use_mfcc}\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

def load_config(config_path):
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip()
                value = value.strip()
                # Convert string values to appropriate types
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.replace('.','',1).isdigit():
                    value = float(value) if '.' in value else int(value)
                config[key] = value
    return config

# -----------------------------------------------------------------------------
#  WGAN-GP gradient penalty
# -----------------------------------------------------------------------------
def gradient_penalty(D, real, fake, device):
    B = real.size(0)
    α = torch.rand(B, 1, 1, 1, device=device)
    inter = α * real + (1 - α) * fake
    inter.requires_grad_(True)
    d_inter = D(inter)
    grads = torch.autograd.grad(
        outputs=d_inter.sum(), inputs=inter,
        create_graph=True, retain_graph=True
    )[0]
    return ((grads.view(B, -1).norm(2, dim=1) - 1)**2).mean()

# -----------------------------------------------------------------------------
#  Training loop
# -----------------------------------------------------------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config if resuming
    start_epoch = 0
    if args.resume:
        config = load_config(os.path.join(args.resume, 'configuration.txt'))
        args.z_dim = config.get('z_dim', args.z_dim)
        args.n_clusters = config.get('n_clusters', args.n_clusters)
        args.use_mel = config.get('use_mel', args.use_mel)
        args.use_mfcc = config.get('use_mfcc', args.use_mfcc)
        start_epoch = config.get('resume_epoch', 0)

        # dataset & loader: one .npy file per class under data_root/
    ds = MultiNpyFilesDataset(
        dir_path=args.data_root,
        max_samples_per_file=None      # or set an int to limit samples per file
    )
    # inspect one sample to get channel / spatial dims
    x0, _ = ds[0]
    n_features, H, W = x0.shape

    save_config(args, n_features)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )


    # models
    G = MultiGenerator(
        args.z_dim, out_channels=n_features,
        base_channels=32, img_size=H,
        n_layers=3, n_clusters=args.n_clusters
    ).to(device)
    E = MultiEncoder(
        in_channels=n_features, z_dim=args.z_dim,
        base_channels=32, img_size=H,
        n_layers=3, n_clusters=args.n_clusters
    ).to(device)
    D = Discriminator(
        in_channels=n_features, base_channels=32,
        n_layers=3
    ).to(device)

    # classifier input channels
    with torch.no_grad():
        dummy = torch.zeros(1, n_features, H, W, device=device)
        feat = D.intermediate(dummy)
    cls_in = feat.shape[1]
    C = Classifier(in_channels=cls_in, n_clusters=args.n_clusters).to(device)

    # optimizers
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.5, 0.9))
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.5, 0.9))
    opt_E = torch.optim.Adam(E.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0.5, 0.9))

    writer = SummaryWriter(args.log_dir)
    d_steps = 0
    global_step = 0

    # Load saved weights if resuming
    if args.resume:
        print(f"Resuming from epoch {start_epoch}")
        load_saved_models(G, E, D, C, args.resume, start_epoch,
                          opt_G=opt_G, opt_E=opt_E, opt_D=opt_D, opt_C=opt_C)

    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            for real, k in loader:
                real, k = real.to(device), k.to(device)
                z = torch.randn(args.batch_size, args.z_dim, device=device)

                # — Discriminator update —
                fake = G(z, k, target_hw=real.shape[-2:])
                loss_D = (D(fake.detach()).mean()
                          - D(real).mean()
                          + args.gamma_gp * gradient_penalty(D, real, fake.detach(), device))
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
                d_steps += 1
                writer.add_scalar('Loss/D', loss_D.item(), global_step)

                # — Generator, Encoder & Classifier every n_critic steps —
                if d_steps % args.n_critic == 0:
                    # Generator + classification
                    fake = G(z, k, target_hw=real.shape[-2:])
                    d_fake2 = D(fake)
                    feat_fake = D.intermediate(fake)
                    logits = C(feat_fake)
                    loss_cls = F.cross_entropy(logits, k)
                    loss_G   = -d_fake2.mean() - args.lambda_cls * loss_cls
                    opt_G.zero_grad()
                    loss_G.backward(retain_graph=True)
                    opt_G.step()
                    writer.add_scalar('Loss/G', loss_G.item(), global_step)
                    writer.add_scalar('Loss/Cls_on_G', loss_cls.item(), global_step)

                    # Encoder reconstructs z
                    z_hat = E(fake.detach(), k)
                    loss_E = F.mse_loss(z_hat, z.detach())
                    opt_E.zero_grad()
                    loss_E.backward()
                    opt_E.step()
                    writer.add_scalar('Loss/E', loss_E.item(), global_step)

                    # Classifier standalone
                    opt_C.zero_grad()
                    logits2 = C(feat_fake.detach())
                    loss_C2 = F.cross_entropy(logits2, k)
                    loss_C2.backward()
                    opt_C.step()
                    writer.add_scalar('Loss/C', loss_C2.item(), global_step)

                global_step += 1

            print(f"Epoch {epoch+1}/{args.epochs} — D={loss_D:.4f}  G={loss_G:.4f}")
            if (epoch + 1) % 20 == 0:
                save_all_models(G, E, D, C, args.log_dir, epoch+1,
                                opt_G=opt_G, opt_E=opt_E, opt_D=opt_D, opt_C=opt_C)

        save_all_models(G, E, D, C, args.log_dir, args.epochs,
                        opt_G=opt_G, opt_E=opt_E, opt_D=opt_D, opt_C=opt_C)

    except KeyboardInterrupt:
        print("Interrupted—saving models…")
        save_all_models(G, E, D, C, args.log_dir, f"interrupt_{epoch+1}",
                        opt_G=opt_G, opt_E=opt_E, opt_D=opt_D, opt_C=opt_C)
        raise

    finally:
        writer.close()

# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True,
                   help="Root folder (e.g. data_40) with class subfolders")
    p.add_argument("--log_dir",    required=True)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--epochs",     type=int,   default=2000)
    p.add_argument("--z_dim",      type=int,   default=128)
    p.add_argument("--n_clusters", type=int,   default=1)
    p.add_argument("--n_critic",   type=int,   default=1) #maybe decrease this to 1
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--lr_D",       type=float, default=2e-5, help="Learning rate for Discriminator (overrides --lr if set)")
    p.add_argument("--lr_G",       type=float, default=2e-5, help="Learning rate for Discriminator (overrides --lr if set)")
    p.add_argument("--lambda_cls", type=float, default=0.025) # maybe decrease this to 0.01
    p.add_argument("--gamma_gp",   type=float, default=10.0)
    # p.add_argument("--use_mel",    action="store_true",
    #                help="Include mel channel")
    # p.add_argument("--use_mfcc",   action="store_true",
    #                help="Include mfcc channel")
    p.add_argument("--resume", type=str, 
                   help="Path to directory containing saved models and configuration.txt")
    args = p.parse_args()

    if args.resume:
        args.log_dir = args.resume
    else:
        os.makedirs(args.log_dir, exist_ok=True)
    
    train(args)
