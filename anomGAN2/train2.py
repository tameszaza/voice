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

# -----------------------------------------------------------------------------
#  Save models (no Bandit)
# -----------------------------------------------------------------------------
def save_all_models(G, E, D, C, log_dir, suffix, opt_D=None, opt_Gs=None, opt_E=None, opt_C=None):
    torch.save(G.state_dict(), os.path.join(log_dir, f"G_{suffix}.pt"))
    torch.save(E.state_dict(), os.path.join(log_dir, f"E_{suffix}.pt"))
    torch.save(D.state_dict(), os.path.join(log_dir, f"D_{suffix}.pt"))
    torch.save(C.state_dict(), os.path.join(log_dir, f"C_{suffix}.pt"))
    # Save optimizers only if provided (used only for interrupt)
    if opt_D is not None:
        torch.save(opt_D.state_dict(), os.path.join(log_dir, f"opt_D_{suffix}.pt"))
    if opt_Gs is not None:
        for i, opt_G in enumerate(opt_Gs):
            torch.save(opt_G.state_dict(), os.path.join(log_dir, f"opt_G_{i}_{suffix}.pt"))
    if opt_E is not None:
        torch.save(opt_E.state_dict(), os.path.join(log_dir, f"opt_E_{suffix}.pt"))
    if opt_C is not None:
        torch.save(opt_C.state_dict(), os.path.join(log_dir, f"opt_C_{suffix}.pt"))

def load_saved_models(G, E, D, C, weights_dir, epoch, opt_D=None, opt_Gs=None, opt_E=None, opt_C=None):
    G.load_state_dict(torch.load(os.path.join(weights_dir, f"G_{epoch}.pt")))
    E.load_state_dict(torch.load(os.path.join(weights_dir, f"E_{epoch}.pt")))
    D.load_state_dict(torch.load(os.path.join(weights_dir, f"D_{epoch}.pt")))
    C.load_state_dict(torch.load(os.path.join(weights_dir, f"C_{epoch}.pt")))
    # Load optimizers if provided
    if opt_D is not None:
        opt_D_path = os.path.join(weights_dir, f"opt_D_{epoch}.pt")
        if os.path.exists(opt_D_path):
            opt_D.load_state_dict(torch.load(opt_D_path))
    if opt_Gs is not None:
        for i, opt_G in enumerate(opt_Gs):
            opt_G_path = os.path.join(weights_dir, f"opt_G_{i}_{epoch}.pt")
            if os.path.exists(opt_G_path):
                opt_G.load_state_dict(torch.load(opt_G_path))
    if opt_E is not None:
        opt_E_path = os.path.join(weights_dir, f"opt_E_{epoch}.pt")
        if os.path.exists(opt_E_path):
            opt_E.load_state_dict(torch.load(opt_E_path))
    if opt_C is not None:
        opt_C_path = os.path.join(weights_dir, f"opt_C_{epoch}.pt")
        if os.path.exists(opt_C_path):
            opt_C.load_state_dict(torch.load(opt_C_path))

# -----------------------------------------------------------------------------
#  Dataset: each subfolder under data_root is a class.
#  Inside each, there are 'mel/' and/or 'mfcc/' subfolders.
# -----------------------------------------------------------------------------
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class StackedOnlyDataset(Dataset):
    """
    Assumes under data_root/<class> you have:
        mel/stacked.npy    of shape (N_i, H, W)   if use_mel
        mfcc/stacked.npy   of shape (N_i, H, W)   if use_mfcc

    This class:
    1) Scans all classes, loads each stacked.npy exactly once.
    2) Builds two big arrays:
         - full_data: (N_total, C, H, W)  float32
         - full_labels: (N_total,)         int64
    3) Optionally converts full_data to a single torch.Tensor.
    """

    def __init__(self,
                 data_root: str,
                 use_mel: bool = True,
                 use_mfcc: bool = True,
                 preload: bool = True):
        if not (use_mel or use_mfcc):
            raise ValueError("At least one of use_mel or use_mfcc must be True")

        classes = sorted(
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        )
        if not classes:
            raise ValueError(f"No subfolders in {data_root}")

        data_list = []
        label_list = []

        for label, cls in enumerate(classes):
            base = os.path.join(data_root, cls)
            arrs = []

            if use_mel:
                mel_path = os.path.join(base, "mel", "stacked.npy")
                if not os.path.isfile(mel_path):
                    raise FileNotFoundError(f"Missing {mel_path}")
                arr_m = np.load(mel_path)       # shape: (N_i, H, W)
                arrs.append(arr_m)

            if use_mfcc:
                mfcc_path = os.path.join(base, "mfcc", "stacked.npy")
                if not os.path.isfile(mfcc_path):
                    raise FileNotFoundError(f"Missing {mfcc_path}")
                arr_f = np.load(mfcc_path)     # shape: (N_i, H, W)
                arrs.append(arr_f)

            # stack channels → (N_i, C, H, W)
            x_cls = np.stack(arrs, axis=1).astype(np.float32)
            n_i = x_cls.shape[0]

            data_list.append(x_cls)
            label_list.append(np.full((n_i,), label, dtype=np.int64))

        # concatenate across classes → (N_total, C, H, W) & (N_total,)
        full_data   = np.concatenate(data_list,  axis=0)
        full_labels = np.concatenate(label_list, axis=0)

        if preload:
            # one big tensor in RAM
            self.data   = torch.from_numpy(full_data)
            self.labels = torch.from_numpy(full_labels)
        else:
            # keep numpy around, convert per-sample
            self.data   = full_data
            self.labels = full_labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if isinstance(self.data, torch.Tensor):
            # very fast: single tensor & label lookup
            return self.data[idx], self.labels[idx]

        # numpy fallback: one slice + conversion
        sample = self.data[idx]            # shape (C, H, W), float32
        label  = int(self.labels[idx])
        return torch.from_numpy(sample), label


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
        f.write(f"use_mel   : {args.use_mel}\n")
        f.write(f"use_mfcc  : {args.use_mfcc}\n")
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

    # dataset & loader
    ds = StackedOnlyDataset(
        args.data_root, use_mel=args.use_mel, use_mfcc=args.use_mfcc
    )
    x0, _ = ds[0]
    n_features, H, W = x0.shape

    save_config(args, n_features)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)

    # models
    G = MultiGenerator(
        args.z_dim, out_channels=n_features,
        base_channels=32, img_size=H,
        n_layers=4, n_clusters=args.n_clusters
    ).to(device)
    E = MultiEncoder(
        in_channels=n_features, z_dim=args.z_dim,
        base_channels=32, img_size=H,
        n_layers=4, n_clusters=args.n_clusters
    ).to(device)
    D = Discriminator(
        in_channels=n_features, base_channels=32,
        n_layers=4
    ).to(device)

    # classifier input channels
    with torch.no_grad():
        dummy = torch.zeros(1, n_features, H, W, device=device)
        feat = D.intermediate(dummy)
    cls_in = feat.shape[1]
    C = Classifier(in_channels=cls_in, n_clusters=args.n_clusters).to(device)

    # optimizers
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_Gs = [
        torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.0, 0.9))
        for g in G.generators
    ]
    opt_Es = [
        torch.optim.Adam(enc.parameters(), lr=args.lr, betas=(0.0, 0.9))
        for enc in E.encoders
    ]

    opt_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0.0, 0.9))

    writer = SummaryWriter(args.log_dir)
    d_steps = 0
    global_step = 0

    # Load saved weights if resuming
    if args.resume:
        print(f"Resuming from epoch {start_epoch}")
        load_saved_models(G, E, D, C, args.resume, start_epoch,
                          opt_D=opt_D, opt_Gs=opt_Gs, opt_E=opt_Es, opt_C=opt_C)

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
                    # — Generator + classification + latent consistency, per cluster —
                    # for each cluster i present in this batch
                    for i in torch.unique(k):
                        idx = (k == i).nonzero(as_tuple=False).view(-1)
                        z_i = z[idx]
                        real_hw = real.shape[-2:]
                        # generate only for cluster i
                        fake_i = G.generators[i](z_i, None)
                        if fake_i.shape[-2:] != real_hw:
                            fake_i = F.interpolate(fake_i, size=real_hw,
                                                mode='bilinear', align_corners=False)

                        # adversarial loss
                        adv_i    = -D(fake_i).mean()
                        # classification loss on generator outputs
                        feat_i   = D.intermediate(fake_i)
                        logits_i = C(feat_i)
                        cls_i    = F.cross_entropy(logits_i, k[idx])
                        # latent consistency loss
                        z_hat_i  = E(fake_i.detach(), k[idx])
                        lat_i    = F.mse_loss(z_hat_i, z_i)

                        loss_G_i = adv_i - args.lambda_cls * cls_i + args.lambda_lat * lat_i

                        # update only generator i
                        opt_Gs[i].zero_grad()
                        loss_G_i.backward(retain_graph=True)
                        opt_Gs[i].step()

                        # log per-cluster
                        writer.add_scalar(f'Loss/G_cluster_{i}', loss_G_i.item(), global_step)
                        writer.add_scalar(f'Loss/Cls_G_cluster_{i}', cls_i.item(), global_step)
                        writer.add_scalar(f'Loss/Lat_cluster_{i}', lat_i.item(), global_step)

                    # Encoder standalone update
                    # Encoder per-cluster update
                    z_hat2 = E(fake.detach(), k)
                    z_target = z.detach()
                    for i in torch.unique(k):
                        idx      = (k == i).nonzero(as_tuple=False).view(-1)
                        loss_E_i = F.mse_loss(z_hat2[idx], z_target[idx])
                        opt_Es[i].zero_grad()
                        loss_E_i.backward(retain_graph=True)
                        opt_Es[i].step()
                        writer.add_scalar(f'Loss/E_cluster_{i}', loss_E_i.item(), global_step)


                    # Classifier standalone update
                    opt_C.zero_grad()
                    logits2 = C(D.intermediate(fake.detach()))
                    loss_C2 = F.cross_entropy(logits2, k)
                    loss_C2.backward()
                    opt_C.step()
                    # log scalars as before
                    
                    writer.add_scalar('Loss/C',     loss_C2.item(),  global_step)

                global_step += 1

            print(f"Epoch {epoch+1}/{args.epochs} — D={loss_D:.4f}")
            if (epoch + 1) % 20 == 0:
                save_all_models(G, E, D, C, args.log_dir, epoch+1)

        save_all_models(G, E, D, C, args.log_dir, args.epochs)
    except KeyboardInterrupt:
        print("Interrupted—saving models with opt…")
        save_all_models(G, E, D, C, args.log_dir, f"interrupt_{epoch+1}",
                        opt_D=opt_D, opt_Gs=opt_Gs, opt_E=opt_E, opt_C=opt_C)
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
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--epochs",     type=int,   default=700)
    p.add_argument("--z_dim",      type=int,   default=128)
    p.add_argument("--n_clusters", type=int,   default=8)
    p.add_argument("--n_critic",   type=int,   default=5)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--lambda_cls", type=float, default=0.1)
    p.add_argument("--lambda_lat", type=float, default=1.0,
                   help="Weight for latent-consistency loss")
    p.add_argument("--gamma_gp",   type=float, default=5.0)
    p.add_argument("--use_mel",    action="store_true",
                   help="Include mel channel")
    p.add_argument("--use_mfcc",   action="store_true",
                   help="Include mfcc channel")
    p.add_argument("--resume", type=str, 
                   help="Path to directory containing saved models and configuration.txt")
    args = p.parse_args()

    if args.resume:
        args.log_dir = args.resume
    else:
        os.makedirs(args.log_dir, exist_ok=True)
        
        train(args)