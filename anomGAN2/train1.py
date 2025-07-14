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
def save_all_models(G, E, D, C, log_dir, suffix):
    torch.save(G.state_dict(), os.path.join(log_dir, f"G_{suffix}.pt"))
    torch.save(E.state_dict(), os.path.join(log_dir, f"E_{suffix}.pt"))
    torch.save(D.state_dict(), os.path.join(log_dir, f"D_{suffix}.pt"))
    torch.save(C.state_dict(), os.path.join(log_dir, f"C_{suffix}.pt"))

def load_saved_models(G, E, D, C, weights_dir, epoch):
    G.load_state_dict(torch.load(os.path.join(weights_dir, f"G_{epoch}.pt")))
    E.load_state_dict(torch.load(os.path.join(weights_dir, f"E_{epoch}.pt")))
    D.load_state_dict(torch.load(os.path.join(weights_dir, f"D_{epoch}.pt")))
    C.load_state_dict(torch.load(os.path.join(weights_dir, f"C_{epoch}.pt")))

# -----------------------------------------------------------------------------
#  Dataset: each subfolder under data_root is a class.
#  Inside each, there are 'mel/' and/or 'mfcc/' subfolders.
# -----------------------------------------------------------------------------

class MultiFeatureDirectoryDataset(Dataset):
    def __init__(self, data_root, use_mel=True, use_mfcc=True):
        if not (use_mel or use_mfcc):
            raise ValueError("At least one of use_mel or use_mfcc must be True")

        # scan class folders
        classes = sorted(
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        )
        if not classes:
            raise ValueError(f"No subfolders in {data_root}")

        # each sample is (path_pair, idx, label, is_stacked)
        self.samples = []
        for label, cls in enumerate(classes):
            base = os.path.join(data_root, cls)

            # check for stacked.npy in mel/ and mfcc/
            mel_stack  = os.path.join(base, "mel",  "stacked.npy") if use_mel  else None
            mfcc_stack = os.path.join(base, "mfcc", "stacked.npy") if use_mfcc else None

            if (mel_stack and os.path.isfile(mel_stack)) or \
               (mfcc_stack and os.path.isfile(mfcc_stack)):

                # load lengths without fully loading into RAM
                arr_m = np.load(mel_stack,  mmap_mode="r") if mel_stack and os.path.isfile(mel_stack)   else None
                arr_f = np.load(mfcc_stack, mmap_mode="r") if mfcc_stack and os.path.isfile(mfcc_stack) else None

                # sanity-check dims
                if arr_m is not None and arr_m.ndim != 3:
                    raise ValueError(f"{mel_stack} must be 3D, got {arr_m.shape}")
                if arr_f is not None and arr_f.ndim != 3:
                    raise ValueError(f"{mfcc_stack} must be 3D, got {arr_f.shape}")

                n = (arr_m.shape[0] if arr_m is not None else arr_f.shape[0])
                for i in range(n):
                    self.samples.append(((mel_stack, mfcc_stack), i, label, True))
                continue

            # fallback to per-file under mel/ and mfcc/
            mel_dir  = os.path.join(base, 'mel')  if use_mel  else None
            mfcc_dir = os.path.join(base, 'mfcc') if use_mfcc else None

            if use_mel and not os.path.isdir(mel_dir):
                raise FileNotFoundError(f"Missing mel/ under {base}")
            if use_mfcc and not os.path.isdir(mfcc_dir):
                raise FileNotFoundError(f"Missing mfcc/ under {base}")

            fnames = sorted(os.listdir(mel_dir if use_mel else mfcc_dir))
            for fn in fnames:
                if not fn.endswith('.npy'):
                    continue
                m_path = os.path.join(mel_dir, fn)  if use_mel  else None
                f_path = os.path.join(mfcc_dir, fn) if use_mfcc else None
                if use_mel and use_mfcc and not os.path.exists(f_path):
                    raise FileNotFoundError(f"{f_path} missing")
                # idx is ignored in per-file mode
                self.samples.append(((m_path, f_path), None, label, False))

        if not self.samples:
            raise ValueError("No samples found!")

        # cache for __getitem__
        self._cache = [None] * len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # return cached if available
        if self._cache[idx] is not None:
            return self._cache[idx]

        path_pair, slice_idx, is_stacked = self.samples[idx]
        mel_path, mfcc_path = path_pair
        feats = []

        if is_stacked:
            # load only the needed slice
            if mel_path and os.path.isfile(mel_path):
                arr_m = np.load(mel_path, mmap_mode="r")
                feats.append(arr_m[slice_idx])
            if mfcc_path and os.path.isfile(mfcc_path):
                arr_f = np.load(mfcc_path, mmap_mode="r")
                feats.append(arr_f[slice_idx])
        else:
            # per-file
            if mel_path:
                feats.append(np.load(mel_path))
            if mfcc_path:
                feats.append(np.load(mfcc_path))

        x = np.stack(feats, axis=0).astype(np.float32)  # [C,H,W]
        tensor = (torch.from_numpy(x), self.samples[idx][2])
        self._cache[idx] = tensor
        return tensor


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
    ds = MultiFeatureDirectoryDataset(
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
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_Gs = [
    torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5, 0.9))
    for g in G.generators
]

    opt_E = torch.optim.Adam(E.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0.5, 0.9))

    writer = SummaryWriter(args.log_dir)
    d_steps = 0
    global_step = 0

    # Load saved weights if resuming
    if args.resume:
        print(f"Resuming from epoch {start_epoch}")
        load_saved_models(G, E, D, C, args.resume, start_epoch)

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
                    # — Generator + classification, per cluster (new) —
                    # for each cluster i present in this batch
                    for i in torch.unique(k):
                        idx = (k == i).nonzero(as_tuple=False).view(-1)
                        # noise and real cluster labels for this subgroup
                        z_i  = z[idx]
                        k_i  = k[idx]

                        # generate only for cluster i
                        fake_i   = G.generators[i](z_i, None)
                        # if output size mismatches, resize to match real
                        if fake_i.shape[-2:] != real.shape[-2:]:
                            fake_i = F.interpolate(fake_i,
                                                size=real.shape[-2:],
                                                mode='bilinear',
                                                align_corners=False)

                        # adversarial loss for cluster i
                        adv_i     = -D(fake_i).mean()
                        # classification loss for cluster i’s generator outputs
                        feat_i    = D.intermediate(fake_i)
                        logits_i  = C(feat_i)
                        cls_i     = F.cross_entropy(logits_i, k_i)
                        loss_G_i  = adv_i + args.lambda_cls * cls_i

                        # zero, back-prop, step only optimizer for generator i
                        opt_Gs[i].zero_grad()
                        loss_G_i.backward(retain_graph=True)
                        opt_Gs[i].step()

                        # log per-cluster if you like
                        writer.add_scalar(f'Loss/G_cluster_{i}', loss_G_i.item(), global_step)
                        writer.add_scalar(f'Loss/Cls_on_G_cluster_{i}',
                                        cls_i.item(),
                                        global_step)


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
                save_all_models(G, E, D, C, args.log_dir, epoch+1)

        save_all_models(G, E, D, C, args.log_dir, args.epochs)

    except KeyboardInterrupt:
        print("Interrupted—saving models…")
        save_all_models(G, E, D, C, args.log_dir, f"interrupt_{epoch+1}")
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
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=1000)
    p.add_argument("--z_dim",      type=int,   default=128)
    p.add_argument("--n_clusters", type=int,   default=7)
    p.add_argument("--n_critic",   type=int,   default=5)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--lambda_cls", type=float, default=0.1)
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
