import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

import math
from models import MultiGenerator, MultiEncoder, Discriminator, Classifier
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
def ddp_worker(rank: int, world_size: int, args):
    """
    One process = one GPU.
    rank        : global rank 0 … world_size-1
    world_size  : #GPUs we’re usingtorch.autograd.set_detect_anomaly(True)       
    """
    # ----------- distributed initialisation -----------
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    # each rank sees every GPU, choose its own
    torch.cuda.set_device(rank)
    args.rank = rank                    # pass along to train()
    args.world_size = world_size
    train(args)                         # run the original training loop
    dist.destroy_process_group()



def save_optimizer_states(opt_D, opt_Gs, opt_Es, opt_C, log_dir):
    # overwrite previous optimizer files each step
    torch.save(opt_D.state_dict(), os.path.join(log_dir, "opt_D.pt"))
    for i, opt_G in enumerate(opt_Gs):
        torch.save(opt_G.state_dict(), os.path.join(log_dir, f"opt_G_{i}.pt"))
    for i, opt_E in enumerate(opt_Es):
        torch.save(opt_E.state_dict(), os.path.join(log_dir, f"opt_E_{i}.pt"))
    torch.save(opt_C.state_dict(), os.path.join(log_dir, "opt_C.pt"))

def save_all_models(G, E, D, C, log_dir, suffix, opt_D=None, opt_Gs=None, opt_Es=None, opt_C=None):
    torch.save(G.module.state_dict(), os.path.join(log_dir, f"G_{suffix}.pt"))
    torch.save(E.module.state_dict(), os.path.join(log_dir, f"E_{suffix}.pt"))
    torch.save(D.module.state_dict(), os.path.join(log_dir, f"D_{suffix}.pt"))
    torch.save(C.module.state_dict(), os.path.join(log_dir, f"C_{suffix}.pt"))
    if opt_D is not None:
        torch.save(opt_D.state_dict(), os.path.join(log_dir, f"opt_D.pt"))
    if opt_Gs is not None:
        for i, opt_G in enumerate(opt_Gs):
            torch.save(opt_G.state_dict(), os.path.join(log_dir, f"opt_G_{i}.pt"))
    if opt_Es is not None:
        for i, opt_E in enumerate(opt_Es):
            torch.save(opt_E.state_dict(), os.path.join(log_dir, f"opt_E_{i}.pt"))

    if opt_C is not None:
        torch.save(opt_C.state_dict(), os.path.join(log_dir, f"opt_C.pt"))
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

class MultiFeatureDirectoryDataset(Dataset):
    def __init__(self, data_root, use_mel=True, use_mfcc=True):
        if not (use_mel or use_mfcc):
            raise ValueError("At least one of use_mel or use_mfcc must be True")

        classes = sorted(
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        )
        if not classes:
            raise ValueError(f"No subfolders in {data_root}")

        # accumulate per‑class arrays and labels
        data_parts = []
        label_parts = []

        for label, cls in enumerate(classes):
            base = os.path.join(data_root, cls)

            # paths to the stacked .npy files
            mel_path  = os.path.join(base, "mel",  "stacked.npy") if use_mel  else None
            mfcc_path = os.path.join(base, "mfcc", "stacked.npy") if use_mfcc else None

            if use_mel and not os.path.isfile(mel_path):
                raise FileNotFoundError(f"Missing {mel_path}")
            if use_mfcc and not os.path.isfile(mfcc_path):
                raise FileNotFoundError(f"Missing {mfcc_path}")

            # load full arrays into numpy (one call each)
            arrs = []
            if use_mel:
                arrs.append(np.load(mel_path))    # shape (N, H, W)
            if use_mfcc:
                arrs.append(np.load(mfcc_path))   # shape (N, H, W)

            # stack channels → (N, C, H, W)
            # arrs = [ (N, H, W), ... ]  
            x_cls = np.stack(arrs, axis=1).astype(np.float32)

            data_parts.append(x_cls)              # for concatenation
            label_parts.extend([label] * x_cls.shape[0])

        # concatenate all classes → (N_total, C, H, W)
        full_data = np.concatenate(data_parts, axis=0)
        full_labels = np.array(label_parts, dtype=np.int64)

        # convert once to torch tensors
        self.data   = torch.from_numpy(full_data)    # float32
        self.labels = torch.from_numpy(full_labels)  # int64

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # two tensor indexing ops and you’re done
        return self.data[idx], self.labels[idx]



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
        create_graph=True
    )[0]
    return ((grads.view(B, -1).norm(2, dim=1) - 1)**2).mean()

# -----------------------------------------------------------------------------
#  Training loop
# -----------------------------------------------------------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rank = getattr(args, "rank", 0)
    torch.cuda.set_device(rank)          # already set in ddp_worker, harmless here
    torch.set_num_threads(max(1, args.num_threads // args.num_workers))


    # scale learning rate by batch size relative to 256
    base_batch = 256
    scaled_lr = args.lr * math.sqrt(args.batch_size / base_batch)


    ds = MultiFeatureDirectoryDataset(
        args.data_root, use_mel=args.use_mel, use_mfcc=args.use_mfcc
    )
    sampler = DistributedSampler(ds, num_replicas=args.world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,  
    )


    x0, _ = ds[0]
    n_features, H, W = x0.shape

    # build models
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

    with torch.no_grad():
        dummy = torch.zeros(1, n_features, H, W, device=device)
        feat = D.intermediate(dummy)
    cls_in = feat.shape[1]
    C = Classifier(in_channels=cls_in, n_clusters=args.n_clusters).to(device)

    # wrap with DDP
    G = DDP(G, device_ids=[rank], output_device=rank)
    E = DDP(E, device_ids=[rank], output_device=rank)


    D = DDP(D, device_ids=[rank], output_device=rank)
    C = DDP(C, device_ids=[rank], output_device=rank)

    # optimizers with scaled lr
    opt_D = torch.optim.Adam(D.parameters(), lr=scaled_lr, betas=(0.0, 0.9))
    opt_Gs = [
        torch.optim.Adam(g.parameters(), lr=scaled_lr, betas=(0.0, 0.9))
        for g in G.module.generators
    ]
    opt_Es = [
        torch.optim.Adam(enc.parameters(), lr=scaled_lr, betas=(0.0, 0.9))
        for enc in E.module.encoders
    ]
    opt_C = torch.optim.Adam(C.parameters(), lr=scaled_lr, betas=(0.0, 0.9))

    writer = None
    if rank == 0:
        save_config(args, n_features)
        writer = SummaryWriter(args.log_dir)

    start_epoch = 0
    if args.resume and rank == 0:
        cfg = load_config(os.path.join(args.resume, 'configuration.txt'))
        start_epoch = cfg.get('resume_epoch', 0)
    
    try:
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            global_step = epoch * len(loader)
                # Generator, Encoder, Classifier eve
            for real, k in loader:
                global_step += 1
                real = real.to(device, non_blocking=True)
                k = k.to(device, non_blocking=True)
                z = torch.randn(args.batch_size, args.z_dim, device=device)

                # Discriminator update
                fake = G(z, k, target_hw=real.shape[-2:])
                loss_D = (
                    D(fake.detach()).mean()
                    - D(real).mean()
                    + args.gamma_gp * gradient_penalty(D, real, fake.detach(), device)
                )
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                if writer is not None:
                    writer.add_scalar('Loss/D', loss_D.item(), epoch * len(loader))
                
                if global_step % args.n_critic == 0:
    # ─── Generator update ───
    # freeze C’s BatchNorm so it doesn’t modify its running stats during G’s backward
                    C.eval()
                    for p in C.parameters():
                        p.requires_grad_(False)

                    # zero generator grads
                    for opt_G in opt_Gs:
                        opt_G.zero_grad()

                    total_g = torch.tensor(0., device=device)
                    for i in torch.unique(k):
                        idx     = (k == i).nonzero(as_tuple=True)[0]
                        z_i     = z[idx]
                        fake_i  = G.module.generators[i](z_i, None)
                        if fake_i.shape[-2:] != real.shape[-2:]:
                            fake_i = F.interpolate(fake_i,
                                                size=real.shape[-2:],
                                                mode='bilinear',
                                                align_corners=False)

                        adv_i    = -D(fake_i).mean()
                        feat_i   = D.module.intermediate(fake_i)
                        logits_i = C(feat_i)
                        cls_i    = F.cross_entropy(logits_i, k[idx])
                        z_hat_i  = E(fake_i.detach(), k[idx])
                        lat_i    = F.mse_loss(z_hat_i, z_i)

                        total_g += adv_i - args.lambda_cls * cls_i + args.lambda_lat * lat_i
                        if writer is not None:
                            writer.add_scalar(f'Loss/G_cluster_{i}',    total_g.item(),  global_step)
                            writer.add_scalar(f'Loss/Cls_G_cluster_{i}', cls_i.item(),    global_step)
                            writer.add_scalar(f'Loss/Lat_cluster_{i}',  lat_i.item(),    global_step)

                    total_g.backward()
                    for opt_G in opt_Gs:
                        opt_G.step()

                    # restore C for training
                    C.train()
                    for p in C.parameters():
                        p.requires_grad_(True)

                    # ─── Encoder update ───
                    for opt_E in opt_Es:
                        opt_E.zero_grad()
                    z_hat2 = E(fake.detach(), k)
                    loss_E = F.mse_loss(z_hat2, z.detach())
                    loss_E.backward()
                    for opt_E in opt_Es:
                        opt_E.step()

                    # ─── Classifier update ───
                    opt_C.zero_grad()
                    logits2 = C(D.module.intermediate(fake.detach()))
                    loss_C2 = F.cross_entropy(logits2, k)
                    loss_C2.backward()
                    opt_C.step()
                    if writer is not None:
                        writer.add_scalar('Loss/C', loss_C2.item(), global_step)

            save_optimizer_states(opt_D, opt_Gs, opt_Es, opt_C, args.log_dir)
            if rank == 0 and (epoch + 1) % args.save_every == 0:
                save_all_models(G, E, D, C, args.log_dir, epoch+1,
                                opt_D=opt_D, opt_Gs=opt_Gs, opt_Es=opt_Es, opt_C=opt_C)

        if rank == 0:
            save_all_models(G, E, D, C, args.log_dir, args.epochs)

    except KeyboardInterrupt:
        if rank == 0:
            save_all_models(G, E, D, C, args.log_dir, f"interrupt_{epoch+1}",
                            opt_D=opt_D, opt_Gs=opt_Gs, opt_Es=opt_Es, opt_C=opt_C)
        raise
    finally:
        if writer is not None:
            writer.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # ------------ (all your existing CLI args) ------------
    p.add_argument("--data_root",  required=True)
    p.add_argument("--log_dir",    required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs",     type=int, default=700)
    p.add_argument("--z_dim",      type=int, default=128)
    p.add_argument("--n_clusters", type=int, default=8)
    p.add_argument("--n_critic",   type=int, default=5)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--lambda_cls", type=float, default=0.1)
    p.add_argument("--lambda_lat", type=float, default=1.0)
    p.add_argument("--gamma_gp",   type=float, default=5.0)
    p.add_argument("--use_mel",    action="store_true")
    p.add_argument("--use_mfcc",   action="store_true")
    p.add_argument("--resume",     type=str)
    p.add_argument("--num_workers",  type=int, default=16,
                   help="DataLoader worker processes per GPU")
    p.add_argument("--num_threads",  type=int, default=64,
                   help="Total CPU threads available; will be divided per worker")
    p.add_argument("--save_every",  type=int, default=50,
                   help="Total CPU threads available; will be divided per worker")
    args = p.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    # ------------------------------------------------------
    # decide how many GPUs to use (all visible by default)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found!")

    # spawn <world_size> copies of ddp_worker; each becomes rank 0…N‑1
    mp.spawn(
        ddp_worker,
        args=(world_size, args),   # (rank is inserted automatically)
        nprocs=world_size,
        join=True,
    )
