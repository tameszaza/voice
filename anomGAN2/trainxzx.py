import os
import sys
import glob
import re
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp


from models import MultiGenerator, MultiEncoder, Discriminator, Classifier, Bandit

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class StackedOnlyDataset(Dataset):
    def __init__(self,
                 real_data_root: str,
                 use_mel: bool = True,
                 use_mfcc: bool = True,
                 cache_in_ram: bool = True):
        if not (use_mel or use_mfcc):
            raise ValueError("Need at least one of use_mel or use_mfcc")

        self.cache = cache_in_ram
        self.data = []

        # scan class folders
        classes = sorted(
            d for d in os.listdir(real_data_root)
            if os.path.isdir(os.path.join(real_data_root, d))
        )
        if not classes:
            raise ValueError(f"No subfolders in {real_data_root}")

        # load each class’s stacked files and concatenate
        all_arrays = []
        for cls in classes:
            base = os.path.join(real_data_root, cls)
            mel_path  = os.path.join(base, "mel",  "stacked.npy") if use_mel  else None
            mfcc_path = os.path.join(base, "mfcc", "stacked.npy") if use_mfcc else None

            if use_mel and not os.path.isfile(mel_path):
                raise FileNotFoundError(f"Missing {mel_path}")
            if use_mfcc and not os.path.isfile(mfcc_path):
                raise FileNotFoundError(f"Missing {mfcc_path}")

            # load both arrays fully (or memmap if you prefer)
            arrs = []
            if use_mel:
                arr_m = np.load(mel_path)   # shape: (N_i, H, W)
                arrs.append(arr_m)
            if use_mfcc:
                arr_f = np.load(mfcc_path)
                arrs.append(arr_f)

            # stack channels → shape (N_i, C, H, W)
            x_cls = np.stack(arrs, axis=1).astype(np.float32)
            all_arrays.append(x_cls)

        # concatenate all classes → shape (N_total, C, H, W)
        full = np.concatenate(all_arrays, axis=0)
        if self.cache:
            # convert once to a single tensor
            self.data = torch.from_numpy(full)
        else:
            # keep numpy array around for on‑the‑fly slicing
            self.data = full

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.cache:
            # just a tensor index → very fast
            return self.data[idx]
        else:
            # numpy slice + torch conversion
            sample = self.data[idx]          # shape (C, H, W)
            return torch.from_numpy(sample)




def load_models(args, device, n_features, H, W):
    G = MultiGenerator(args.z_dim, out_channels=n_features,
                       base_channels=32, img_size=H, n_layers=4,
                       n_clusters=args.n_clusters).to(device)
    E = MultiEncoder(in_channels=n_features, z_dim=args.z_dim,
                     base_channels=32, img_size=H, n_layers=4,
                     n_clusters=args.n_clusters).to(device)
    D = Discriminator(in_channels=n_features,
                      base_channels=32, n_layers=4).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, n_features, H, W, device=device)
        feat  = D.intermediate(dummy)
        c_in  = feat.shape[1]
    C = Classifier(in_channels=c_in,
                   n_clusters=args.n_clusters).to(device)
    B = Bandit(args.n_clusters).to(device)

    sfx = args.ckpt_suffix
    ckpt_dir = args.log_dir
    G.load_state_dict(torch.load(os.path.join(ckpt_dir, f"G_{sfx}.pt"), map_location=device))
    E.load_state_dict(torch.load(os.path.join(ckpt_dir, f"E_{sfx}.pt"), map_location=device))
    D.load_state_dict(torch.load(os.path.join(ckpt_dir, f"D_{sfx}.pt"), map_location=device))
    C.load_state_dict(torch.load(os.path.join(ckpt_dir, f"C_{sfx}.pt"), map_location=device))
    # B.load_state_dict(torch.load(os.path.join(ckpt_dir, f"B_{sfx}.pt"), map_location=device))

    for net in (G, D, C, B):
        for p in net.parameters():
            p.requires_grad = False

    return G, E, D, C, B


def find_latest_interrupt(log_dir):
    pattern = re.compile(r"E_xzx_(\d+)_interrupt\.pt$")
    candidates = glob.glob(os.path.join(log_dir, "E_xzx_*_interrupt.pt"))
    best_epoch = 0
    best_paths = None
    for p in candidates:
        m = pattern.search(os.path.basename(p))
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_paths = (
                    p,
                    os.path.join(log_dir, f"opt_E_xzx_{ep}_interrupt.pt")
                )
    return best_epoch, best_paths


# ------------------ Training Loop ------------------
def train_xzx(args):
    rank       = args.rank
    world_size = args.world_size
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    # set device for this rank
    device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # build your dataset
    ds = StackedOnlyDataset(
        args.real_data_root,
        use_mel=args.use_mel,
        use_mfcc=args.use_mfcc,
        cache_in_ram=args.cache_in_ram
    )


    # exactly like the bug‑free code:
    sampler = DistributedSampler(
        ds,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True
    )
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


    # load pretrained models
    x0 = ds[0]
    n_features, H, W = x0.shape
    G, E, D, C, B = load_models(args, device, n_features, H, W)

    # wrap encoder E for DDP
    E = DDP(E, device_ids=[rank], output_device=rank)

    # optimizer for each sub-encoder
    opt_Es = [
        torch.optim.Adam(enc.parameters(), lr=args.lr, betas=(0.5, 0.9))
        for enc in E.module.encoders
    ]

    # resume logic
    start_epoch = 1
    latest_ep, paths = find_latest_interrupt(args.log_dir)
    if latest_ep > 0 and os.path.exists(paths[1]):
        if rank == 0:
            print(f"Resuming from interrupted epoch {latest_ep}")
        E.module.load_state_dict(torch.load(paths[0], map_location=device))
        start_epoch = latest_ep + 1

    # tensorboard only on rank0
    writer = None
    if rank == 0:
        tb_dir = os.path.join(args.log_dir, "xzx_ddp")
        writer = SummaryWriter(tb_dir)

    step = (start_epoch - 1) * len(loader)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            sampler.set_epoch(epoch)
            running = 0.0
            for x in loader:
                x = x.to(device, non_blocking=True)
                with torch.no_grad():
                    feat_real = D.intermediate(x)
                    k_pred    = C(feat_real).argmax(dim=1)

                # forward
                z_hat = E(x, k_pred)
                x_hat = G(z_hat, k_pred, target_hw=(H, W))

                # per-cluster loss
                losses = []
                for i in torch.unique(k_pred):
                    idx = (k_pred == i).nonzero(as_tuple=True)[0]
                    losses.append(F.mse_loss(x_hat[idx], x[idx]))
                total_e_loss = sum(losses)

                # zero grads
                for opt in opt_Es:
                    opt.zero_grad()
                total_e_loss.backward()
                for opt in opt_Es:
                    opt.step()

                if rank == 0:
                    writer.add_scalar("Loss/E_xzx_ddp", total_e_loss.item(), step)
                running += total_e_loss.item()
                step += 1

            if rank == 0:
                avg = running / len(loader)
                print(f"[xzx-ddp] Epoch {epoch}/{args.epochs}  Loss={avg:.4f}")
              
                if epoch % args.save_every == 0:
                    # save encoder
                    torch.save(
                        E.state_dict(),
                        os.path.join(args.log_dir, f"E_xzx_{epoch}.pt")
                    )
                    # save each sub‐encoder's optimizer state
                    for i, opt in enumerate(opt_Es):
                        torch.save(
                            opt.state_dict(),
                            os.path.join(args.log_dir, f"opt_E_xzx_{i}.pt")
                        )


    except KeyboardInterrupt:
        if rank == 0:
            print(f"Interrupted at epoch {epoch}. Saving encoder and optimizer.")
            torch.save(
                E.module.state_dict(),
                os.path.join(args.log_dir, f"E_xzx_{epoch}_interrupt.pt")
            )
        sys.exit(0)

    # final save
    if rank == 0:
        torch.save(
            E.module.state_dict(),
            os.path.join(args.log_dir, "E_xzx_final.pt")
        )
        writer.close()

# ------------------ DDP Worker ------------------
def ddp_worker(rank, world_size, args):
    import os
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    args.rank = rank
    args.world_size = world_size
    train_xzx(args)                    # call the unified train function
    dist.destroy_process_group()


# ------------------ Main ------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real_data_root", required=True)
    p.add_argument("--log_dir",       required=True)
    p.add_argument("--ckpt_suffix",   required=True)
    p.add_argument("--batch_size",    type=int, default=256)
    p.add_argument("--epochs",        type=int, default=5000)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--save_every",    type=int, default=10)
    p.add_argument("--z_dim",         type=int, default=128)
    p.add_argument("--n_clusters",    type=int, default=7)
    p.add_argument("--use_mel",       action="store_true")
    p.add_argument("--use_mfcc",      action="store_true")
    p.add_argument("--cache_in_ram",  action="store_true")
    p.add_argument("--num_workers",   type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found!")

    mp.spawn(
        ddp_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
