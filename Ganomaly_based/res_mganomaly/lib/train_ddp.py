# train_ddp.py
import torch
import torch.multiprocessing as mp
from ddp_utils import ddp_setup

# ←——— reuse your existing parser verbatim ———→
from train_res_no_eval import build_option_parser, ClusteredLogMelDataset
from ddp_RES_GANomaly_model import RES_MGanomaly
from torch.utils.data import DataLoader, DistributedSampler

def _run(rank: int, world_size: int, cli_args):
    ddp_setup(rank, world_size)

    # 1) fix device & visible GPU for this worker
    cli_args.device = f"cuda:{rank}"

    # 2) data — use DistributedSampler
    train_ds = ClusteredLogMelDataset(cli_args.data_path)
    if cli_args.n_branches is None:
        cli_args.n_branches = int(train_ds.labels.max()) + 1
    sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=cli_args.batchsize,
        sampler=sampler,
        num_workers=cli_args.num_workers if hasattr(cli_args, "num_workers") else 8,
        pin_memory=True,
    )
    dataloader = {"train": train_loader}

    # 3) build model *on this rank’s GPU*
    model = RES_MGanomaly(cli_args, dataloader)

    # 4) make sure each epoch is shuffled differently
    for epoch in range(cli_args.niter):
        sampler.set_epoch(epoch)
        model.train_periodic_save()          # unchanged

    torch.distributed.destroy_process_group()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    opt = build_option_parser().parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(_run, args=(world_size, opt), nprocs=world_size)
