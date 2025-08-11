# ddp_utils.py
import os, torch, torch.distributed as dist

def ddp_setup(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialise a single-node multi-GPU process group and pin the
    current process to its local GPU.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
