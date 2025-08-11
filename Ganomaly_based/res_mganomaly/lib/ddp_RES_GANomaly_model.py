from __future__ import annotations
import os, random, numpy as np
from collections import OrderedDict
from typing import Dict, Any
import sys
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm
import torch.distributed as dist



from Resnetworks import (
     NetD_RES_GANomaly,                 # unchanged
     NetG_MultiDecoder_RES_GANomaly,    # <-- new class you just added
     BranchClassifier,                  # <-- new class you just added
     weights_init,
)
from loss import (
    generator_total_loss,
    gradient_penalty,
    discriminator_loss,
    LossWeights,
    classifier_loss_danogan
)
import tempfile

def atomic_save(obj: Any, path: str):
    """
    Write `obj` to *path* atomically:
      1) save to a temporary file in the same directory
      2) os.replace() ➜ guarantees readers never see a half-written file
    """
    dir_, fname = os.path.split(path)
    os.makedirs(dir_, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=dir_, delete=False) as tmp:
        torch.save(obj, tmp.name)
        tmp.flush(); os.fsync(tmp.fileno())       # safety first
        tmp_path = tmp.name

    os.replace(tmp_path, path)                    # atomic on POSIX
    return path

def _seed_everything(seed: int):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False      # ← add this line


# -----------------------------------------------------------------------------
#  Base class (tensorboard, IO helpers)
# -----------------------------------------------------------------------------
class BaseModel:
    def __init__(self, opt, dataloader: Dict[str, torch.utils.data.DataLoader]):
        _seed_everything(getattr(opt, "manualseed", -1))
        self.opt = opt
        self.dataloader = dataloader
        self.device = torch.device("cpu" if opt.device == "cpu" or not torch.cuda.is_available() else opt.device)

        self.input = torch.empty(1, opt.nc, opt.isize, opt.isize, device=self.device)
        self.gt = torch.empty(1, dtype=torch.long, device=self.device)
        self.fixed_input = None

        self.rank = (dist.get_rank()
                     if dist.is_available() and dist.is_initialized() else 0)

        tb_dir = os.path.join(opt.outf, opt.name, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(tb_dir) if self.rank == 0 else None

    def set_input(self, batch):
        """
        Accepts any of the following tuples coming from the DataLoader:

            (x,)                    – purely unsupervised; branch_idx defaults to 0
            (x, k)                  – unsupervised + explicit branch index
            (x, y, k)               – test/val batch with ground-truth labels y
                                    (e.g. bona-/spoof) and branch index k

        * x : FloatTensor (B,C,H,W)
        * y : LongTensor  (B,)       – optional class labels for ROC/ACC
        * k : LongTensor  (B,)       – which decoder should reconstruct this sample
        """
        # ------------------------------------------------------------------ #
        #  0) images X
        # ------------------------------------------------------------------ #
        x = batch[0].to(self.device, non_blocking=True)
        if x.ndim == 3:                           # single image → add batch dim
            x = x.unsqueeze(0)

        self.input.resize_(x.size()).copy_(x)

        # ------------------------------------------------------------------ #
        #  1) default placeholders
        # ------------------------------------------------------------------ #
        B = x.size(0)
        if self.gt.numel() != B:                  # resize once if needed
            self.gt = torch.empty(B, dtype=torch.long, device=self.device)

        self.branch_idx = torch.zeros(
            B, dtype=torch.long, device=self.device)   # default decoder = 0
        self.gt.fill_(-1)                              # “unknown” class

        # ------------------------------------------------------------------ #
        #  2) parse optional fields
        # ------------------------------------------------------------------ #
        if len(batch) == 2:                   # could be (x,k) or (x,y)
            fld = batch[1].to(self.device, non_blocking=True)
            if fld.dtype in (torch.long, torch.int64) and fld.dim() == 1:
                # assume branch indices
                self.branch_idx.copy_(fld)
            else:                             # otherwise treat as labels y
                self.gt.resize_(fld.size()).copy_(fld)

        elif len(batch) >= 3:                 # (x,y,k)
            y = batch[1].to(self.device, non_blocking=True)
            k = batch[2].to(self.device, non_blocking=True)
            self.gt.resize_(y.size()).copy_(y)
            self.branch_idx.copy_(k)

        # ------------------------------------------------------------------ #
        #  3) store a fixed batch for TensorBoard image grids, etc.
        # ------------------------------------------------------------------ #
        if self.fixed_input is None:
            self.fixed_input = self.input.clone()


    def load(self, ckpt_path: str, strict: bool = True):
        """
        Load networks (and optimizers, if present) from a full checkpoint.
        If strict=False, missing/unexpected keys in state_dicts are ignored.
        """
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        # load generator + discriminator weights
        self.netg.load_state_dict(ckpt['netG'], strict=strict)
        self.netd.load_state_dict(ckpt['netD'], strict=strict)
        print(f"[Load] netG/netD weights loaded from {ckpt_path} (strict={strict})")

        # if optimizer states are in the checkpoint, load them too
        if 'branchClf' in ckpt:
            self.branch_clf.load_state_dict(ckpt['branchClf'], strict=strict)

        # ---- load optimisers (names must match save_checkpoint)
        if 'optimG' in ckpt and 'optimD' in ckpt and 'optimClf' in ckpt:
            self.optimizer_g.load_state_dict(ckpt['optimG'])
            self.optimizer_d.load_state_dict(ckpt['optimD'])
            self.optimizer_cls.load_state_dict(ckpt['optimClf'])
            print(f"[Load] optimizer states loaded from {ckpt_path}")

        # ---- resume counters
        self.start_epoch  = ckpt.get('epoch', 0) + 1
        self.epoch        = self.start_epoch
        self.total_steps  = ckpt.get('total_steps', 0)
        print(f"[Load] Will resume from epoch {self.start_epoch}, step {self.total_steps}")


# -----------------------------------------------------------------------------
#  RES_Ganomaly model
# -----------------------------------------------------------------------------
class RES_MGanomaly(BaseModel):
    @property
    def name(self):
        return "RES_Ganomaly"

    def __init__(self, opt, dataloader):
        super().__init__(opt, dataloader)

        if self.rank == 0:
            print(f"[Init] Networks on {self.device}")
        self.netg = NetG_MultiDecoder_RES_GANomaly(opt, opt.n_branches).to(self.device)
        self.netd = NetD_RES_GANomaly(opt).to(self.device)
        # grab the *last* Conv2d inside the discriminator’s feature extractor
        c_feat = next(
            m for m in reversed(self.netd.features_extractor)
            if isinstance(m, nn.Conv2d)
        ).out_channels

        self.branch_clf = BranchClassifier(c_feat, opt.n_branches).to(self.device)
        
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        self.branch_clf.apply(weights_init)
        local_rank = dist.get_rank() if dist.is_initialized() else 0

        # 1) global BatchNorm statistics
        self.netg = SyncBatchNorm.convert_sync_batchnorm(self.netg)
        self.netd = SyncBatchNorm.convert_sync_batchnorm(self.netd)

        # 2) move to the local GPU *before* wrapping
        self.netg.to(opt.device)
        self.netd.to(opt.device)
        self.branch_clf.to(opt.device)

        # 3) wrap with DDP (each network has its own optimiser)
        self.netg = DDP(self.netg, device_ids=[local_rank], output_device=local_rank)
        self.netd = DDP(self.netd, device_ids=[local_rank], output_device=local_rank)
        self.branch_clf = DDP(self.branch_clf, device_ids=[local_rank], output_device=local_rank)
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        beta1 = getattr(opt, "beta1", 0.5)
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=opt.lr_d, betas=(beta1, 0.999))
        self.optimizer_g = optim.Adam(
            list(self.netg.parameters()),
            lr=opt.lr_g, betas=(beta1, 0.999)
        )
        
        self.optimizer_cls = optim.Adam(
            self.branch_clf.parameters(),
            lr=opt.lr_cls,
            betas=(beta1, 0.999)
        )

        self.lambda_gp = getattr(opt, "lambda_gp", 10.0)  # paper’s best λ
        self.n_critic = getattr(opt, "n_critic", 1)      # match Algorithm‑1 exactly
        self.loss_weights = LossWeights(
            w_adv=getattr(opt, "w_adv", 1.0),
            w_con=getattr(opt, "w_con", 50.0),
            w_enc=getattr(opt, "w_enc", 1.0),
            w_cls=getattr(opt, "w_cls", 1.0), 
        )

        self.total_steps = 0
        self.epoch = 0
        
    def _tb_log(self):
        if self.rank != 0:          # <── EARLY EXIT for non-zero ranks
            return
        w = self.writer
        step = self.total_steps

        # discriminator
        w.add_scalar("D/loss_total", self.err_d.item(),       step)
        w.add_scalar("D/real_prob",  self.err_d_real.item(),  step)
        w.add_scalar("D/fake_prob",  self.err_d_fake.item(),  step)
        w.add_scalar("D/gp",         self.gp.item(),          step)

        # generator
        w.add_scalar("G/loss_total", self.err_g.item(),       step)
        w.add_scalar("G/cls",        self.err_g_cls.item(),   step)   # ← new
        w.add_scalar("G/adv",        self.err_g_adv.item(),   step)
        w.add_scalar("G/latent_con", self.err_g_con.item(),   step)
        w.add_scalar("G/recon",      self.err_g_enc.item(),   step)

    def optimise(self):
        # ============================================================== #
        # 1) -----------  D (critic) update ---------------------------- #
        # ============================================================== #
        self.optimizer_d.zero_grad(set_to_none=True)

        #   G → X̂  (but *no* grad to G)
        with torch.no_grad():
            x_recon, _, _ = self.netg(self.input, self.branch_idx)

        d_real_logits, _ = self.netd(self.input)         # logits (B,)
        d_fake_logits, _ = self.netd(x_recon)            # logits (B,)

        gp_val = gradient_penalty(
            self.netd,
            self.input,
            x_recon.detach(),
            self.device,
            self.lambda_gp,
        )

        d_real_prob = torch.sigmoid(d_real_logits)
        d_fake_prob = torch.sigmoid(d_fake_logits)

        self.err_d = discriminator_loss(d_real_prob, d_fake_prob, gp_val)
        self.err_d.backward()
        self.optimizer_d.step()

        # logging helpers
        self.err_d_real = d_real_prob.mean().detach()
        self.err_d_fake = d_fake_prob.mean().detach()
        self.gp         = gp_val.detach()

        # ============================================================== #
        # 2) -----------  G+E  +  auxiliary classifier update ---------- #
        # ============================================================== #
        if self.total_steps % self.n_critic == 0:
            self.optimizer_g.zero_grad(set_to_none=True)
            self.optimizer_cls.zero_grad(set_to_none=True)

            x_recon, z_i, z_o = self.netg(self.input, self.branch_idx)
            d_fake_logits_G, feats_fake = self.netd(x_recon)

            # auxiliary classifier
            logits_cls   = self.branch_clf(feats_fake)
            self.err_g_cls = classifier_loss_danogan(logits_cls, self.branch_idx)

            # ---- reuse D(x) from the D-step (detach to keep graph small)
            d_real_det = d_real_logits.detach()

            err_g_base, self.err_g_adv, self.err_g_con, self.err_g_enc = \
                generator_total_loss(
                    d_real_det, d_fake_logits_G,
                    self.input, x_recon, z_i, z_o,
                    self.loss_weights
                )

            self.err_g = err_g_base + self.err_g_cls * self.loss_weights.w_cls
            self.err_g.backward()

            # optional: clip classifier gradients (helps stability)
            torch.nn.utils.clip_grad_norm_(self.branch_clf.parameters(), max_norm=5.0)

            self.optimizer_g.step()
            self.optimizer_cls.step()

        else:
            # when skipping G update (WGAN-GP training)
            zero = torch.zeros(1, device=self.device)
            self.err_g      = zero
            self.err_g_adv  = zero
            self.err_g_con  = zero
            self.err_g_enc  = zero
            self.err_g_cls  = zero
        
    # ─────────────────────────────────────────────────────────────────────
    #  SAVE / LOAD HELPERS  (instance methods)
    # ─────────────────────────────────────────────────────────────────────
    def _ckpt_dir(self) -> str:
        d = os.path.join(self.opt.outf, self.opt.name, "checkpoints")
        os.makedirs(d, exist_ok=True)
        return d

    def save_light(self, tag: str):
        """
        Weights-only snapshot (fits on disk, quick to load).
        """
        if self.rank != 0:                           # ← guard
            return
        path = os.path.join(self._ckpt_dir(), f"weights_{tag}.pth")
        ckpt = {
            "epoch":     self.epoch,
            "netG":      self.netg.state_dict(),
            "netD":      self.netd.state_dict(),
            "branchClf": self.branch_clf.state_dict(),
        }
        atomic_save(ckpt, path)
        print(f"[✓] Light weights snapshot  →  {path}")

    def save_checkpoint(self, tag: str):
        """
        Full, resumable checkpoint – all nets, all optimisers, opt, counters.
        """
        if self.rank != 0:                           # ← guard
            return
        path = os.path.join(self._ckpt_dir(), f"checkpoint_{tag}.pth")
        ckpt = {
            "epoch":        self.epoch,
            "total_steps":  self.total_steps,
            "netG":         self.netg.state_dict(),
            "netD":         self.netd.state_dict(),
            "branchClf":    self.branch_clf.state_dict(),
            "optimG":       self.optimizer_g.state_dict(),
            "optimD":       self.optimizer_d.state_dict(),
            "optimClf":     self.optimizer_cls.state_dict(),
            "opt":          dict(vars(self.opt)),          # save CLI flags
        }
        atomic_save(ckpt, path)
        print(f"[✓] Full checkpoint saved   →  {path}")

    def train_periodic_save(self):
        """
        One unified training loop (no separate train_one_epoch required).
        • Runs for opt.niter epochs.
        • Accumulates epoch-level metrics.
        • Every 10 epochs: prints debug stats and writes a *light* snapshot.
        • On Ctrl-C: writes a *full* checkpoint before exiting.
        """
        try:
            for epoch in range(self.epoch, self.opt.niter):
                self.epoch = epoch
                self.netg.train(); self.netd.train()
                if isinstance(self.dataloader["train"].sampler,
                                torch.utils.data.distributed.DistributedSampler):
                    self.dataloader["train"].sampler.set_epoch(epoch)
                # ─── reset epoch accumulators ────────────────────────────
                d_loss_tot = d_real_tot = d_fake_tot = gp_tot = 0.0
                g_loss_tot = g_adv_tot = g_con_tot = g_enc_tot = g_cls_tot = 0.0
                num_batches = 0

                for batch in tqdm(
                    self.dataloader["train"],
                    leave=False,
                    desc=f"Epoch {epoch+1}/{self.opt.niter}"
                ):
                    num_batches += 1
                    self.total_steps += 1

                    # ---- forward + optimisers --------------------------
                    self.set_input(batch)
                    self.optimise()

                    # ---- accumulate losses ----------------------------
                    d_loss_tot  += self.err_d.item()
                    d_real_tot  += self.err_d_real.item()
                    d_fake_tot  += self.err_d_fake.item()
                    gp_tot      += self.gp.item()

                    g_loss_tot  += self.err_g.item()
                    g_adv_tot   += self.err_g_adv.item()
                    g_con_tot   += self.err_g_con.item()
                    g_enc_tot   += self.err_g_enc.item()
                    g_cls_tot   += self.err_g_cls.item() 

                    # ---- TensorBoard (optional) ------------------------
                    if self.total_steps % self.opt.tb_freq == 0:
                        if hasattr(self, "_tb_log"):
                            self._tb_log()

                # ─── epoch-level averages ───────────────────────────────
                self.epoch_metrics = {
                    "d_total": d_loss_tot / num_batches,
                    "d_real":  d_real_tot / num_batches,
                    "d_fake":  d_fake_tot / num_batches,
                    "gp":      gp_tot     / num_batches,
                    "g_total": g_loss_tot / num_batches,
                    "g_adv":   g_adv_tot  / num_batches,
                    "g_con":   g_con_tot  / num_batches,
                    "g_enc":   g_enc_tot  / num_batches,
                    "g_cls":   g_cls_tot  / num_batches, 
                }

                # ─── periodic debug & light save every 10 epochs ────────
                if self.rank == 0 and (epoch + 1) % 10 == 0:
                    m = self.epoch_metrics
                    print(
                        f"[Debug] Epoch {epoch+1:3d} — "
                        f"D_loss {m['d_total']:.4f} | D_real {m['d_real']:.4f} "
                        f"| D_fake {m['d_fake']:.4f} | GP {m['gp']:.4f}"
                    )
                    print(
                        f"[Debug] Epoch {epoch+1:3d} — "
                        f"G_loss {m['g_total']:.4f} | G_cls {m['g_cls']:.4f} "
                        f"| G_adv {m['g_adv']:.4f} | G_con {m['g_con']:.4f} | G_enc {m['g_enc']:.4f}"
                    )
                    self.save_light(f"epoch{epoch+1}")


        except KeyboardInterrupt:
            if self.rank == 0:
                tag = f"interrupt_epoch{self.epoch+1}"
                print(f"\n[Interrupt] CTRL+C — writing full checkpoint '{tag}' …")
                self.save_checkpoint(tag)
            sys.exit(0)

        finally:
            if self.rank == 0 and self.writer is not None:
                self.writer.close()



