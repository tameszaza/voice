from __future__ import annotations
import os, random, numpy as np
from collections import OrderedDict
from typing import Dict, Any
import sys
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from Resnetworks import (
     NetD_RES_GANomaly,                 # unchanged
     NetG_MultiDecoder_RES_GANomaly,    # <-- new class you just added
     BranchClassifier,                  # <-- new class you just added
     weights_init,
)
from loss import (
    generator_total_loss,
    adversarial_loss_ganomaly,
    latent_consistency_loss,
    reconstruction_loss,
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

        tb_dir = os.path.join(opt.outf, opt.name, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)

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
        # default: unknown label & random branch per sample
        if self.gt.numel() != B:
            self.gt = torch.empty(B, dtype=torch.long, device=self.device)

        self.gt.fill_(-1)

        # sample k ~ Uniform{0..K-1}
        self.branch_idx = torch.randint(
            low=0, high=self.netg.n_branches, size=(B,), device=self.device
        )

        # If the dataloader provided a branch index, respect it (overwrite the random one)
        if len(batch) == 2:
            fld = batch[1].to(self.device, non_blocking=True)
            if fld.dtype in (torch.long, torch.int64) and fld.dim() == 1:
                self.branch_idx.copy_(fld)
            else:
                self.gt.resize_(fld.size()).copy_(fld)
        elif len(batch) >= 3:
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
        if 'optimG_enc' in ckpt and 'optimG_dec' in ckpt and 'optimD' in ckpt and 'optimClf' in ckpt:
            self.opt_g_enc.load_state_dict(ckpt['optimG_enc'])
            # rebuild decoder optimizers if shape changed; otherwise just load:
            for opt, state in zip(self.opt_g_dec, ckpt['optimG_dec']):
                opt.load_state_dict(state)
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
        
        beta1 = getattr(opt, "beta1", 0.5)
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=opt.lr_d, betas=(beta1, 0.999))
        self.opt_g_enc = optim.Adam(self.netg.encoder.parameters(), lr=opt.lr_g, betas=(beta1, 0.999))
        self.opt_g_dec = [
            optim.Adam(dec.parameters(), lr=opt.lr_g, betas=(beta1, 0.999))
            for dec in self.netg.decoders
        ]
                
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
        self.branch_counts = torch.zeros(opt.n_branches, dtype=torch.long)
        self.tb_branch_freq = getattr(opt, "tb_branch_freq", self.opt.tb_freq * 5)  # log less often

        
    def _tb_log(self):
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
    @torch.no_grad()
    def _tb_log_branches(self):
        """Evaluate each decoder on the same fixed batch; log per-branch losses."""
        if self.fixed_input is None:
            return

        # preserve modes
        g_mode, d_mode = self.netg.training, self.netd.training
        self.netg.eval(); self.netd.eval()

        x = self.fixed_input
        z = self.netg.encoder(x)  # shared for speed
        d_real, _ = self.netd(x)

        for k, dec in enumerate(self.netg.decoders):
            x_hat  = dec(z)
            z_hat  = self.netg.encoder(x_hat)

            #d_real, _  = self.netd(x)       # raw logits (eval mode)
            d_fake, f  = self.netd(x_hat)

            l_adv = adversarial_loss_ganomaly(d_real, d_fake).item()
            l_con = latent_consistency_loss(z, z_hat).item()
            l_enc = reconstruction_loss(x, x_hat).item()

            self.writer.add_scalar(f"branch/k{k}/adv_fixed", l_adv, self.total_steps)
            self.writer.add_scalar(f"branch/k{k}/con_fixed", l_con, self.total_steps)
            self.writer.add_scalar(f"branch/k{k}/enc_fixed", l_enc, self.total_steps)

            if self.loss_weights.w_cls > 0:
                logits = self.branch_clf(f)
                target = torch.full((x.size(0),), k, dtype=torch.long, device=logits.device)
                cls_loss = classifier_loss_danogan(logits, target).item()
                acc = (logits.argmax(1) == target).float().mean().item()
                self.writer.add_scalar(f"branch/k{k}/cls_loss_fixed", cls_loss, self.total_steps)
                self.writer.add_scalar(f"branch/k{k}/cls_acc_fixed",  acc,      self.total_steps)

        # usage share (normalized counts)
        total = int(self.branch_counts.sum().item())
        if total > 0:
            for k in range(self.netg.n_branches):
                share = float(self.branch_counts[k].item()) / total
                self.writer.add_scalar(f"branch/usage_share/k{k}", share, self.total_steps)

        # restore modes
        self.netg.train(g_mode); self.netd.train(d_mode)

    def optimise(self):
        # ==========================
        # 1) --- D / Critic update
        # ==========================
        with torch.no_grad():
            bc = torch.bincount(self.branch_idx, minlength=self.netg.n_branches)
        self.branch_counts += bc.cpu()
        self.optimizer_d.zero_grad(set_to_none=True)

        with torch.no_grad():
            x_recon, _, _ = self.netg(self.input, self.branch_idx)

        d_real_logits, _ = self.netd(self.input)     # RAW logits
        d_fake_logits, _ = self.netd(x_recon)        # RAW logits

        gp_val = gradient_penalty(self.netd, self.input, x_recon.detach(),
                                self.device, self.lambda_gp)
        # Eq. (13):  E[D(x′)] − E[log D(x)] + λ·GP
        #d_real_prob = torch.sigmoid(d_real_logits)
        d_real_logprob = -F.softplus(-d_real_logits).mean() 
        #self.err_d = d_fake_logits.mean() - torch.log(d_real_prob + 1e-12).mean() + gp_val
        self.err_d = d_fake_logits.mean() - d_real_logprob + gp_val
        # WGAN-GP loss (raw logits)
        #self.err_d = discriminator_loss(d_real_logits, d_fake_logits, gp_val)
        self.err_d.backward()
        self.optimizer_d.step()

        # logging (nice to keep)
        self.err_d_real = torch.sigmoid(d_real_logits).mean().detach()
        self.err_d_fake = torch.sigmoid(d_fake_logits).mean().detach()
        self.gp         = gp_val.detach()

        # ==========================
        # 2) --- G + Classifier  (branch-wise micro-batching)
        # ==========================
        if self.total_steps % self.n_critic == 0:
            # freeze D weights; keep graph through D for G grads
            for p in self.netd.parameters():
                p.requires_grad_(False)

            # zero grads for encoder, decoders, and classifier
            self.opt_g_enc.zero_grad(set_to_none=True)
            for opt_k in self.opt_g_dec:
                opt_k.zero_grad(set_to_none=True)
            self.optimizer_cls.zero_grad(set_to_none=True)

            B = self.input.size(0)
            unique_k = torch.unique(self.branch_idx).tolist()
            last_k = unique_k[-1]

            total_g = 0.0  # for logging

            for k in unique_k:
                mask = (self.branch_idx == k)
                x_k  = self.input[mask]                 # (b_k, C, H, W)

                # forward ONLY this subset through G(k) and D
                z_i_k   = self.netg.encoder(x_k)
                x_hat_k = self.netg.decoders[k](z_i_k)
                z_o_k   = self.netg.encoder(x_hat_k)

                with torch.no_grad():
                    d_real_k, _ = self.netd(x_k)        # raw logits for real subset
                d_fake_k, f_k = self.netd(x_hat_k)      # raw logits + features for cls

                # classifier loss on this subset
                cls_k = 0.0
                if self.loss_weights.w_cls > 0:
                    logits_k = self.branch_clf(f_k)
                    target_k = torch.full((x_k.size(0),), k, dtype=torch.long, device=x_k.device)
                    cls_k    = classifier_loss_danogan(logits_k, target_k)

                # your GANomaly-style losses on this subset
                adv_k = adversarial_loss_ganomaly(d_real_k, d_fake_k)
                con_k = latent_consistency_loss(z_i_k, z_o_k)
                enc_k = reconstruction_loss(x_k, x_hat_k)

                # weight by subset fraction so the total equals batch mean
                g_k = (self.loss_weights.w_adv * adv_k
                    + self.loss_weights.w_con * con_k
                    + self.loss_weights.w_enc * enc_k
                    + self.loss_weights.w_cls * cls_k) * (x_k.size(0) / B)

                # backprop immediately to free this subset’s graph
                g_k.backward(retain_graph=(k != last_k))
                total_g += float(g_k.detach().cpu())

                # (optional) free some tensors early
                del z_i_k, x_hat_k, z_o_k, d_real_k, d_fake_k, f_k

            # step once after accumulating grads from all subsets
            self.opt_g_enc.step()
            for k in unique_k:
                self.opt_g_dec[k].step()
            self.optimizer_cls.step()

            # unfreeze D for next critic step
            for p in self.netd.parameters():
                p.requires_grad_(True)

            # for logs
            self.err_g = torch.tensor(total_g, device=self.device)

        else:
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
        path = os.path.join(self._ckpt_dir(), f"checkpoint_{tag}.pth")
        ckpt = {
            "epoch":        self.epoch,
            "total_steps":  self.total_steps,
            "netG":         self.netg.state_dict(),
            "netD":         self.netd.state_dict(),
            "branchClf":    self.branch_clf.state_dict(),
            "optimG_enc":   self.opt_g_enc.state_dict(),
            "optimG_dec":   [opt.state_dict() for opt in self.opt_g_dec],
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

                    # log per-branch summaries less frequently to keep it cheap
                    if self.total_steps % self.tb_branch_freq == 0:
                        if hasattr(self, "_tb_log_branches"):
                            self._tb_log_branches()

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
                if (epoch + 1) % 10 == 0:
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
            tag = f"interrupt_epoch{self.epoch+1}"
            print(f"\n[Interrupt] CTRL+C — writing full checkpoint '{tag}' …")
            self.save_checkpoint(tag)
            sys.exit(0)

        finally:
            self.writer.close()



