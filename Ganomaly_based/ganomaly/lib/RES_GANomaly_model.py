from __future__ import annotations
import os, random, numpy as np
from collections import OrderedDict
from typing import Dict
import sys
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Resnetworks import NetG_RES_GANomaly, NetD_RES_GANomaly, weights_init
from loss import (
    generator_total_loss,
    gradient_penalty,
    discriminator_loss,
    LossWeights,
)
from evaluate import evaluate

# -----------------------------------------------------------------------------
#  helpers
# -----------------------------------------------------------------------------
# ─────────────────────────────────────────────────────────────
#  (put this anywhere near the top of the file, after imports)
# ─────────────────────────────────────────────────────────────
def _load_if_given(model: nn.Module, ckpt_path: str | None, *, name: str):
    """
    If `ckpt_path` is not None/empty and the file exists, load it into `model`.
    Will ignore missing keys so that you can load weights from a slightly
    different experiment without crashing.
    """
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[Init] Loaded {name} from '{ckpt_path}'. "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    elif ckpt_path:
        print(f"[Init] WARNING – '{ckpt_path}' not found, starting {name} from scratch.")


def _seed_everything(seed: int):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

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
        x = batch[0].to(self.device, non_blocking=True)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        self.input.resize_(x.size()).copy_(x)
        if len(batch) > 1 and batch[1] is not None:
            y = batch[1].to(self.device, non_blocking=True)
            self.gt.resize_(y.size()).copy_(y)
        if self.fixed_input is None:
            self.fixed_input = self.input.clone()

# -----------------------------------------------------------------------------
#  RES_Ganomaly model
# -----------------------------------------------------------------------------
class RES_Ganomaly(BaseModel):
    @property
    def name(self):
        return "RES_Ganomaly"

    def __init__(self, opt, dataloader):
        super().__init__(opt, dataloader)

        print(f"[Init] Networks on {self.device}")
        self.netg = NetG_RES_GANomaly(opt).to(self.device)
        self.netd = NetD_RES_GANomaly(opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        
        _load_if_given(self.netg, getattr(opt, "netg_ckpt", None), name="netG")
        _load_if_given(self.netd, getattr(opt, "netd_ckpt", None), name="netD")
        
        beta1 = getattr(opt, "beta1", 0.5)
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=opt.lr, betas=(beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=opt.lr, betas=(beta1, 0.999))

        self.lambda_gp = getattr(opt, "lambda_gp", 1.0)  # paper’s best λ
        self.n_critic = getattr(opt, "n_critic", 1)      # match Algorithm‑1 exactly
        self.loss_weights = LossWeights(
            w_adv=getattr(opt, "w_adv", 1.0),
            w_con=getattr(opt, "w_con", 50.0),
            w_enc=getattr(opt, "w_enc", 1.0),
        )

        self.total_steps = 0
        self.epoch = 0

    # --------------------------------------------------------------
    #  optimisation per batch
    # --------------------------------------------------------------
    def optimise(self):
        # ==============================================================
        #   1) -----------  D   (critic)   update  ----------------------
        # ==============================================================

        self.optimizer_d.zero_grad(set_to_none=True)

        # ----- first forward through G (used ONLY for D) ---------------
        with torch.no_grad():
            x_recon, _, _ = self.netg(self.input)

        d_real_logits, _ = self.netd(self.input)             # logits!!
        d_fake_logits, _ = self.netd(x_recon)       # stop grads to G

        gp_val = gradient_penalty(
            self.netd, x_recon.detach(), self.device, self.lambda_gp
        )

        d_real_prob = torch.sigmoid(d_real_logits)
        d_fake_prob = torch.sigmoid(d_fake_logits)

        self.err_d = discriminator_loss(d_real_prob, d_fake_prob, gp_val)
        self.err_d.backward()                                # single backward
        self.optimizer_d.step()

        # stats for TensorBoard
        self.err_d_real = d_real_prob.mean().detach()
        self.err_d_fake = d_fake_prob.mean().detach()
        self.gp         = gp_val.detach()

        # ==============================================================
        #   2) -----------  G   (generator) update  ---------------------
        # ==============================================================

        if self.total_steps % self.n_critic == 0:
            self.optimizer_g.zero_grad(set_to_none=True)

            # fresh forward through G  (new graph)
            x_recon, z_i, z_o = self.netg(self.input)
            d_fake_logits_G, _ = self.netd(x_recon)

            # we only need σ(D(x)) as a constant tensor (no grad) for L_adv
            with torch.no_grad():
                d_real_logits_fixed, _ = self.netd(self.input)

            (
                self.err_g,
                self.err_g_adv,
                self.err_g_con,
                self.err_g_enc,
            ) = generator_total_loss(
                d_real_logits_fixed,      # detached – no gradients flow into D
                d_fake_logits_G,          # keeps grad wrt G
                self.input,
                x_recon,
                z_i,
                z_o,
                self.loss_weights,
            )
            self.err_g.backward()
            self.optimizer_g.step()
        else:
            zero = torch.zeros(1, device=self.device)
            self.err_g = self.err_g_adv = self.err_g_con = self.err_g_enc = zero
    # --------------------------------------------------------------
    #  train epoch
    # --------------------------------------------------------------
    def train_one_epoch(self):
        self.netg.train(); self.netd.train()

        # ← initialize accumulators
        d_losses_total, d_losses_real, d_losses_fake, gp_values = [], [], [], []
        g_losses_total, g_losses_adv, g_losses_con, g_losses_enc = [], [], [], []

        for batch in tqdm(self.dataloader["train"], leave=False, desc=f"Epoch {self.epoch+1}"):
            self.total_steps += 1
            self.set_input(batch)
            self.optimise()

            # ← accumulate
            d_losses_total.append(self.err_d.item())
            d_losses_real.append(self.err_d_real.item())
            d_losses_fake.append(self.err_d_fake.item())
            gp_values.append(self.gp.item())

            g_losses_total.append(self.err_g.item())
            g_losses_adv.append(self.err_g_adv.item())
            g_losses_con.append(self.err_g_con.item())
            g_losses_enc.append(self.err_g_enc.item())

            if self.total_steps % self.opt.tb_freq == 0:
                self._tb_log()

        # ← store epoch‐level averages for later debugging
        self.epoch_metrics = {
            "d_total": np.mean(d_losses_total),
            "d_real":  np.mean(d_losses_real),
            "d_fake":  np.mean(d_losses_fake),
            "gp":      np.mean(gp_values),
            "g_total": np.mean(g_losses_total),
            "g_adv":   np.mean(g_losses_adv),
            "g_con":   np.mean(g_losses_con),
            "g_enc":   np.mean(g_losses_enc),
        }

    def _tb_log(self):
        w = self.writer; s = self.total_steps
        w.add_scalar("D/loss_total", self.err_d.item(), s)
        w.add_scalar("D/real_prob", self.err_d_real.item(), s)
        w.add_scalar("D/fake_prob", self.err_d_fake.item(), s)
        w.add_scalar("D/gp", self.gp.item(), s)
        w.add_scalar("G/loss_total", self.err_g.item(), s)
        w.add_scalar("G/adv", self.err_g_adv.item(), s)
        w.add_scalar("G/latent_con", self.err_g_con.item(), s)
        w.add_scalar("G/recon", self.err_g_enc.item(), s)

    # --------------------------------------------------------------
    #  training driver
    # --------------------------------------------------------------
    def train(self):
        best_auc = 0.0
        for self.epoch in range(self.opt.niter):
            self.train_one_epoch()

            # ← debug print every 10 epochs
            if (self.epoch + 1) % 10 == 0:
                m = self.epoch_metrics
                print(f"[Debug] Epoch {self.epoch+1:3d} — D_loss: {m['d_total']:.4f}, D_real: {m['d_real']:.4f}, "
                      f"D_fake: {m['d_fake']:.4f}, GP: {m['gp']:.4f}")
                print(f"[Debug] Epoch {self.epoch+1:3d} — G_loss: {m['g_total']:.4f}, G_adv: {m['g_adv']:.4f}, "
                      f"G_con: {m['g_con']:.4f}, G_enc: {m['g_enc']:.4f}")

            auc = self.evaluate_current_model()
            if auc > best_auc:
                best_auc = auc
                self.save("best")
            print(f"[Epoch {self.epoch+1}] AUC={auc:.4f} best={best_auc:.4f}")

        self.writer.close()
        
        
    def train_periodic_save(self):
        """
        A simpler training loop: run for opt.niter epochs,
        call train_one_epoch each time, and every 10 epochs:
          1) print debug stats for all losses (networks only)
          2) save generator+discriminator via save_light()
        On CTRL+C, do a full save (including optimizers) at the current epoch.
        """
        try:
            for epoch in range(self.opt.niter):
                self.epoch = epoch
                self.train_one_epoch()

                # every 10 epochs (1-based count)
                if (epoch + 1) % 10 == 0:
                    m = self.epoch_metrics
                    print(
                        f"[Debug] Epoch {epoch+1:3d} — "
                        f"D_loss: {m['d_total']:.4f}, D_real: {m['d_real']:.4f}, "
                        f"D_fake: {m['d_fake']:.4f}, GP: {m['gp']:.4f}"
                    )
                    print(
                        f"[Debug] Epoch {epoch+1:3d} — "
                        f"G_loss: {m['g_total']:.4f}, G_adv: {m['g_adv']:.4f}, "
                        f"G_con: {m['g_con']:.4f}, G_enc: {m['g_enc']:.4f}"
                    )
                    tag = f"epoch{epoch+1}"
                    self.save_light(tag)
                    print(f"[Periodic Save] Epoch {epoch+1:3d} — saved light ckpt '{tag}'")

        except KeyboardInterrupt:
            # On CTRL+C, save full checkpoint (nets + optimizers) at current epoch
            print(f"\n[Interrupt] CTRL+C caught — saving full checkpoint at epoch {self.epoch+1}")
            self.save(f"interrupt_epoch{self.epoch+1}")
            sys.exit(0)

        finally:
            # Always close the TensorBoard writer
            self.writer.close()
    # --------------------------------------------------------------
    #  evaluation – ROC‑AUC
    # --------------------------------------------------------------
    def evaluate_current_model(self):
        self.netg.eval(); scores, labels = [], []
        with torch.no_grad():
            for batch in self.dataloader["test"]:
                self.set_input(batch)
                _, z, z_p = self.netg(self.input)
                if z.ndim == 4:
                    z, z_p = z.squeeze(-1).squeeze(-1), z_p.squeeze(-1).squeeze(-1)
                scores.append(torch.norm(z - z_p, p=2, dim=1).cpu())
                labels.append(self.gt.cpu())
        auc = evaluate(torch.cat(labels), torch.cat(scores), metric=self.opt.metric)
        self.writer.add_scalar("Val/AUC", auc, self.epoch+1)
        return auc

    # --------------------------------------------------------------
    #  checkpoint helper
    # --------------------------------------------------------------
    def save(self, tag="latest"):
        ckpt_dir = os.path.join(self.opt.outf, self.opt.name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        checkpoint = {
            'epoch':      self.epoch,
            'netG':       self.netg.state_dict(),
            'netD':       self.netd.state_dict(),
            'optimG':     self.optimizer_g.state_dict(),
            'optimD':     self.optimizer_d.state_dict(),
        }
        path = os.path.join(ckpt_dir, f"checkpoint_{tag}.pth")
        torch.save(checkpoint, path)
    
    def save_light(self, tag="latest_light"):
        """ Save only epoch and network weights (no optimizer states). """
        ckpt_dir = os.path.join(self.opt.outf, self.opt.name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'netG':  self.netg.state_dict(),
            'netD':  self.netd.state_dict(),
        }
        path = os.path.join(ckpt_dir, f"checkpoint_{tag}.pth")
        torch.save(checkpoint, path)
    

