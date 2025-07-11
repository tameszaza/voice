#!/usr/bin/env python
"""
model.py — single-decoder GANomaly (BCE-with-logits, fully portable)

• Aligned with *train.py* (07 Jul 2025) — expects dataloaders={"train":[loader]}.
• Epochs driven by opt.niter; light ckpt every opt.save_freq; TB scalars every opt.tb_freq.
• Imports NetG / NetD and loss helpers from your own networks.py / loss.py.
"""

from __future__ import annotations

import math, random
from pathlib import Path
from typing import Any, Dict, List
from tqdm.auto import tqdm 
import subprocess 

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import io

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def _seed_everything(seed: int | None):
    """Deterministic torch / numpy / std-lib RNG."""
    if seed is None:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _atomic_save(obj: Dict[str, Any], path: Path):
    """Write checkpoint atomically <tmp> → <path>."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
#  BaseModel
# ─────────────────────────────────────────────────────────────────────────────


class BaseModel:
    """Common utilities (seed, TB writer, (de)checkpoint, …)."""

    def __init__(self, opt, dataloaders: Dict[str, List[torch.utils.data.DataLoader]]):
        _seed_everything(opt.manualseed)

        self.opt = opt
        self.device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
        self.dataloaders = dataloaders  # {"train": [loader]}
        self.total_steps = 0
        self.epoch = 1
        self.remote_name = "Depression" 

        tb_dir = Path(opt.outf) / opt.name / "tensorboard" / "dec1"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        self.ckpt_dir = Path(opt.outf) / opt.name / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # checkpoint helpers -------------------------------------------------

    def _optim_state_to(self, device: str):
        """Move optimiser state tensors to given device in-place."""
        for o in self.optims():
            for st in o.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
    def _stream_to_drive(self, obj: Dict[str, Any], filename: str):
        """
        Serialize `obj` in-memory and stream it into
        Depression:<opt.outf>/<opt.name>/checkpoints/<filename>
        via `rclone rcat`.
        """
        # 1) serialize
        print("enter fn")
        buf = io.BytesIO()
        torch.save(obj, buf)
        buf.seek(0)

        # 2) build the remote path
        #    e.g. "Depression:output_tradgan_from_local/128singleFakeWavefake/checkpoints"
        remote_subdir = Path(self.opt.outf).joinpath(self.opt.name, "checkpoints")
        # strip leading "./" if any, ensure posix style
        rel = remote_subdir.as_posix().lstrip("./")
        remote_path = f"{self.remote_name}:{rel}/{filename}"
        # 3) stream it up
        subprocess.run(
            ["rclone", "rcat", remote_path],
            input=buf.getvalue(),
            check=True
        )
        print(f"[✓] streamed → {remote_path}")

    def save(self, tag: str):
        """Full ckpt (nets + opts), streamed directly to Drive."""
        # move → CPU, package dict, move back
        self._optim_state_to("cpu")
        obj = {
            "epoch": self.epoch,
            "iter": self.total_steps,
            "netg": self.netg.state_dict(),
            "netd": self.netd.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "opt": vars(self.opt),
        }
        self._optim_state_to(self.device.type)

        filename = f"{tag}.pth"
        self._stream_to_drive(obj, filename)

    def save_light(self, tag: str):
        """Light ckpt (just weights), streamed directly to Drive."""
        obj = {
            "epoch": self.epoch,
            "iter": self.total_steps,
            "netg": self.netg.state_dict(),
            "netd": self.netd.state_dict(),
        }
        filename = f"{tag}_light.pth"
        self._stream_to_drive(obj, filename)

    def load(self, path: str | Path, strict: bool = True):
        """Resume training or eval from <path>."""
        ckpt = torch.load(path, map_location="cpu")
        self.netg.load_state_dict(ckpt["netg"], strict)
        self.netd.load_state_dict(ckpt["netd"], strict)
        if "opt_g" in ckpt:
            self.opt_g.load_state_dict(ckpt["opt_g"])
            self.opt_d.load_state_dict(ckpt["opt_d"])
            self._optim_state_to(self.device.type)
            print("[✓] optimisers restored")
        self.epoch = ckpt.get("epoch", 1)
        self.total_steps = ckpt.get("iter", 0)
        print(f"[✓] resumed from {path} (epoch {self.epoch})")

    # must be implemented by subclass -----------------------------------
    def optims(self):
        raise NotImplementedError

    # TensorBoard helper -------------------------------------------------
    def _tb_log(self, losses: Dict[str, float]):
        for k, v in losses.items():
            self.writer.add_scalar(k, v, self.total_steps)


# ─────────────────────────────────────────────────────────────────────────────
#  GANomaly (single decoder, BCE-with-logits)
# ─────────────────────────────────────────────────────────────────────────────

from networks import NetG, NetD
from loss import reconstruction_loss, latent_consistency_loss, LossWeights, discriminator_loss


class Ganomaly(BaseModel):
        # networks ------------------------------------------------------
        
    @staticmethod  
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight.data, gain=math.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def __init__(self, opt, dataloaders):
        super().__init__(opt, dataloaders)
        self.n_critic = opt.n_critic
        self.netg = NetG(opt).to(self.device)
        self.netd = NetD(opt).to(self.device)

        # use the *un-bound* static function so apply() gets exactly 1 arg
        self.netg.apply(Ganomaly._weights_init)  # ← changed
        self.netd.apply(Ganomaly._weights_init)  # ← changed

        # losses --------------------------------------------------------
        # using logits-version so NetD need NOT include Sigmoid
       # losses --------------------------------------------------------
        # raw helpers from loss.py
        from loss import (
            adversarial_loss_ganomaly, reconstruction_loss,
            latent_consistency_loss, generator_total_loss, 
        )
        self.adversarial_loss = adversarial_loss_ganomaly
        self.reconstruction_loss = reconstruction_loss
        self.latent_loss = latent_consistency_loss
        self.gen_total_loss = generator_total_loss
        #self.grad_penalty = gradient_penalty

        # loss weights (taken from CLI or defaults)
        self.w = LossWeights(
            w_adv=opt.w_adv, w_con=opt.w_con, w_enc=opt.w_enc
        )

        self.lambda_gp = opt.lambdaGP
        # optimisers ----------------------------------------------------
        self.opt_d = optim.Adam(
            self.netd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
        )
        self.opt_g = optim.Adam(
            self.netg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
        )

        self.real_label = 1.0
        self.fake_label = 0.0



    # required by BaseModel --------------------------------------------
    def optims(self):
        return [self.opt_g, self.opt_d]

    # ------------------- single training step --------------------------
    def _forward_g(self, x):
        """Return fake images + latents (zfwd, zbwd)."""
        return self.netg(x)  # xf, z_i, z_o


    def _set_requires_grad(self, net, flag: bool):
        for p in net.parameters():
            p.requires_grad = flag
            
    def _logits(self, d_out):
        return d_out[0] if isinstance(d_out, (tuple, list)) else d_out

    def _features(self, d_out):
        return d_out[1] if isinstance(d_out, (tuple, list)) else None


    # ------------------- Discriminator step ---------------------------

    # ------------------- Discriminator step ---------------------------
    def _backward_d(self, xr, xf):
        """
        Single optimiser step for D using the BCE + R1 loss defined in loss.py.
        """
        # make sure D’s parameters get gradients
        self._set_requires_grad(self.netd, True)
        self.opt_d.zero_grad(set_to_none=True)

        # R1 needs grads wrt the *real* batch
        xr.requires_grad_(True)

        # Forward passes (raw logits)
        logits_real = self._logits(self.netd(xr))           # D(X)
        logits_fake = self._logits(self.netd(xf.detach()))  # D(G(X)); grad to D only

        # Use your centralised loss helper
        from loss import discriminator_loss
        loss_d = discriminator_loss(
            logits_real, logits_fake, xr,
            λ_gp = self.lambda_gp,          # keep or expose via CLI if you like
            label_smooth=0.9   # idem
        )

        loss_d.backward()
        self.opt_d.step()

        return loss_d.item()



    # ------------------- Generator step -------------------------------
    def _backward_g(self, xr, xf, zi, zo):
        """Single optimiser step for G (Eq. 9-12)."""
        self._set_requires_grad(self.netd, False)         # freeze D params
        self.opt_g.zero_grad(set_to_none=True)

        # D(X) is treated as a fixed target in L_adv
        with torch.no_grad():
            feats_real = self._logits(self.netd(xr))

        # D(G(X)) keeps grad path to G, not to D (D params are frozen)
        feats_fake = self._logits(self.netd(xf))

        total, l_adv, l_con, l_enc = self.gen_total_loss(
            feats_real.detach(), feats_fake,   # D(X), D(G(X))
            xr, xf, zi, zo,                      # X, X′,  Z, Z′
            self.w                               # weights
        )

        total.backward()
        self.opt_g.step()

        # restore D parameters for next iteration
        self._set_requires_grad(self.netd, True)

        return {
            "G/total": total.item(),
            "G/adv":   l_adv.item(),
            "G/con":   l_con.item(),
            "G/enc":   l_enc.item(),
        }

    # ------------------- epoch loop ------------------------------------
# ----------------------------------------------------------------------
#  add near the top of model.py (after other imports)
# ----------------------------------------------------------------------
             # auto picks notebook / console style
# ----------------------------------------------------------------------

    # ------------------- epoch loop ------------------------------------
    def _train_epoch(self):
        """One full epoch with a tqdm batch bar."""
        loader = self.dataloaders["train"][0]

        # tqdm setup
        bar = tqdm(loader,
                   desc=f"Epoch {self.epoch}/{self.opt.niter}",
                   leave=False,  # one bar per epoch; removed when done
                   dynamic_ncols=True)

        for (xr,) in bar:
            xr = xr.to(self.device)
            
            for _ in range(self.n_critic):
                xf, _, _ = self._forward_g(xr)
                ld = self._backward_d(xr, xf)

            xf, zi, zo = self._forward_g(xr)
            #ld = self._backward_d(xr, xf)
            lg = self._backward_g(xr, xf, zi, zo)
            lg["D/loss"] = ld

            # live TB logging
            if self.total_steps % self.opt.tb_freq == 0:
                self._tb_log(lg)

            # show running numbers on the tqdm bar
            bar.set_postfix(
                d=f"{ld:.4f}",
                g=f"{lg['G/total']:.4f}",
                refresh=False
            )

            self.total_steps += 1

        # epoch-end print (after bar closes)
        print(f"[epoch {self.epoch}] D: {ld:.4f}  G: {lg['G/total']:.4f}")


    # ------------------- public API ------------------------------------
    # ------------------- public API ------------------------------------
    def fit(self):
        """Training loop with graceful ^C → full checkpoint."""
        try:
            while self.epoch <= self.opt.niter:
                self._train_epoch()

                if self.epoch % self.opt.save_freq == 0:
                    self.save_light(f"epoch{self.epoch}")

                self.epoch += 1

        except KeyboardInterrupt:
            # ⇧ user hit Ctrl-C — save a full checkpoint *right now*
            print("\n[!] Interrupted — saving full checkpoint …")
            self.save(f"INTERRUPT_ep{self.epoch}")

        finally:
            # Always write a final ckpt (may overwrite INTERRUPT tag if loop
            # finished normally, which is fine).
            self.save("final")



# ─────────────────────────────────────────────────────────────────────────────
# eof
