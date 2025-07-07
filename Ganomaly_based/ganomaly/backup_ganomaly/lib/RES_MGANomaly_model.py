from __future__ import annotations
import os, random, numpy as np
from collections import OrderedDict
from typing import Dict
import sys
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Resnetworks import (
    NetG_Multi_RES_GANomaly as NetG,
    NetD_RES_GANomaly,
    weights_init,
)

from loss import (
    generator_total_loss,
    gradient_penalty,
    discriminator_loss,
    LossWeights,
)
from evaluate import evaluate


def _optim_state_to(opt: torch.optim.Optimizer, device):
    """
    Move every tensor in opt.state to `device` ('cpu' or 'cuda:…').
    Parameters themselves are NOT moved – only Adam's exp_avg / exp_avg_sq.
    """
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device, non_blocking=True)

def _load_if_given(model: nn.Module, ckpt_path: str | None, *, name: str):
    if not ckpt_path:
        return

    if not os.path.isfile(ckpt_path):
        print(f"[Init] WARNING – '{ckpt_path}' not found, starting {name} from scratch.")
        return

    sd = torch.load(ckpt_path, map_location="cpu")

    # full or light checkpoint?  → extract sub-dict
    if isinstance(sd, dict) and name in sd:
        sd = sd[name]                    # pull 'netG' or 'netD'

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"[Init] Loaded {name} from '{ckpt_path}'. "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    
def _atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path) 


def _seed_everything(seed: int):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class BaseModel:
    def __init__(self, opt, dataloader: Dict[str, torch.utils.data.DataLoader]):
        _seed_everything(getattr(opt, "manualseed", -1))

        self.opt        = opt
        self.dataloader = dataloader
        self.device     = torch.device(
            "cpu" if opt.device == "cpu" or not torch.cuda.is_available()
            else opt.device
        )

        self.input = torch.empty(1, opt.nc, opt.isize, opt.isize, device=self.device)
        self.gt    = torch.empty(1, dtype=torch.long, device=self.device)
        self.fixed_input = None

        # ── TensorBoard: one run per decoder + one “global” run ──────────
        tb_root = os.path.join(opt.outf, opt.name, "tensorboard")
        os.makedirs(tb_root, exist_ok=True)

        from torch.utils.tensorboard import SummaryWriter
        self.global_writer = SummaryWriter(log_dir=tb_root)  # overall stuff

        self.writers = []
        for j in range(opt.num_generators):
            sub = os.path.join(tb_root, f"dec{j}")
            os.makedirs(sub, exist_ok=True)
            self.writers.append(SummaryWriter(log_dir=sub))

    # ------------------------------------------------------------------
    #  unchanged helper
    # ------------------------------------------------------------------
    def set_input(self, batch, gid=None):
        x = batch[0].to(self.device, non_blocking=True)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        self.input.resize_(x.size()).copy_(x)

        if gid is not None:
            self.gen_idx = torch.tensor(gid, device=self.device)

        if len(batch) > 1 and batch[1] is not None:
            y = batch[1].to(self.device, non_blocking=True)
            self.gt.resize_(y.size()).copy_(y)

        if self.fixed_input is None:
            self.fixed_input = self.input.clone()

class RES_Ganomaly(BaseModel):
    @property
    def name(self):
        return "RES_Ganomaly"

    def __init__(self, opt, dataloader):
        super().__init__(opt, dataloader)

        print(f"[Init] Networks on {self.device}")
        print("[Debug] about to build NetG")
        self.netg = NetG(opt).to(self.device)
        print("[Debug] NetG done")

        print("[Debug] about to build NetD")
        self.netd = NetD_RES_GANomaly(opt).to(self.device)
        print("[Debug] NetD done")

        print("[Debug] applying weight init")
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        print("[Debug] weight init done")


        print("[Debug] trying netG ckpt")
        _load_if_given(self.netg, getattr(opt, "netg_ckpt", None), name="netG")
        print("[Debug] trying netD ckpt")
        _load_if_given(self.netd, getattr(opt, "netd_ckpt", None), name="netD")

        print("[Debug] building D optimiser")
        self.opt_d = optim.Adam(
            self.netd.parameters(), lr=opt.lr,
            betas=(opt.beta1, 0.999)
        )
        print("[Debug] D optimiser done")

        print("[Debug] building G optimisers")
        self.opt_g = []
        for idx, dec in enumerate(self.netg.decoders):
            params = (
                list(self.netg.enc1.parameters()) +
                list(self.netg.enc2.parameters()) +
                list(dec.parameters())
            )
            g_opt = optim.Adam(params, lr=opt.lr,
                              betas=(opt.beta1, 0.999))
            print(f"[Debug]   G-opt {idx} created")
            self.opt_g.append(g_opt)
        print("[Debug] all G optimisers done")

        print("[Debug] RES_Ganomaly.__init__ exit")



        self.lambda_gp = getattr(opt, "lambda_gp", 1.0)  # paper’s best λ
        self.n_critic = getattr(opt, "n_critic", 1)     
        self.loss_weights = LossWeights(
            w_adv=getattr(opt, "w_adv", 1.0),
            w_con=getattr(opt, "w_con", 50.0),
            w_enc=getattr(opt, "w_enc", 1.0),
        )

        self.total_steps = 0
        self.epoch = 0
        self.scaler = torch.amp.GradScaler(device='cuda')



    # --------------------------------------------------------------
    #  optimisation per batch
    #--------------------------------------------------------------
    def optimise(self):
        # -----------------------------------------------------------
        # pick the active decoder index chosen by the dataloader
        # -----------------------------------------------------------
        j = self.gen_idx.item()

        # ===========================================================
        # 1) -----------  critic / discriminator (D) update  --------
        # ===========================================================
        _optim_state_to(self.opt_d, self.device)
        self.opt_d.zero_grad(set_to_none=True)

        # ---- create a fake batch (no grad to G) -------------------
        with torch.no_grad():
            x_hat, _, _ = self.netg.forward_one(self.input, j)  # X′ = G_j(X)

        # ---- critic raw scores (no sigmoid in NetD!) -------------
        with torch.amp.autocast(device_type='cuda'):
            d_real_score, _ = self.netd(self.input)
            d_fake_score, _ = self.netd(x_hat.detach())
            gp_val = gradient_penalty(                # see §4 for fp32 note
                self.netd, self.input, x_hat.detach(),
                device=self.device, λ=self.lambda_gp
            )
            self.err_d = d_fake_score.mean() - d_real_score.mean() + gp_val

        # AMP-aware backward + step
        self.scaler.scale(self.err_d).backward()
        self.scaler.step(self.opt_d)
        self.scaler.update()

        # ---- numbers for TensorBoard (detach so no autograd) -----
        self.err_d_real = torch.sigmoid(d_real_score.detach()).mean()
        self.err_d_fake = torch.sigmoid(d_fake_score.detach()).mean()
        self.gp         = gp_val.detach()

        g_opt = self.opt_g[j]
        _optim_state_to(g_opt, self.device)     # bring moments to GPU
        g_opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda'):
            x_hat, z_i, z_o = self.netg.forward_one(self.input, j)
            d_fake_G, _     = self.netd(x_hat)
            with torch.no_grad():
                d_real_fixed, _ = self.netd(self.input)

            ( self.err_g,
            self.err_g_adv,
            self.err_g_con,
            self.err_g_enc ) = generator_total_loss(
                d_real_fixed, d_fake_G,
                self.input, x_hat, z_i, z_o,
                self.loss_weights
            )

        self.scaler.scale(self.err_g).backward()
        self.scaler.step(g_opt)

        _optim_state_to(g_opt, "cpu")           # park moments back on CPU
        torch.cuda.empty_cache()
        
    def train_one_epoch(self):
        # set networks to training mode
        self.netg.train()
        self.netd.train()

        # initialize accumulators
        d_losses_total, d_losses_real, d_losses_fake, gp_values = [], [], [], []
        g_losses_total, g_losses_adv, g_losses_con, g_losses_enc = [], [], [], []

        # build a single tqdm bar for this epoch
        total_batches = sum(len(loader) for loader in self.dataloader["train"])
        from tqdm import tqdm
        pbar = tqdm(
            total=total_batches,
            desc=f"Epoch {self.epoch+1}/{self.opt.niter}",
            unit="batch",
            ncols=80
        )

        # iterate over each decoder’s DataLoader
        for gid, loader in enumerate(self.dataloader["train"]):
            for batch in loader:
                self.total_steps += 1
                self.set_input(batch, gid)      # pass generator id
                self.optimise()

                # accumulate stats
                d_losses_total.append(self.err_d.item())
                d_losses_real.append(self.err_d_real.item())
                d_losses_fake.append(self.err_d_fake.item())
                gp_values.append(self.gp.item())

                g_losses_total.append(self.err_g.item())
                g_losses_adv.append(self.err_g_adv.item())
                g_losses_con.append(self.err_g_con.item())
                g_losses_enc.append(self.err_g_enc.item())

                # tensorboard log
                if self.total_steps % self.opt.tb_freq == 0:
                    self._tb_log()

                # advance progress bar
                pbar.update(1)

        # compute epoch‐level averages
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

        # close tqdm bar
        pbar.close()


    def _tb_log(self):
        gid = getattr(self, "current_gid", 0)       # set in optimise()
        w   = self.writers[gid]
        s   = self.total_steps

        # discriminator scalars
        w.add_scalar("D/loss_total", self.err_d.item(),   s)
        w.add_scalar("D/real_prob",  self.err_d_real,     s)
        w.add_scalar("D/fake_prob",  self.err_d_fake,     s)
        w.add_scalar("D/gp",         self.gp,             s)

        # generator scalars
        w.add_scalar("G/loss_total", self.err_g,          s)
        w.add_scalar("G/adv",        self.err_g_adv,      s)
        w.add_scalar("G/latent_con", self.err_g_con,      s)
        w.add_scalar("G/recon",      self.err_g_enc,      s)


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

        try:
            for epoch in range(self.epoch, self.opt.niter):
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
            self.global_writer.close()
            for w in self.writers:
                w.close()
                
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


    def save(self, tag="latest"):
        ckpt_dir = os.path.join(self.opt.outf, self.opt.name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        for opt in self.opt_g: _optim_state_to(opt, "cpu")
        _optim_state_to(self.opt_d, "cpu")

        checkpoint = {
            "epoch": self.epoch,
            "netG":  self.netg.state_dict(),
            "netD":  self.netd.state_dict(),
            "optG": [opt.state_dict() for opt in self.opt_g],
            "optD":  self.opt_d.state_dict(),
        }
        _atomic_save(checkpoint, os.path.join(ckpt_dir, f"checkpoint_{tag}.pth"))

    def save_light(self, tag="latest_light"):
        ckpt_dir = os.path.join(self.opt.outf, self.opt.name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        light = {"epoch": self.epoch,
                "netG":  self.netg.state_dict(),
                "netD":  self.netd.state_dict()}
        _atomic_save(light, os.path.join(ckpt_dir, f"checkpoint_{tag}.pth"))

    def load(self, tag="latest"):
        """
        Load one of three formats

        1. Full checkpoint  (save())      → contains epoch, nets, *and* optims
        2. Light checkpoint (save_light)  → contains epoch + nets   (no optims)
        3. Raw state-dict                  → tensor keys directly   (legacy)

        • Networks are always restored.
        • If the file has no optimiser states we keep the optims you
          constructed from CLI hyper-parameters (lr, β₁, …) and print a note.
        • All optimiser tensors are moved to CPU after loading so they can be
          off-loaded safely.
        """
        ckpt_dir = os.path.join(self.opt.outf, self.opt.name, "checkpoints")
        path     = os.path.join(ckpt_dir, f"checkpoint_{tag}.pth")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        ckpt = torch.load(path, map_location="cpu")

        # --------- 1) Networks --------------------------------------------
        if isinstance(ckpt, dict) and "netG" in ckpt:          # full / light
            self.netg.load_state_dict(ckpt["netG"], strict=False)
            self.netd.load_state_dict(ckpt["netD"], strict=False)
        else:                                                  # raw state-dict
            self.netg.load_state_dict(ckpt,         strict=False)
            # netD has to stay at init weights in this legacy case
            print("[Load] raw state-dict detected → loaded into netG only")
            self.epoch = 0
            return                                           # nothing else to do

        # --------- 2) Optimisers  -----------------------------------------
        if "optD" in ckpt and "optG" in ckpt:                 # FULL checkpoint
            self.opt_d.load_state_dict(ckpt["optD"])
            saved_g = ckpt["optG"]

            if len(saved_g) != len(self.opt_g):
                raise RuntimeError(
                    f"Checkpoint has {len(saved_g)} decoders but "
                    f"current run expects {len(self.opt_g)}."
                )

            for opt, sd in zip(self.opt_g, saved_g):
                opt.load_state_dict(sd)

            print(f"[Load] loaded networks *and* optimiser states from '{tag}'")

        else:                                                 # LIGHT checkpoint
            print(f"[Load] light checkpoint – kept optimisers from CLI "
                  f"(lr={self.opt.lr}, β₁={self.opt.beta1})")

        # park all optimiser tensors on CPU so VRAM is free
        _optim_state_to(self.opt_d, "cpu")
        for opt in self.opt_g:
            _optim_state_to(opt, "cpu")

        # --------- 3) Epoch counter ---------------------------------------
        self.epoch = ckpt.get("epoch", 0)
        print(f"[Load] restored epoch {self.epoch+1} from '{tag}'")


    

