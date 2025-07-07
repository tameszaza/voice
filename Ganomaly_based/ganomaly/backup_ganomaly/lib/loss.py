from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.autograd as autograd


# --------------------------------------------------------------------------
#  Hyper-parameters
# --------------------------------------------------------------------------

@dataclass
class LossWeights:
    """Weights W_adv, W_con, W_enc in Eq. (12)."""
    w_adv: float = 1.0
    w_con: float = 50.0
    w_enc: float = 1.0


# --------------------------------------------------------------------------
#  Generator losses  (Eq. 9-12)
# --------------------------------------------------------------------------

import torch
import torch.nn.functional as F          # only for clarity; torch.norm used below

def adversarial_loss_ganomaly(d_real_logits: torch.Tensor,
                              d_fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Exact Eq. 9
    L_adv = || σ(D(X)) – σ(D(X′)) ||_2
    • ℓ₂ (Euclidean) norm per sample, NOT squared, then mean over the batch.
    """
    real_prob = torch.sigmoid(d_real_logits)
    fake_prob = torch.sigmoid(d_fake_logits)
    diff = (real_prob - fake_prob).view(real_prob.size(0), -1)  # flatten per sample
    return torch.norm(diff, p=2, dim=1).mean()


def reconstruction_loss(x: torch.Tensor,
                        x_recon: torch.Tensor) -> torch.Tensor:
    """
    Exact Eq. 11
    L_enc = || X – X′ ||_1
    • ℓ₁ norm (sum of absolutes) per sample, then mean over the batch.
    """
    diff = torch.abs(x - x_recon).view(x.size(0), -1)  # flatten per sample
    return diff.sum(dim=1).mean()


def latent_consistency_loss(z: torch.Tensor,
                            z_recon: torch.Tensor) -> torch.Tensor:
    """
    Exact Eq. 10
    L_con = || Z – Z′ ||_2
    • ℓ₂ norm per sample, NOT squared, then mean over the batch.
    """
    diff = (z - z_recon).view(z.size(0), -1)  # flatten per sample
    return torch.norm(diff, p=2, dim=1).mean()




def generator_total_loss(d_real_logits: torch.Tensor,
                         d_fake_logits: torch.Tensor,
                         x: torch.Tensor,
                         x_recon: torch.Tensor,
                         z: torch.Tensor,
                         z_recon: torch.Tensor,
                         weights: LossWeights) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L_G = W_adv·L_adv + W_con·L_con + W_enc·L_enc   (Eq. 12)
    Returns total, L_adv, L_con, L_enc.
    """
    l_adv = adversarial_loss_ganomaly(d_real_logits, d_fake_logits)
    l_con = latent_consistency_loss(z, z_recon)
    l_enc = reconstruction_loss(x, x_recon)

    total = (weights.w_adv * l_adv +
             weights.w_con * l_con +
             weights.w_enc * l_enc)
    return total, l_adv, l_con, l_enc


# --------------------------------------------------------------------------
#  Discriminator loss with gradient penalty  (Eq. 13)
# --------------------------------------------------------------------------

# ---- loss.py ------------------------------------------------------
def gradient_penalty(D, real, fake, device, λ=10.0):
    with torch.amp.autocast(device_type='cuda',enabled=False): 
        α = torch.rand(real.size(0), 1, 1, 1, device=device)
        x_hat = (α * real + (1 - α) * fake).requires_grad_(True)

        d_hat = D(x_hat)[0]                       # raw score
        grad   = autograd.grad(
            outputs=d_hat, inputs=x_hat,
            grad_outputs=torch.ones_like(d_hat),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        grad = grad.view(grad.size(0), -1)
        gp   = ((grad.norm(2, dim=1) - 1) ** 2).mean() * λ
        return gp



def discriminator_loss(d_real_prob: torch.Tensor,
                       d_fake_prob: torch.Tensor,
                       gp: torch.Tensor) -> torch.Tensor:
    """
    L_D = E[D(x′)]  –  E[log D(x)]  +  λ·GP          (Eq. 13)
    • Inputs d_real_prob & d_fake_prob are **post-sigmoid probabilities**.
    • The loss is minimised with ordinary gradient descent.
    """
    fake_term = d_fake_prob.mean()
    real_term = torch.log(d_real_prob + 1e-12).mean()   # avoid log(0)

    return fake_term - real_term + gp
