import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False
# -----------------------------------------------------------------------------
#  Models
# -----------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, out_channels, base_channels, img_size, n_layers):
        super().__init__()
        # compute spatial size after n_layers of upsampling (×2 each)
        self.init_size = img_size // (2 ** n_layers)
        self.fc = nn.Linear(z_dim, base_channels * self.init_size * self.init_size)
        modules = []
        in_ch = base_channels
        # three upsample+conv blocks
        for i in range(n_layers):
            out_ch = base_channels * (2**i)
            modules += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
                nn.SELU(inplace=True)
            ]
            in_ch = out_ch
        # final conv back to data channels
        modules.append(nn.Conv2d(in_ch, out_channels, kernel_size=5, stride=1, padding=2))
        self.net = nn.Sequential(*modules)

    def forward(self, z, k):
        """
        z: (B, z_dim)
        k: (B,) with cluster assignments in [0, n_clusters)
        We assume one Generator instance per cluster is created externally,
        so here we just generate for a single cluster.
        """
        batch = z.size(0)
        x = self.fc(z).view(batch, -1, self.init_size, self.init_size)
        x = self.net(x)
        return torch.tanh(x)

class MultiGenerator(nn.Module):
    def __init__(self, z_dim, out_channels, base_channels, img_size, n_layers, n_clusters):
        super().__init__()
        self.generators = nn.ModuleList([
            Generator(z_dim, out_channels, base_channels, img_size, n_layers)
            for _ in range(n_clusters)
        ])
    def forward(self, z, k, target_hw=None):
        B = z.size(0)
        device = z.device
        gen0 = self.generators[0]
        # Use target_hw if provided, else compute as before
        if target_hw is not None:
            h, w = target_hw
        else:
            n_scales = sum(isinstance(m, nn.Upsample) for m in gen0.net)
            h = int(gen0.init_size * (2 ** n_scales))
            w = h
        out = torch.zeros(B, gen0.net[-1].out_channels, h, w, device=device)
        for i, g in enumerate(self.generators):
            idx = (k == i).nonzero(as_tuple=False).view(-1)
            if idx.numel():
                gi_out = g(z[idx], None)
                # If output size does not match, upsample/crop as needed
                if gi_out.shape[2:] != (h, w):
                    gi_out = F.interpolate(gi_out, size=(h, w), mode='bilinear', align_corners=False)
                out[idx] = gi_out
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, z_dim, base_channels, img_size, n_layers):
        super().__init__()
        modules = []
        ch = in_channels
        for i in range(n_layers):
            out_ch = base_channels * (2**i)
            kernel = 4 if i < n_layers-1 else 3
            modules.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, kernel_size=kernel, stride=2, padding=1),
                nn.SELU(inplace=True)
            ))
            ch = out_ch
        self.conv = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(ch, z_dim)

    def forward(self, x):
        x0 = self.conv(x)
        if DEBUG:
            print(f"[Encoder] after conv: {x0.shape}")      # e.g. (N, ch, H', W')
        x1 = self.pool(x0)
        if DEBUG:
            print(f"[Encoder] after pool: {x1.shape}")      # (N, ch, 1, 1)
        x2 = x1.view(x1.size(0), -1)
        if DEBUG:
            print(f"[Encoder] after view: {x2.shape}")      # (N, ch)
            print(f"[Encoder] fc.weight.shape: {self.fc.weight.shape}")  # (z_dim, ch)
        out = self.fc(x2)
        if DEBUG:
            print(f"[Encoder] output: {out.shape}")        # (N, z_dim)
        return out



class MultiEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, base_channels, img_size, n_layers, n_clusters):
        super().__init__()
        self.encoders = nn.ModuleList([
            Encoder(in_channels, z_dim, base_channels, img_size, n_layers)
            for _ in range(n_clusters)
        ])
    def forward(self, x, k):
        B = x.size(0)
        z = torch.zeros(B, self.encoders[0].fc.out_features, device=x.device)
        for i, e in enumerate(self.encoders):
            mask = (k == i).nonzero(as_tuple=False).view(-1)
            if DEBUG:
                print(f"[MultiEncoder] cluster {i}: {mask.numel()} samples, mask={mask}")
            if mask.numel():
                xi = x[mask]
                zi = e(xi)
                z[mask] = zi
        if DEBUG:
            print(f"[MultiEncoder] final z shape: {z.shape}")
        return z

class Discriminator(nn.Module):
    def __init__(self, in_channels, base_channels, n_layers):
        super().__init__()
        modules = []
        ch = in_channels
        for i in range(n_layers):
            out_ch = base_channels * (2**i)
            modules.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            ch = out_ch
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        out = self.net(x)
        return out.mean(dim=[1,2,3])   # scalar critic score per sample

    def intermediate(self, x):
        # feature map after second conv layer:
        return self.net[:2](x)

class Classifier(nn.Module):
    def __init__(self, in_channels, n_clusters):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2)
        self.bn   = nn.BatchNorm2d(128)
        self.act  = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(128, n_clusters)

    def forward(self, feat):
        x = self.act(self.bn(self.conv(feat)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

class Bandit(nn.Module):
    def __init__(self, n_clusters):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_clusters))
    def prior(self):
        return F.softmax(self.logits, dim=0)
    def sample(self, batch):
        p = self.prior()
        return torch.multinomial(p, batch, replacement=True)

