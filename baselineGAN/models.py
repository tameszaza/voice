import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, out_channels, base_channels, img_size, n_layers):
        super().__init__()
        self.init_size = img_size // (2 ** n_layers)
        self.fc = nn.Linear(z_dim, base_channels * self.init_size * self.init_size)
        modules = []
        in_ch = base_channels
        for i in range(n_layers):
            out_ch = base_channels * (2**i)
            modules += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
                nn.SELU(inplace=True)
            ]
            in_ch = out_ch
        modules.append(nn.Conv2d(in_ch, out_channels, kernel_size=5, stride=1, padding=2))
        self.net = nn.Sequential(*modules)

    def forward(self, z, target_hw=None):
        batch = z.size(0)
        x = self.fc(z).view(batch, -1, self.init_size, self.init_size)
        x = self.net(x)
        if target_hw is not None:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return torch.tanh(x)

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
        self.fc = nn.Linear(ch, z_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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
        return self.net(x).mean(dim=[1,2,3])

    def intermediate(self, x):
        return self.net[:2](x)
