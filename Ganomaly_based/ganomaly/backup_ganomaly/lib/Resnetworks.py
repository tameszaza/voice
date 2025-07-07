""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.parallel
import math

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Squeeze
        self.excite = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True), 
            nn.ReLU(inplace=False),
            # Your printout had bias=False for the second linear, so I'll match that.
            # If the paper or source intended True, you can change it back.
            nn.Linear(channel // reduction, channel, bias=True), 
            nn.Sigmoid()
        ) # Excitation

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size() # b, c from your code
        
        # Squeeze
        squeezed_tensor = self.avg_pool(x) # Shape: (batch_size, num_channels, 1, 1)
        
        # Reshape for FC layers: from (b, c, 1, 1) to (b, c)
        reshaped_for_excite = squeezed_tensor.view(batch_size, num_channels)
        
        # Excitation: self.excite expects (b, c) and outputs (b, c)
        channel_weights = self.excite(reshaped_for_excite)
        
        # Reshape weights back for scaling: from (b, c) to (b, c, 1, 1)
        reshaped_weights_for_scaling = channel_weights.view(batch_size, num_channels, 1, 1)
        
        # Scale original features
        scaled_features = x * reshaped_weights_for_scaling.expand_as(x)
        return scaled_features

class ResidualSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ResidualSEBlock, self).__init__()
        # first 3×3 conv
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=False)
        # second 3×3 conv
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        # squeeze-and-excitation on the **input** X
        self.se    = SELayer(channels, reduction)

    def forward(self, x):
        identity = x
        # 1) SE‐recalibrate the input
        out = self.se(x)
        # 2) first conv→BN→ReLU
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        # 3) second conv→BN
        out = self.conv2(out)
        out = self.bn2(out)
        # 4) add skip
        out = out + identity
        return out

class Encoder_RES_GANomaly(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, add_final_conv=True):
        super(Encoder_RES_GANomaly, self).__init__()
        self.ngpu = ngpu
        self.add_final_conv = add_final_conv

        if not (isize > 0 and (isize & (isize - 1) == 0) and isize >= 4):
            raise ValueError("isize has to be a power of 2 and >= 4, e.g., 32, 64, 128")

        main = nn.Sequential()
        
        main.add_module(f'initial-conv-{nc}to{ngf}',
                        nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module(f'initial-bn-{ngf}', nn.BatchNorm2d(ngf))
        main.add_module(f'initial-lrelu-{ngf}', nn.LeakyReLU(0.2, inplace=False))
        main.add_module(f'initial-resblock-{ngf}', ResidualSEBlock(ngf))

        current_channels = ngf
        current_isize = isize // 2

        # Number of further downsampling stages needed to reach 4x4 spatial size
        # Example: if isize=128, current_isize=64. To get from 64 to 4: 64->32, 32->16, 16->8, 8->4 (4 stages)
        num_pyramid_stages = int(math.log2(current_isize / 4))

        for i in range(num_pyramid_stages):
            in_feat = current_channels
            # Double channels, but cap at 1024 as per Figure 4 (e.g., 512 -> 1024)
            out_feat = min(current_channels * 2, 1024) 
            
            # Downsampling Block (Conv + BN + LReLU)
            # This corresponds to the "Conv2d" labels in Figure 4's main path
            main.add_module(f'pyramid-{i}-conv-{in_feat}to{out_feat}',
                            nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module(f'pyramid-{i}-bn-{out_feat}', nn.BatchNorm2d(out_feat))
            main.add_module(f'pyramid-{i}-lrelu-{out_feat}', nn.LeakyReLU(0.2, inplace=False))
            
            # Add ResidualSEBlock after downsampling, operating at 'out_feat' channels
            main.add_module(f'pyramid-{i}-resblock-{out_feat}', ResidualSEBlock(out_feat))
            
            current_channels = out_feat
            current_isize //= 2
            

        if self.add_final_conv:
            main.add_module(f'final-conv-{current_channels}to{nz}',
                            nn.Conv2d(current_channels, nz, kernel_size=4, stride=1, padding=0, bias=False))

        self.main = main

    def forward(self, input_tensor): # Renamed from 'input' to avoid shadowing built-in
        if self.ngpu > 1 and input_tensor.device.type == 'cuda':
            output = nn.parallel.data_parallel(self.main, input_tensor, range(self.ngpu))
        else:
            output = self.main(input_tensor)
        return output
    
class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        # Ensure query_key_chan is at least 1
        query_key_chan = max(1, in_dim // 8) 

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=query_key_chan, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=query_key_chan, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        N = width * height

        proj_query = self.query_conv(x).view(m_batchsize, -1, N).permute(0, 2, 1) # B x N x C_qk
        proj_key = self.key_conv(x).view(m_batchsize, -1, N) # B x C_qk x N
        energy = torch.bmm(proj_query, proj_key) # B x N x N
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(m_batchsize, -1, N) # B x C_v x N (C_v = C here)
        
        out_bmm = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x C x N
        out = out_bmm.view(m_batchsize, C, width, height)
        
        out = self.gamma * out + x
        return out
    
class Decoder_RES_GANomaly(nn.Module):
    def __init__(self, isize, nz, nc, ngf_deepest, ngpu=0):
        """
        Decoder for RES-GANomaly, based on Figure 5 and text.
        isize: target output image size (e.g., 128)
        nz: size of latent z vector (e.g., 100)
        nc: number of channels in output image (e.g., 1 for grayscale)
        ngf_deepest: number of filters at the deepest part of the decoder (e.g., 1024 at 4x4 stage)
        ngpu: number of GPUs
        """
        super(Decoder_RES_GANomaly, self).__init__()
        self.ngpu = ngpu
        self.isize = isize # Target output size

        main = nn.Sequential()

        current_channels = nz
        target_channels_block1 = ngf_deepest # e.g. 1024
        main.add_module(f'initial-{current_channels}to{target_channels_block1}-convt',
                        nn.ConvTranspose2d(current_channels, target_channels_block1, kernel_size=4, stride=1, padding=0, bias=False))
        main.add_module(f'initial-{target_channels_block1}-batchnorm', nn.BatchNorm2d(target_channels_block1))
        main.add_module(f'initial-{target_channels_block1}-relu', nn.ReLU(False))
        # According to Figure 5, Self-Attention is present at this stage (1024*4*4)
        main.add_module(f'initial-{target_channels_block1}-selfattention', SelfAttention(target_channels_block1))
        current_channels = target_channels_block1 # Now 1024
        current_csize = 4

        up_block_idx = 0
        while current_csize < self.isize // 2:
            target_channels_block_up = current_channels // 2 # Halve channels
            if target_channels_block_up < nc*2 and current_csize*2 < self.isize//2 : # Ensure we don't go below a reasonable minimum before the last two stages

                 pass

            main.add_module(f'pyramid-{up_block_idx}-{current_channels}to{target_channels_block_up}-convt',
                            nn.ConvTranspose2d(current_channels, target_channels_block_up, kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module(f'pyramid-{up_block_idx}-{target_channels_block_up}-batchnorm', nn.BatchNorm2d(target_channels_block_up))
            main.add_module(f'pyramid-{up_block_idx}-{target_channels_block_up}-relu', nn.ReLU(False))
            # Add Self-Attention as per paper
            main.add_module(f'pyramid-{up_block_idx}-{target_channels_block_up}-selfattention', SelfAttention(target_channels_block_up))
            
            current_channels = target_channels_block_up
            current_csize *= 2
            up_block_idx += 1
        
        main.add_module(f'final-{current_channels}to{nc}-convt',
                        nn.ConvTranspose2d(current_channels, nc, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module(f'final-{nc}-tanh', nn.Tanh()) # Only Tanh for the last layer
        
        self.main = main

    def forward(self, input_tensor): # Renamed from 'input'
        if self.ngpu > 1 and input_tensor.device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            output = nn.parallel.data_parallel(self.main, input_tensor, list(range(self.ngpu)))
        else:
            output = self.main(input_tensor)
        return output

class NetD_RES_GANomaly(nn.Module):
    """
    Multi-scale discriminator for RES-GANomaly.
    All conv layers: kernel_size=5, stride=2, padding=2, no BatchNorm.
    Activation: LeakyReLU(0.2).
    """
    def __init__(self, opt):
        super(NetD_RES_GANomaly, self).__init__()
        self.ngpu    = opt.ngpu
        self.isize   = opt.isize
        self.nc      = opt.nc          
        ndf          = getattr(opt, 'ndf', 128)

        main = nn.Sequential()
        # → 1) Initial downsample: nc → ndf
        main.add_module('conv0',
            nn.Conv2d(self.nc, ndf, kernel_size=5, stride=2, padding=2, bias=False))
        main.add_module('lrelu0', nn.LeakyReLU(0.2, inplace=False))

        current_channels = ndf
        current_size     = self.isize // 2  
        main.add_module('pyramid0_conv',
            nn.Conv2d(current_channels, current_channels*2,
                      kernel_size=5, stride=2, padding=2, bias=False))
        main.add_module('pyramid0_lrelu', nn.LeakyReLU(0.2, inplace=False))
        current_channels *= 2
        current_size     //= 2

        main.add_module('pyramid1_conv',
            nn.Conv2d(current_channels, current_channels,
                      kernel_size=5, stride=2, padding=2, bias=False))
        main.add_module('pyramid1_lrelu', nn.LeakyReLU(0.2, inplace=False))
        current_size //= 2

        if current_size > 4:
            main.add_module('pyramid2_conv',
                nn.Conv2d(current_channels, current_channels,
                          kernel_size=5, stride=2, padding=2, bias=False))
            main.add_module('pyramid2_lrelu', nn.LeakyReLU(0.2, inplace=False))
            current_size //= 2

        self.features_extractor = main

        self.classifier = nn.Sequential(
            nn.Conv2d(current_channels, 1,
                      kernel_size=current_size, bias=False),
            #nn.Sigmoid()  # if you want probabilities; omit for pure WGAN-GP critic
        )

    def forward(self, x):
        # Multi-GPU support
        if self.ngpu > 1 and x.is_cuda:
            feats = nn.parallel.data_parallel(self.features_extractor, x, list(range(self.ngpu)))
            out   = nn.parallel.data_parallel(self.classifier,         feats, list(range(self.ngpu)))
        else:
            feats = self.features_extractor(x)
            out   = self.classifier(feats)

        out = out.view(-1)  # (batch_size,)
        return out, feats

##
class NetG_RES_GANomaly(nn.Module):
    def __init__(self, opt):
        super(NetG_RES_GANomaly, self).__init__()
        
        _isize = opt.isize
        _ngf = opt.ngf
        _nc = opt.nc
        _nz = opt.nz # Latent dimension
        _ngpu = opt.ngpu

        temp_encoder = Encoder_RES_GANomaly(isize=_isize, nz=_nz, nc=_nc, ngf=_ngf, ngpu=_ngpu, add_final_conv=False)
        
        with torch.no_grad():
            dummy_input_for_shape = torch.randn(1, _nc, _isize, _isize)
            encoder_output_before_latent = temp_encoder(dummy_input_for_shape)
            ngf_deepest_calculated = encoder_output_before_latent.size(1) # Get channels at 4x4
        
        print(f"NetG_RES_GANomaly: Calculated ngf_deepest for decoder: {ngf_deepest_calculated}")

        self.encoder1 = Encoder_RES_GANomaly(isize=opt.isize, nz=opt.nz, nc=opt.nc, ngf=opt.ngf, ngpu=opt.ngpu)
        self.decoder = Decoder_RES_GANomaly(isize=opt.isize, nz=opt.nz, nc=opt.nc, ngf_deepest=ngf_deepest_calculated, ngpu=opt.ngpu)
        self.encoder2 = Encoder_RES_GANomaly(isize=opt.isize, nz=opt.nz, nc=opt.nc, ngf=opt.ngf, ngpu=opt.ngpu)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o

class NetG_Multi_RES_GANomaly(nn.Module):
    """
    G = Encoder₁  →  {Decoder_j}_{j=1..k}  →  Encoder₂
        • k independent decoders
        • shared encoders
    """
    def __init__(self, opt):
        super().__init__()
        self.k = opt.num_generators
        _isize, _ngf, _nc, _nz, _ngpu = (
            opt.isize, opt.ngf, opt.nc, opt.nz, opt.ngpu
        )

        # ---------- shared encoders --------------------------------------
        self.enc1 = Encoder_RES_GANomaly(_isize, _nz, _nc, _ngf, _ngpu)
        self.enc2 = Encoder_RES_GANomaly(_isize, _nz, _nc, _ngf, _ngpu)

        # figure out deepest channel count once
        with torch.no_grad():
            dummy = torch.randn(1, _nc, _isize, _isize)
            deepest = self.enc1(dummy).size(1)

        # ---------- k independent decoders --------------------------------
        self.decoders = nn.ModuleList(
            [
                Decoder_RES_GANomaly(
                    isize=_isize, nz=_nz, nc=_nc,
                    ngf_deepest=deepest, ngpu=_ngpu
                )
                for _ in range(self.k)
            ]
        )

    # single-decoder forward (training)
    def forward_one(self, x, j:int):
        z_i   = self.enc1(x)
        x_hat = self.decoders[j](z_i)
        z_o   = self.enc2(x_hat)
        return x_hat, z_i, z_o

    # all-decoder forward (inference)
    def forward_all(self, x):
        z_i = self.enc1(x)
        x_hats, z_os = [], []
        for dec in self.decoders:
            x_hat = dec(z_i)
            x_hats.append(x_hat)
            z_os.append(self.enc2(x_hat))
        return x_hats, z_i, z_os