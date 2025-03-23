import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------
# Positional Encoding (Same as before)
# ----------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding.
    Adds positional information to the input embeddings.
    """
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# -----------------------------------------------------------
# Residual Convolution Block (Modified to crop shortcut branch)
# -----------------------------------------------------------
class ResidualConvBlock(nn.Module):
    """
    A residual convolution block that applies convolution, activation,
    and adds the input (using a 1x1 convolution if needed). If there is a mismatch
    in the time dimension due to external padding, the shortcut branch is cropped.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, use_spectral_norm=True):
        super(ResidualConvBlock, self).__init__()
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        self.conv = conv
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        # If the input and output channels differ or stride is not 1, use a shortcut convolution.
        if in_channels != out_channels or stride != 1:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.res_conv = None

    def forward(self, x):
        # x shape: [B, C_in, T_in]
        residual = x
        out = self.activation(self.conv(x))  # out shape: [B, C_out, T_out]
        if self.res_conv is not None:
            residual = self.res_conv(x)         # residual shape: [B, C_out, T_res]
            # If there's a mismatch in the temporal dimension, crop the residual.
            if residual.shape[2] != out.shape[2]:
                diff = residual.shape[2] - out.shape[2]
                crop_left = diff // 2
                crop_right = diff - crop_left
                residual = residual[:, :, crop_left:residual.shape[2]-crop_right]
        return out + residual

# -----------------------------------------------------------
# Conformer Components (State-of-the-Art for Audio Tasks)
# -----------------------------------------------------------
class FeedForwardModule(nn.Module):
    """
    A feed-forward module as used in Conformer.
    Uses two linear layers with an activation in between.
    """
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # Alternatively, you could use GELU or Swish
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d_model]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

class ConvolutionModule(nn.Module):
    """
    A convolution module as used in Conformer.
    Applies pointwise conv, GLU activation, depthwise conv, normalization, activation,
    and a final pointwise conv.
    """
    def __init__(self, d_model, conv_kernel_size, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        # Depthwise convolution: groups equals the number of channels (d_model)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ReLU()  # Can also use Swish or GELU
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d_model] -> transpose for conv layers: [B, d_model, T]
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # GLU splits the channels
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # Back to [B, T, d_model]
        return x

class ConformerBlock(nn.Module):
    """
    A single Conformer block that combines feed-forward modules,
    multi-head self-attention, and convolutional modules with normalization
    and residual connections.
    """
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, conv_kernel_size=31):
        super(ConformerBlock, self).__init__()
        self.ffn1 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.norm_ffn1 = nn.LayerNorm(d_model)
        
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm_mha = nn.LayerNorm(d_model)
        
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        
        self.ffn2 = FeedForwardModule(d_model, dim_feedforward, dropout)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, T, d_model]
        residual = x
        x = self.norm_ffn1(x)
        x = residual + 0.5 * self.ffn1(x)
        
        residual = x
        x = self.norm_mha(x)
        attn_output, _ = self.mha(x, x, x)
        x = residual + self.dropout(attn_output)
        
        residual = x
        x = self.norm_conv(x)
        x = residual + self.dropout(self.conv_module(x))
        
        residual = x
        x = self.norm_ffn2(x)
        x = residual + 0.5 * self.ffn2(x)
        
        return x

# -----------------------------------------------------------
# State-of-the-Art Discriminator Using Conformer Blocks
# -----------------------------------------------------------
class Discriminator(nn.Module):
    """
    Discriminator that uses a CNN front-end with residual blocks followed by
    two stacked Conformer blocks. The final projection layer produces a time-distributed
    output in the same format [B, 1, T'].
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        # 1) CNN Front-End using Residual Blocks
        self.conv_block1 = nn.Sequential(
            nn.ReflectionPad1d(7),
            ResidualConvBlock(1, 32, kernel_size=15, stride=1, padding=0, use_spectral_norm=True)
        )
        self.conv_block2 = ResidualConvBlock(32, 64, kernel_size=41, stride=4, padding=20, groups=4)
        self.conv_block3 = ResidualConvBlock(64, 128, kernel_size=41, stride=4, padding=20, groups=16)
        self.conv_block4 = ResidualConvBlock(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)

        # 2) Positional Encoding and Stacked Conformer Blocks
        self.d_model = 128
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.conformer_block1 = ConformerBlock(d_model=self.d_model, nhead=4, dim_feedforward=256, dropout=0.1, conv_kernel_size=31)
        self.conformer_block2 = ConformerBlock(d_model=self.d_model, nhead=4, dim_feedforward=256, dropout=0.1, conv_kernel_size=31)

        # 3) Final Projection to produce time-distributed output [B, 1, T']
        self.out_conv = nn.Conv1d(self.d_model, 1, kernel_size=1)

    def forward(self, x):
        """
        Input: x of shape [B, 1, T]
        Output: x of shape [B, 1, T'] (time-distributed score)
        """
        # CNN front-end for local feature extraction
        x = self.conv_block1(x)   # -> [B, 32, T] (after reflection pad and conv)
        x = self.conv_block2(x)   # -> [B, 64, T_downsampled]
        x = self.conv_block3(x)   # -> [B, 128, T_downsampled]
        x = self.conv_block4(x)   # -> [B, 128, T_downsampled]

        # Prepare for Conformer: reshape to [B, T, d_model]
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.conformer_block1(x)
        x = self.conformer_block2(x)

        # Convert back to [B, d_model, T] for final projection
        x = x.permute(0, 2, 1)
        x = self.out_conv(x)  # -> [B, 1, T']
        return x

# -----------------------------------------------------------
# Quick Test to Verify Output Format
# -----------------------------------------------------------
if __name__ == '__main__':
    model = Discriminator()
    x = torch.randn(3, 1, 24000)  # Example input [B, 1, T]
    score = model(x)
    print("Output shape:", score.shape)  # Should be [B, 1, T']
