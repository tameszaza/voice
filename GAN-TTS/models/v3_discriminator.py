import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding for Transformers.
    This helps the model understand the relative ordering of sequence elements.
    """
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [max_len, d_model] -> [1, max_len, d_model] for easy broadcast
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: [B, T, d_model]
        We'll add the positional encoding up to length T
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TransformerBlock(nn.Module):
    """
    A single Transformer encoder block (multi-head self-attention + feed-forward).
    """
    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # So we can keep shape as [B,T,C]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        """
        x shape: [B, T, d_model]
        """
        out = self.transformer(x)  # [B, T, d_model]
        return out

class Discriminator(nn.Module):
    """
    An improved discriminator that first uses convolution to extract local features,
    then uses a Transformer block to learn global contexts, and finally projects
    down to a single output channel.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        # ----------------------------
        # 1) Convolutional front-end
        # ----------------------------
        # We'll keep some of the logic from your old v2 version, but reduce the depth
        # a bit so we can feed it into the Transformer.

        # reflection pad -> conv -> leaky relu
        self.conv1 = nn.Sequential(
            nn.ReflectionPad1d(7),
            nn.utils.spectral_norm(nn.Conv1d(1, 32, kernel_size=15)),  # up from 16 -> 32 channels
            nn.LeakyReLU(0.2, inplace=True),
        )

        # conv -> leaky relu (with stride for downsampling)
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(32, 64, kernel_size=41, stride=4, padding=20, groups=4)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # conv -> leaky relu (more channels)
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(64, 128, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # We'll stop at 128 channels to keep the dimension moderate for the Transformer.
        # If we keep pushing up to 1024 channels, it becomes large and potentially unwieldy
        # for a self-attention mechanism.

        # ----------------------------
        # 2) Transformer block
        # ----------------------------
        self.d_model = 128  # same as the channel count from the last conv
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.transformer_block = TransformerBlock(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=512,  # a bit smaller to save memory
            dropout=0.1
        )

        # ----------------------------
        # 3) Final projection
        # ----------------------------
        # We project back to 1 channel (like your old code),
        # with kernel_size=1 so it doesn't look at extra context here.
        # We'll skip spectral norm on the last step or you can keep it if you like.
        self.out_conv = nn.Conv1d(self.d_model, 1, kernel_size=1)

    def forward(self, x):
        """
        x: [B, 1, T]
        returns: [B, 1, T_reduced] or possibly [B, 1, T'].
        """
        # 1) CNN front-end
        x = self.conv1(x)    # [B, 32, T1]
        x = self.conv2(x)    # [B, 64, T2]
        x = self.conv3(x)    # [B, 128, T3]

        # 2) Prepare for Transformer
        #    Transform from shape [B, C, T3] -> [B, T3, C]
        x = x.permute(0, 2, 1)  # [B, T3, 128]

        # 2a) Add positional encodings
        x = self.pos_encoder(x) # [B, T3, 128]

        # 2b) Transformer
        x = self.transformer_block(x)  # [B, T3, 128]

        # 3) Project back to [B, 1, T3]
        #    So first reorder [B, T3, 128] -> [B, 128, T3]
        x = x.permute(0, 2, 1)
        x = self.out_conv(x)  # [B, 1, T3]

        return x

# --------------------------------------------------------------------
# Quick test if running this file standalone
# --------------------------------------------------------------------
if __name__ == '__main__':
    model = Discriminator()
    x = torch.randn(3, 1, 24000)  # batch=3, single-channel, T=24000
    score = model(x)
    print("Output shape:", score.shape)
