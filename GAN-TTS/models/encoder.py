import torch.nn as nn
from modules import Conv1d  # Using your custom Conv1d module

class Encoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        self.layers = nn.Sequential(
            Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3),  # Keep sequence length
            nn.LeakyReLU(0.2, inplace=True),
            Conv1d(64, 128, kernel_size=4, stride=2, padding=1),  # Downsample by 2
            nn.LeakyReLU(0.2, inplace=True),
            Conv1d(128, 256, kernel_size=4, stride=2, padding=1),  # Downsample by 2 again
            nn.LeakyReLU(0.2, inplace=True),
            Conv1d(256, feature_dim, kernel_size=4, stride=2, padding=1),  # Final feature extraction
            nn.AdaptiveAvgPool1d(1)  # Reduce sequence to 1 (global pooling)
        )

    def forward(self, x):
        # Input shape: (batch_size, input_channels, sequence_length)
        x = self.layers(x)
        return x.squeeze(-1)  # Output shape: (batch_size, feature_dim)
