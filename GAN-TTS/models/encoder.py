import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad1d(3),  # Reflection padding to preserve input length
            nn.utils.spectral_norm(nn.Conv1d(input_channels, 64, kernel_size=7, stride=1)),  # Conv layer 1
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)),  # Conv layer 2
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1)),  # Conv layer 3
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv1d(256, feature_dim, kernel_size=4, stride=2, padding=1)),  # Final Conv
            nn.AdaptiveAvgPool1d(1)  # Reduce sequence to a single feature
        )

    def forward(self, x):
        # Input: (batch_size, input_channels, sequence_length)
        x = self.layers(x)
        return x.squeeze(-1)  # Output: (batch_size, feature_dim)

if __name__ == '__main__':
    model = Encoder(input_channels=1, feature_dim=128)
    x = torch.randn(3, 1, 24000)  # Example input (batch_size=3, input_channels=1, sequence_length=24000)
    output = model(x)
    print(output.shape)  # Expected output: (3, 128)
