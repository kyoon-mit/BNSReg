# Autoencoder for sequential data with 1 dimensional convolutional layers

import torch.nn as nn
from collections import OrderedDict

class ConvAE(nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        latent_channels: int = 8,
        kernel_size: int = 5,
        pool_stride: int = 2,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError('kernel_size should be odd to use same-length padding cleanly.')

        self.n_layers = n_layers
        self.latent_channels = latent_channels
        self.kernel_size = kernel_size
        self.pool_stride = pool_stride
        self.pad = kernel_size // 2

        enc = OrderedDict()
        for i in range(n_layers):
            in_ch = 1 if i == 0 else latent_channels
            enc[f'conv{i}'] = nn.Conv1d(
                in_channels=in_ch,
                out_channels=latent_channels,
                kernel_size=kernel_size,
                padding=self.pad,
            )
            enc[f'act{i}'] = nn.LeakyReLU()
            enc[f'pool{i}'] = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.encoder = nn.Sequential(enc)

        dec = OrderedDict()
        for i in range(n_layers):
            dec[f'deconv{i}'] = nn.ConvTranspose1d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=pool_stride,
                stride=pool_stride,
            )
            dec[f'act{i}'] = nn.LeakyReLU()
        dec['out'] = nn.Conv1d(latent_channels, 1, kernel_size=1)
        self.decoder = nn.Sequential(dec)

    def forward(self, x):
        # x: (B, L)
        L = x.shape[-1]

        x = x.unsqueeze(1)  # (B, 1, L)
        z = self.encoder(x)
        y = self.decoder(z)

        if y.shape[-1] > L:
            y = y[..., :L]
        elif y.shape[-1] < L:
            y = nn.functional.pad(y, (0, L - y.shape[-1]))

        return y
