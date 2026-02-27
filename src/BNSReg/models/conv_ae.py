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
        for i in range(self.n_layers):
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
        for i in range(self.n_layers):
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
        # x: (N, L)
        L = x.shape[-1]

        x = x.unsqueeze(1)  # (N, 1, L)
        z = self.encoder(x)
        y = self.decoder(z)

        if y.shape[-1] > L:
            y = y[..., :L]
        elif y.shape[-1] < L:
            y = nn.functional.pad(y, (0, L - y.shape[-1]))

        return y
    
class ConvAEAP(nn.Module): # Area-preserving
    def __init__(
        self,
        seq_length: int = 4, # typically, multiple of a power of 2
        n_layers: int = 8,
        base_dim: int = 16, # base channel dimension
    ):
        super().__init__()

        self.n_layers = n_layers
        self.seq_length = seq_length
        self.base_dim = base_dim

        enc, dec = OrderedDict(), OrderedDict()

        self.channel_dims = [1 if i==0
                        else self.base_dim * (2**i) for i in range(self.n_layers+1)]
        self.kernel_sizes = [self.seq_length // (2**(i+1)) + 1 for i in range(self.n_layers)]

        for i in range(self.n_layers):
            enc[f'conv{i}'] = nn.Conv1d(
                in_channels=self.channel_dims[i],
                out_channels=self.channel_dims[i+1],
                kernel_size=self.kernel_sizes[i]
            )
            enc[f'norm{i}'] = nn.InstanceNorm1d(self.channel_dims[i+1])
            enc[f'act{i}'] = nn.LeakyReLU()
            ###
            j = self.n_layers - i
            dec[f'deconv{i}'] = nn.ConvTranspose1d(
                in_channels=self.channel_dims[j],
                out_channels=self.channel_dims[j-1],
                kernel_size=self.kernel_sizes[j-1]
            )
            dec[f'norm{i}'] = nn.InstanceNorm1d(self.channel_dims[j-1])
            dec[f'act{i}'] = nn.LeakyReLU()
        
        self.encoder = nn.Sequential(enc)
        self.bottleneck = nn.Linear(self.channel_dims[-1], self.channel_dims[-1])
        self.decoder = nn.Sequential(dec)

    def forward(self, x):
        # x: (N, L)
        L = x.shape[-1]
        assert L == self.seq_length

        x = x.unsqueeze(1)  # (N, 1, L)
        z = self.encoder(x)
        z = z.transpose(1, 2)
        for _ in range(4):
            z = self.bottleneck(z)
        z = z.transpose(1, 2)
        y = self.decoder(z)

        if y.shape[-1] > L:
            y = y[..., :L]
        elif y.shape[-1] < L:
            y = nn.functional.pad(y, (0, L - y.shape[-1]))

        return y

class ConvAttentionAEAP(ConvAEAP): # Attention + Area-preserving
    def __init__(
        self,
        seq_length: int = 4, # typically, multiple of a power of 2
        n_layers: int = 8,
        base_dim: int = 16, # base channel dimension
        num_heads: int = 4,
        dropout: float = 0.01,
        
    ):
        super().__init__(seq_length=seq_length, n_layers=n_layers, base_dim=base_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.channel_dims[-1],
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True # (N, L, C)
        )

    def forward(self, x):
        # x: (N, L)
        L = x.shape[-1]
        assert L == self.seq_length

        x = x.unsqueeze(1)  # (N, 1, L)
        z = self.encoder(x).transpose(1, 2) # (N, C, L) -> (N, L, C)
        h, _ = self.attention(z, z, z) # (N, L, C) -> (N, L, E: embed_dim) | E = C
        for _ in range(4):
            h = self.bottleneck(h)
        h, _ = self.attention(h, h, h)
        z = h.transpose(1, 2) # (N, L, C) -> (N, C, L)
        y = self.decoder(z)

        if y.shape[-1] > L:
            y = y[..., :L]
        elif y.shape[-1] < L:
            y = nn.functional.pad(y, (0, L - y.shape[-1]))

        return y