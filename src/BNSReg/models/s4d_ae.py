# Autoencoder based on the S4D architecture

import torch.nn as nn
from BNSReg.models.s4d import S4Model

class S4DAE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        seq_length: int,
        s4d_d_model: int,
        s4d_d_state: int,
        s4d_n_layers: int,
        dropout: int,
        prenorm: bool = False,
        lr: int | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1
    ):
        self.s4dae = S4Model(
            d_input=input_channels,
            d_output=seq_length,
            d_model=s4d_d_model,
            d_state=s4d_d_state,
            n_layers=s4d_n_layers,
            dropout=dropout,
            prenorm=prenorm,
            lr=lr,
            dt_min=dt_min,
            dt_max=dt_max
        )

    def forward(self, x):
        self.s4dae(x)