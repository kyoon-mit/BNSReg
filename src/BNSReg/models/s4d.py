# SPDX-License-Identifier: Apache-2.0
#
# This file is a derivative work of the S4 repository:
#   https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
#   https://github.com/state-spaces/s4/blob/main/examples.py
#
# Copyright (c) 2023 The S4 Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications copyright (c) 2026 Kyungseop Yoon (kyoon@mit.edu)
# This project (BNSReg) is released under the MIT License; see LICENSE.
# Third-party attributions are listed in NOTICE.
#
# Modifications made on 2026-01-14:
#   - Adjusted the import path for DropoutNd to match this repository:
#       from `src.models.nn` -> `BNSReg.functions.dropout`
#   - `BNSReg.functions.dropout` is itself derived from S4 and retains
#     Apache-2.0 licensing and attribution.
#   - Removed unused imports and simplified the code for pedagogical clarity.
#   - Replaced the local `dropout_fn` alias with direct use of `DropoutNd`
#     in S4Model.
#   - Added arguments to S4Model and passed it to S4D initialization
#     (without further modification to the upstream kernel logic).
#   - Replaced use of Python complex literals (e.g. `1j`) with
#     `torch.complex(...)` to ensure compatibility with `torch.compile`.
#
# Modifications made on 2026-04-18:
#   - Added optional `input_norm` flag to S4Model that applies per-channel
#     InstanceNorm1d (affine=True) along the time axis before the encoder,
#     controlled by the `input_norm` field in S4DModelConfig.

"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes.

See docs/ssm_memory_scaling.md for a detailed analysis of why S4D's Vandermonde kernel
materialises an (H, N//2, L) intermediate that makes large-L training infeasible on
consumer GPUs, and how LinOSS (models/linoss.py) avoids this bottleneck.
"""

import math
import torch
import torch.nn as nn
from einops import repeat
from BNSReg.models.resnet1d import ResNet1D

from typing import Optional

from BNSReg.functions.dropout import DropoutNd

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        # ORIGINAL CODE
        # A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)
        # MODIFIED CODE FOR torch.compile SAFETY
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified

class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        d_state=64,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=None,
        dt_min=0.001,
        dt_max=0.1,
        input_norm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Optional per-channel instance norm along the time axis; normalises amplitude scale
        # without destroying relative phase/frequency content between channels.
        self._input_norm = nn.InstanceNorm1d(d_input, affine=True) if input_norm else None

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, d_state=d_state, dropout=dropout, transposed=True,
                    dt_min=dt_min, dt_max=dt_max, lr=lr)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        if self._input_norm is not None:
            x = self._input_norm(x.transpose(1, 2)).transpose(1, 2)  # InstanceNorm1d expects (B, C, L)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

class S4ModelSeq2Seq(S4Model):
    """S4D sequence-to-sequence model."""

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        prenorm: bool = True,
        num_groups: Optional[int] = None,
        lr: Optional[float] = None,
        input_norm: bool = False,
    ):
        super().__init__(
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            dt_min=dt_min,
            dt_max=dt_max,
            lr=lr,
            input_norm=input_norm,
        )
        self.prenorm = prenorm
        self._groupnorm = num_groups is not None
        if self._groupnorm:
            self.norms = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
                    for _ in range(n_layers)
                ]
            )

    def _apply_norm(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # x is (B, d_model, L). GroupNorm1D normalizes channels directly;
        # LayerNorm needs the (B, L, d_model) view.
        if self._groupnorm:
            return norm(x)
        return norm(x.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_input, L)

        Returns:
            (B, d_output, L)
        """
        x = x.transpose(-1, -2)  # (B, L, d_input)
        x = self.encoder(x)  # (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, d_model, L)
        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropouts, strict=True
        ):
            # S4D.forward returns (y, state); the state is unused here.
            if self.prenorm:
                z = self._apply_norm(norm, x)
                z, _ = layer(z)
                z = dropout(z)
                x = x + z
            else:
                z, _ = layer(x)
                z = dropout(z)
                x = self._apply_norm(norm, z + x)
        x = x.transpose(-1, -2)  # (B, L, d_model)
        x = self.decoder(x)  # (B, L, d_output)
        return x.transpose(-1, -2)  # (B, d_output, L)

class S4ModelResNetMLPDecoder(S4Model):
    """S4Model backbone with a ResNet1D + MLP readout head in place of the
    mean-pool + linear decoder. Ported from aframe's S4ModelResNetMLPDecoder.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        prenorm: bool = False,
        lr: Optional[float] = None,
        input_norm: bool = False,
        resnet_layers: tuple[int, ...] = (2, 2, 2),
        resnet_latent_dim: int = 64,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__(
            d_input=d_input, d_output=d_output, d_model=d_model, d_state=d_state,
            n_layers=n_layers, dropout=dropout, dt_min=dt_min, dt_max=dt_max,
            lr=lr, input_norm=input_norm,
        )
        self.prenorm = prenorm
        self.resnet = ResNet1D(in_channels=d_model, layers=list(resnet_layers), classes=resnet_latent_dim)
        width = resnet_latent_dim
        mlp: list[nn.Module] = []
        for _ in range(mlp_depth):
            mlp += [nn.Linear(width, mlp_width), nn.GELU()]
            width = mlp_width
        mlp.append(nn.Linear(width, d_output))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_input) -> (B, d_output)"""
        if self._input_norm is not None:
            x = self._input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.encoder(x)  # (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts, strict=True):
            # S4D.forward returns (y, state); the state is unused here.
            if self.prenorm:
                z = norm(x.transpose(-1, -2)).transpose(-1, -2)
                z, _ = layer(z)
                z = dropout(z)
                x = x + z
            else:
                z, _ = layer(x)
                z = dropout(z)
                x = norm((z + x).transpose(-1, -2)).transpose(-1, -2)
        h = self.resnet(x)  # (B, d_model, L) -> (B, resnet_latent_dim)
        return self.mlp(h)  # (B, d_output)