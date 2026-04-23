# PyTorch port of LinOSS (Linear Oscillatory State-Space model).
# Original JAX/Equinox implementation:
#   Rusch & Rus, "Oscillatory State-Space Models" (ICLR 2025 Oral)
#   https://github.com/tk-rusch/linoss
#
# Port (c) 2026 Kyungseop Yoon (kyoon@mit.edu)
#
# Key differences from S4D (see s4d.py for comparison):
#   - No (H, N, L) Vandermonde intermediate; SSM state lives in P-dim space (P << H)
#   - Memory scales as O(B * L * P) for the scan, not O(H * N * L) for the kernel
#   - The core recurrence uses torch._higher_order_ops.associative_scan —
#     an O(log L) parallel algorithm equivalent to jax.lax.associative_scan.
#     Complex tensors are split into real/imag components to avoid torchinductor's
#     current lack of complex-op code generation.
#   - Discretization is stable by construction (LinOSS-IM: implicit; LinOSS-IMEX:
#     implicit-explicit), motivated by forced harmonic oscillators

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _GLU(nn.Module):
    """Sigmoid-gated linear unit: w1(x) * sigmoid(w2(x))."""

    def __init__(self, H: int):
        super().__init__()
        self.w1 = nn.Linear(H, H)
        self.w2 = nn.Linear(H, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x) * torch.sigmoid(self.w2(x))


class LinOSSLayer(nn.Module):
    """Core LinOSS SSM layer.

    Implements a multi-dimensional forced harmonic oscillator:
        d²y/dt² + A*y = B*u

    discretized either with the Implicit Method (IM) or the Implicit-Explicit
    Method (IMEX).  The recurrence is evaluated as a sequential scan across L,
    vectorized over the batch dimension B.

    Input/output shape: (B, L, H).

    Parameters
    ----------
    ssm_size : int
        State dimension P.  The SSM state is 2P-dimensional (position + velocity
        for each of the P oscillators), so P can be small relative to H.
    H : int
        Feature/channel dimension d_model.
    discretization : str
        'IM' (implicit, default) or 'IMEX' (implicit-explicit).
    """

    def __init__(self, ssm_size: int, H: int, discretization: str = 'IM'):
        super().__init__()
        self.P = ssm_size
        self.H = H
        self.discretization = discretization

        # Oscillation frequencies: positive after relu
        self.A_raw = nn.Parameter(torch.rand(ssm_size))

        # Input matrix B: (P, H) complex, stored as (P, H, 2) real so that
        # torch.compile and torch.save work without complex-tensor issues.
        self.B_raw = nn.Parameter(
            torch.randn(ssm_size, H, 2) / math.sqrt(H)
        )

        # Output matrix C: (H, P) complex, stored as (H, P, 2) real
        self.C_raw = nn.Parameter(
            torch.randn(H, ssm_size, 2) / math.sqrt(ssm_size)
        )

        # Feedthrough D
        self.D = nn.Parameter(torch.randn(H))

        # Discretization step sizes: in (0, 1) after sigmoid
        self.step_raw = nn.Parameter(torch.rand(ssm_size))

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, L, H) real float — input sequence

        Returns:
            y: (B, L, H) real float — output sequence
        """
        B_batch, L, H = u.shape
        P = self.P

        A_diag = F.relu(self.A_raw)                          # (P,)  positive
        B = torch.view_as_complex(self.B_raw.contiguous())   # (P, H) complex64
        C = torch.view_as_complex(self.C_raw.contiguous())   # (H, P) complex64
        step = torch.sigmoid(self.step_raw)                  # (P,)  in (0,1)

        # Project input to SSM dimension: (B, L, P) complex
        # B is (P, H); B.T is (H, P) (regular transpose, no conjugate)
        # Cast u to complex so matmul with complex B works.
        Bu = u.to(dtype=torch.complex64) @ B.T               # (B, L, P)

        if self.discretization == 'IM':
            ys = self._scan_im(A_diag, Bu, step, B_batch)
        elif self.discretization == 'IMEX':
            ys = self._scan_imex(A_diag, Bu, step, B_batch)
        else:
            raise ValueError(
                f"discretization must be 'IM' or 'IMEX', got '{self.discretization}'"
            )

        # Project from SSM dimension back to H: (B, L, H) real
        # C is (H, P); C.T is (P, H)
        ssm_out = (ys @ C.T).real                            # (B, L, H)

        # Feedthrough and cast back to the original dtype (handles fp16 training)
        return ssm_out.to(u.dtype) + self.D * u

    # ------------------------------------------------------------------
    # Discretization helpers
    # ------------------------------------------------------------------

    def _scan_im(
        self,
        A_diag: torch.Tensor,   # (P,) real
        Bu: torch.Tensor,        # (B, L, P) complex
        step: torch.Tensor,      # (P,) real
        B_batch: int,
    ) -> torch.Tensor:
        """LinOSS-IM sequential scan.  Returns (B, L, P) complex."""
        P = self.P
        L = Bu.shape[1]

        schur = 1.0 / (1.0 + step ** 2 * A_diag)   # (P,)
        M11 = 1.0 - step ** 2 * A_diag * schur       # (P,)
        M12 = -step * A_diag * schur                  # (P,)
        M21 = step * schur                             # (P,)
        M22 = schur                                    # (P,)

        # Driving forces: (B, L, P) complex
        F1 = Bu * (M11 * step)   # (B, L, P) * (P,)  broadcasts
        F2 = Bu * (M21 * step)

        return self._associative_scan(M11, M12, M21, M22, F1, F2, B_batch, P)

    def _scan_imex(
        self,
        A_diag: torch.Tensor,
        Bu: torch.Tensor,
        step: torch.Tensor,
        B_batch: int,
    ) -> torch.Tensor:
        """LinOSS-IMEX sequential scan.  Returns (B, L, P) complex."""
        P = self.P

        M11 = torch.ones_like(A_diag)          # (P,)
        M12 = -step * A_diag                    # (P,)
        M21 = step                              # (P,)
        M22 = 1.0 - step ** 2 * A_diag         # (P,)

        F1 = Bu * step
        F2 = Bu * (step ** 2)

        return self._associative_scan(M11, M12, M21, M22, F1, F2, B_batch, P)

    @staticmethod
    def _associative_scan(
        M11: torch.Tensor,  # (P,) real
        M12: torch.Tensor,
        M21: torch.Tensor,
        M22: torch.Tensor,
        F1: torch.Tensor,   # (B, L, P) complex
        F2: torch.Tensor,
        B_batch: int,
        P: int,
    ) -> torch.Tensor:
        """O(log L) parallel scan — equivalent to jax.lax.associative_scan.

        The recurrence  [z1; z2]_{t+1} = M @ [z1; z2]_t + [F1; F2]_t  with
        zero initial state is solved by the associative composition:

            (Am, Af) ∘ (Bm, Bf) = (Bm @ Am,  Bm @ Af + Bf)

        where each element carries the accumulated transition matrix power and
        the accumulated driving force.  PyTorch executes this in O(log L) via
        work-efficient parallel tree-reduction.

        Complex tensors are split into real/imag components because torchinductor
        cannot yet generate CUDA code for complex operators.
        """
        from torch._higher_order_ops.associative_scan import associative_scan

        L = F1.shape[1]

        # (L, B, P) real — scan dimension first
        f1r = F1.real.permute(1, 0, 2).contiguous()
        f1i = F1.imag.permute(1, 0, 2).contiguous()
        f2r = F2.real.permute(1, 0, 2).contiguous()
        f2i = F2.imag.permute(1, 0, 2).contiguous()

        # All scan tensors must have identical shape: (L, B, P)
        # Broadcast constant M over L and B
        m11 = M11.unsqueeze(0).unsqueeze(0).expand(L, B_batch, -1)
        m12 = M12.unsqueeze(0).unsqueeze(0).expand(L, B_batch, -1)
        m21 = M21.unsqueeze(0).unsqueeze(0).expand(L, B_batch, -1)
        m22 = M22.unsqueeze(0).unsqueeze(0).expand(L, B_batch, -1)

        def combine_fn(a, b):
            am11, am12, am21, am22, af1r, af1i, af2r, af2i = a
            bm11, bm12, bm21, bm22, bf1r, bf1i, bf2r, bf2i = b

            # Accumulated matrix: cm = bm @ am  (2×2 block, P independent oscillators)
            cm11 = bm11 * am11 + bm12 * am21
            cm12 = bm11 * am12 + bm12 * am22
            cm21 = bm21 * am11 + bm22 * am21
            cm22 = bm21 * am12 + bm22 * am22

            # Accumulated forcing: cf = bm @ af + bf  (real and imag independently)
            cf1r = bm11 * af1r + bm12 * af2r + bf1r
            cf1i = bm11 * af1i + bm12 * af2i + bf1i
            cf2r = bm21 * af1r + bm22 * af2r + bf2r
            cf2i = bm21 * af1i + bm22 * af2i + bf2i

            return cm11, cm12, cm21, cm22, cf1r, cf1i, cf2r, cf2i

        _, _, _, _, _, _, z2r, z2i = associative_scan(
            combine_fn,
            (m11, m12, m21, m22, f1r, f1i, f2r, f2i),
            dim=0,
            combine_mode='generic',
        )
        # z2r, z2i: (L, B, P) → (B, L, P) complex
        return torch.complex(z2r, z2i).permute(1, 0, 2)


class LinOSSBlock(nn.Module):
    """One LinOSS residual block.

    LayerNorm → LinOSSLayer → GELU → Dropout → GLU → Dropout + residual skip.
    """

    def __init__(
        self,
        ssm_size: int,
        H: int,
        dropout: float,
        discretization: str = 'IM',
    ):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        self.ssm = LinOSSLayer(ssm_size, H, discretization)
        self.dropout = nn.Dropout(dropout)
        self.glu = _GLU(H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        skip = x
        z = self.norm(x)
        z = self.ssm(z)
        z = self.dropout(F.gelu(z))
        z = self.glu(z)
        z = self.dropout(z)
        return z + skip


# ---------------------------------------------------------------------------
# Full sequence model
# ---------------------------------------------------------------------------

class LinOSSModel(nn.Module):
    """Full LinOSS sequence model for regression / classification.

    Architecture:
        Linear encoder  (d_input → d_model)
        n_layers × LinOSSBlock
        Mean pool over L
        Linear decoder  (d_model → d_output)

    Input:  (B, L, d_input)
    Output: (B, d_output)
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 64,
        ssm_size: int = 64,
        n_layers: int = 4,
        dropout: float = 0.0,
        discretization: str = 'IM',
    ):
        super().__init__()

        self.encoder = nn.Linear(d_input, d_model)

        self.blocks = nn.ModuleList([
            LinOSSBlock(ssm_size, d_model, dropout, discretization)
            for _ in range(n_layers)
        ])

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_input)

        Returns:
            (B, d_output)
        """
        x = self.encoder(x)          # (B, L, d_model)
        for block in self.blocks:
            x = block(x)             # (B, L, d_model)
        x = x.mean(dim=1)            # (B, d_model)  — pool over sequence
        return self.decoder(x)       # (B, d_output)
