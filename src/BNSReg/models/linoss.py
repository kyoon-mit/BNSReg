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
#   - The core recurrence is a sequential loop (batch-vectorized over B), compiled
#     via torch.compile into a single fused graph per sequence length
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

        return self._sequential_scan(M11, M12, M21, M22, F1, F2, B_batch, P)

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

        return self._sequential_scan(M11, M12, M21, M22, F1, F2, B_batch, P)

    @staticmethod
    def _sequential_scan(
        M11: torch.Tensor,  # (P,)
        M12: torch.Tensor,
        M21: torch.Tensor,
        M22: torch.Tensor,
        F1: torch.Tensor,   # (B, L, P) complex
        F2: torch.Tensor,
        B_batch: int,
        P: int,
    ) -> torch.Tensor:
        """Batch-vectorized sequential recurrence.

        State: z = [z1, z2] — position (z1) and velocity (z2) of P oscillators.
        z1_{t+1} = M11 * z1_t + M12 * z2_t + F1_t
        z2_{t+1} = M21 * z1_t + M22 * z2_t + F2_t
        Output: the velocity z2 sequence, shape (B, L, P) complex.

        The loop runs L steps but each step updates all B batch elements
        simultaneously, so the GPU works on (B, P) tensors per step.
        torch.compile fuses all L steps into one kernel per forward pass
        (no per-step Python overhead at inference/training time).
        """
        L = F1.shape[1]
        device = F1.device

        z1 = torch.zeros(B_batch, P, dtype=F1.dtype, device=device)
        z2 = torch.zeros(B_batch, P, dtype=F1.dtype, device=device)

        ys: list[torch.Tensor] = []
        for t in range(L):
            z1_new = M11 * z1 + M12 * z2 + F1[:, t]
            z2_new = M21 * z1 + M22 * z2 + F2[:, t]
            z1 = z1_new
            z2 = z2_new
            ys.append(z2)

        return torch.stack(ys, dim=1)  # (B, L, P) complex


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
