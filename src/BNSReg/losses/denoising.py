"""Loss functions for GW signal denoising tasks.

All losses operate on tensors with shape (B, L, n_ifos) — i.e. the
sequence dimension is dim=1 — matching the convention of S4ModelSeq2Seq.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMixtureLoss(nn.Module):
    """Mixture of time-domain MSE and frequency-domain spectral loss.

    The mixing weight ``alpha`` is externally mutable so that the training
    task can schedule it epoch-by-epoch (see LitModelS4DMixtureLoss).

    Args:
        alpha: Initial weight for the MSE term (0 = pure spectral,
               1 = pure MSE). Updated in-place by the task each epoch.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {alpha}')
        self.alpha = alpha  # mutable; updated each epoch by the task

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: (B, L, n_ifos)
        Returns:
            scalar loss
        """
        mse = F.mse_loss(pred, target)

        pred_mag   = torch.fft.rfft(pred,   dim=1).abs()   # (B, L//2+1, n_ifos)
        target_mag = torch.fft.rfft(target, dim=1).abs()
        spectral   = F.mse_loss(pred_mag, target_mag)

        return self.alpha * mse + (1.0 - self.alpha) * spectral


class BandWeightedSpectralLoss(nn.Module):
    """Spectral MSE weighted by a soft frequency-band mask.

    Amplifies the loss contribution from frequency bins inside the GW
    signal band (default ~20–500 Hz) and suppresses out-of-band bins,
    counteracting the tendency to fit broadband noise at the expense of
    the narrow signal peak.

    The mask is:
        W[f] = sigmoid((f - f_low) / bw) * sigmoid((f_high - f) / bw)

    and the loss is:
        L = mean_f W[f] * |FFT(pred)[f] - FFT(target)[f]|²

    Args:
        seq_len:     Number of time samples (after windowing & downsampling).
        sample_rate: Effective sample rate in Hz (strain_frequency / downsample_factor).
        f_low:       Lower edge of signal band in Hz.
        f_high:      Upper edge of signal band in Hz.
        bandwidth:   Sigmoid transition width in Hz (smaller = sharper edge).
    """

    def __init__(
        self,
        seq_len:     int,
        sample_rate: float,
        f_low:       float = 20.0,
        f_high:      float = 500.0,
        bandwidth:   float = 10.0,
    ):
        super().__init__()
        freqs = torch.fft.rfftfreq(seq_len, d=1.0 / sample_rate)   # Hz, shape (L//2+1,)
        W = (
            torch.sigmoid((freqs - f_low)  / bandwidth) *
            torch.sigmoid((f_high - freqs) / bandwidth)
        )
        self.register_buffer('W', W)   # (L//2+1,)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: (B, L, n_ifos)
        Returns:
            scalar loss
        """
        pred_mag   = torch.fft.rfft(pred,   dim=1).abs()   # (B, L//2+1, n_ifos)
        target_mag = torch.fft.rfft(target, dim=1).abs()

        diff2 = (pred_mag - target_mag).pow(2)              # (B, L//2+1, n_ifos)
        W = self.W.unsqueeze(0).unsqueeze(-1)               # (1,  L//2+1,  1)
        return (diff2 * W).mean()


class TripletSpectralLoss(nn.Module):
    """Contrastive triplet loss in the PSD domain.

    Pushes the denoised output (anchor) toward the clean signal (positive)
    and away from pure background noise (negative), addressing the
    fundamental scale-mismatch problem where MSE/PSD losses collapse to zero.

    Loss:
        L_triplet = max(0, d(anchor, pos) - d(anchor, neg) + margin)
        L_total   = L_triplet + lambda_rec * MSE(anchor, pos)

    where d(·,·) is the mean squared difference between one-sided power
    spectra.  Working in PSD space makes the metric invariant to phase and
    naturally compares spectral shape rather than exact waveform alignment.

    Args:
        margin:     Triplet margin (start small, e.g. 0.1, and tune).
        lambda_rec: Weight on the reconstruction MSE term for stability.
    """

    def __init__(self, margin: float = 0.1, lambda_rec: float = 1.0):
        super().__init__()
        self.margin     = margin
        self.lambda_rec = lambda_rec

    @staticmethod
    def _psd(x: torch.Tensor) -> torch.Tensor:
        """One-sided power spectrum.  x: (B, L, n_ifos) → (B, L//2+1, n_ifos)."""
        return torch.fft.rfft(x, dim=1).abs().pow(2)

    def forward(
        self,
        anchor:   torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor:   denoised model output  (B, L, n_ifos)
            positive: clean signal           (B, L, n_ifos)
            negative: background-only noise  (B, L, n_ifos)
        Returns:
            scalar loss
        """
        a_psd = self._psd(anchor)
        p_psd = self._psd(positive)
        n_psd = self._psd(negative)

        dist_pos = (a_psd - p_psd).pow(2).mean()
        dist_neg = (a_psd - n_psd).pow(2).mean()
        triplet  = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        rec      = F.mse_loss(anchor, positive)

        return triplet + self.lambda_rec * rec
