"""Loss functions for GW signal denoising tasks.

All losses operate on tensors with shape (B, L, n_ifos) — i.e. the
sequence dimension is dim=1 — matching the convention of S4ModelSeq2Seq.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LOSS_FUNCTIONS = (
    'mse', 'mae', 'rmse', 'huber', 'smooth_l1', 'smae', 'logcosh',
)


def loss_helper(
    loss: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Scalar discrepancy between two tensors, selected by name.

    All options are means over every element, so they stay comparable in
    magnitude and can be swapped without retuning the mixing weight.

        mse        squared error. Standard, but one large residual dominates
                   many small ones.
        mae        absolute error. Every residual counts in proportion to its
                   size, so outliers pull less.
        rmse       square root of mse, back in the units of the data itself.
        huber      squared below `delta`, absolute above it. Behaves like mse
                   for small residuals and mae for large ones.
        smooth_l1  same shape as huber but scaled down by `delta`; this is
                   torch's own convention. 'smae' is accepted as an alias.
        logcosh    smooth throughout, asymptotically absolute error, with no
                   kink at zero the way huber has.

    Args:
        loss: one of LOSS_FUNCTIONS.
        pred, target: same-shaped tensors.
        delta: transition point for 'huber' / 'smooth_l1'. Ignored otherwise.
    """
    if loss == 'mse':
        return F.mse_loss(pred, target)
    if loss == 'mae':
        return F.l1_loss(pred, target)
    if loss == 'rmse':
        # clamp keeps the sqrt gradient finite when the error reaches zero.
        return F.mse_loss(pred, target).clamp_min(1e-12).sqrt()
    if loss == 'huber':
        return F.huber_loss(pred, target, delta=delta)
    if loss in ('smooth_l1', 'smae'):
        return F.smooth_l1_loss(pred, target, beta=delta)
    if loss == 'logcosh':
        # log(cosh(d)) written as |d| + log1p(exp(-2|d|)) - log(2), which does
        # not overflow the way cosh does for large residuals.
        d = (pred - target).abs()
        return (d + torch.log1p(torch.exp(-2.0 * d)) - math.log(2.0)).mean()
    raise ValueError(
        f'Unknown loss {loss!r}. Choose one of {LOSS_FUNCTIONS}.'
    )


class DynamicMixtureLoss(nn.Module):
    """Mixture of time-domain MSE and frequency-domain spectral loss.

    The mixing weight ``alpha`` is externally mutable so that the training
    task can schedule it epoch-by-epoch (see LitModelS4DMixtureLoss).

    Args:
        alpha: Initial weight for the MSE term (0 = pure spectral,
               1 = pure MSE). Updated in-place by the task each epoch.
        density: If True (default), divide each term by its own target's mean
            squared energy before mixing, turning both into dimensionless
            relative errors. Without this, the two terms can differ by
            orders of magnitude (spectral energy summed over frequency bins
            vs. time-domain MSE), making `alpha` not actually mean "equal
            weight" at 0.5. Keep True unless reproducing a run predating this
            option.
        time_loss: Which discrepancy to use for the time-domain term.
            One of LOSS_FUNCTIONS; see `loss_helper` for what each
            one does. Defaults to 'mse'.
        spectral_loss: What the frequency-domain term compares. 'mse'
            (default) compares magnitudes directly; 'msle' compares
            log(magnitude + log_floor), which compresses the dynamic range so
            loud bins stop dominating and quiet in-band structure still
            contributes. Any name in LOSS_FUNCTIONS is also accepted and
            applies that discrepancy to plain magnitudes, so e.g. 'mae' means
            absolute error on |FFT|.
        log_floor: Floor inside the log for 'msle': log(|FFT| + log_floor).
            Controls how much near-zero bins get compressed: a smaller floor
            weights quiet bins more heavily. Ignored unless spectral_loss is
            'msle'.
        huber_delta: Transition point for 'huber' and 'smooth_l1', shared by
            both terms. Residuals smaller than this are treated like squared
            error, larger ones like absolute error.

    Note on `density`: each term is divided by the same discrepancy measured
    between the target and zero, i.e. the loss you would get by predicting
    nothing. That keeps the ratio dimensionless for every option, including
    the ones like 'mae' and 'rmse' that are not in squared units.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        density: bool = True,
        time_loss: str = 'mse',
        spectral_loss: str = 'mse',
        log_floor: float = 1e-3,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {alpha}')
        if time_loss not in LOSS_FUNCTIONS:
            raise ValueError(
                f'time_loss must be one of {LOSS_FUNCTIONS}, '
                f'got {time_loss!r}'
            )
        if spectral_loss not in LOSS_FUNCTIONS + ('msle',):
            raise ValueError(
                f"spectral_loss must be 'msle' or one of {LOSS_FUNCTIONS}, "
                f'got {spectral_loss!r}'
            )
        if log_floor <= 0.0:
            raise ValueError(f'log_floor must be > 0, got {log_floor}')
        if huber_delta <= 0.0:
            raise ValueError(f'huber_delta must be > 0, got {huber_delta}')
        self.alpha = alpha  # mutable; updated each epoch by the task
        self.density = density
        self.time_loss = time_loss
        self.spectral_loss = spectral_loss
        self.log_floor = log_floor
        self.huber_delta = huber_delta

    def _term(self, loss: str, pred: torch.Tensor, target: torch.Tensor):
        """One term of the mixture, optionally made dimensionless."""
        value = loss_helper(loss, pred, target, self.huber_delta)
        if self.density:
            # Same discrepancy against a zero prediction, so the ratio is
            # "error relative to predicting nothing" whatever `loss` is.
            scale = loss_helper(
                loss, torch.zeros_like(target), target, self.huber_delta
            )
            value = value / scale.clamp_min(1e-8)
        return value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: (B, L, n_ifos)
        Returns:
            scalar loss
        """
        time_term = self._term(self.time_loss, pred, target)

        pred_mag   = torch.fft.rfft(pred,   dim=1).abs()   # (B, L//2+1, n_ifos)
        target_mag = torch.fft.rfft(target, dim=1).abs()

        # 'msle' is squared error on log magnitudes; log_floor keeps the log
        # finite for the zero magnitudes empty bins produce.
        if self.spectral_loss == 'msle':
            pred_mag   = torch.log(pred_mag   + self.log_floor)
            target_mag = torch.log(target_mag + self.log_floor)
            spectral_term = self._term('mse', pred_mag, target_mag)
        else:
            spectral_term = self._term(self.spectral_loss, pred_mag, target_mag)

        return self.alpha * time_term + (1.0 - self.alpha) * spectral_term


class SpectrogramLoss(nn.Module):
    """Spectral convergence + log magnitude loss on STFT magnitudes.

    Unlike the single full-window FFT used by DynamicMixtureLoss, an STFT
    resolves *when* each frequency appears, so this sees the chirp's sweep
    rather than only its integrated content.

        spectral convergence  ||S_t - S_p||_F / ||S_t||_F, scale free, driven
                              by the loud parts of the spectrogram.
        log magnitude         mean |log S_t - log S_p|, which lets the quiet
                              parts matter too.

    Both use magnitudes only and so carry no phase information, meaning this
    does not by itself constrain time alignment.

    Args:
        n_fft: STFT window length in samples. Smaller resolves time better,
            larger resolves frequency better.
        hop_length: samples between consecutive STFT frames.
        eps: floor inside the log, and on the spectral convergence
            denominator so a near-silent target cannot blow up the gradient.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        eps: float = 1e-8
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        # Buffer so the window follows the module across .to(device).
        self.register_buffer('window', torch.hann_window(n_fft))

    def _spec(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, n_ifos) -> (B * n_ifos, freqs, frames) magnitudes."""
        b, l, c = x.shape
        x = x.permute(0, 2, 1).reshape(b * c, l)
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            # center=True is required here: with center=False the final frame
            # is dropped and a merger sitting on the last sample contributes
            # essentially nothing to the loss.
            center=True,
            return_complex=True,
        ).abs()

    def terms(self, pred: torch.Tensor, target: torch.Tensor):
        """Returns (spectral_convergence, log_magnitude), both scalars."""
        sp, st = self._spec(pred), self._spec(target)
        convergence = (
            torch.linalg.norm(st - sp) / torch.linalg.norm(st).clamp_min(self.eps)
        )
        log_magnitude = (
            torch.log(st + self.eps) - torch.log(sp + self.eps)
        ).abs().mean()
        return convergence, log_magnitude

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: (B, L, n_ifos)
        Returns:
            scalar loss
        """
        return sum(self.terms(pred, target))


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
