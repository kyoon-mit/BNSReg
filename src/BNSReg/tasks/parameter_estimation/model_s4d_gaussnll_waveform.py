import math
import torch

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4Model
from BNSReg.core.config import S4DModelConfig
from BNSReg.callbacks.log_metric import log_GaussianNLLLoss


class LitModelS4DGaussianNLLLoss(LitBaseTask):
    # NOTE: chirp_mass is assumed to be target_variables[0].
    # See docs/snr_loss.md for the theory behind the waveform loss.
    F_MIN      = 20.0   # Hz — lower frequency cutoff; PN phase diverges at f→0
    LAMBDA_SNR = 0.1    # weight of SNR loss relative to GaussNLL; tune as needed

    def __init__(self, model_cfg: S4DModelConfig):
        super().__init__()
        if model_cfg.d_output % 2 != 0:
            raise ValueError(f'{model_cfg.d_output=} must be even and twice the number of regressed variables.')
        self.save_hyperparameters()
        self.cfg = model_cfg
        self.n_vars = int(self.cfg.d_output/2)
        self.criterion = torch.nn.GaussianNLLLoss(reduction='mean')
        self.var_activation = torch.nn.Softplus() # Activation function for the variance for positivity enforcement
        self.model = None
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        else:
            self.model = S4Model(**self.cfg.model_kwargs())
            self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        """Pull data-side parameters from the datamodule and register as buffers.

        Buffers are saved in checkpoints and moved to the correct device automatically.
        Requires normalize_variables=True and 'chirp_mass' in target_variables.
        """
        dm  = self.trainer.datamodule
        cfg = dm.cfg
        vs  = dm.train_dataset.var_scales
        if 'chirp_mass' not in vs:
            raise RuntimeError(
                "WaveformLoss requires normalize_variables=True and 'chirp_mass' in target_variables."
            )
        eff_freq = cfg.strain_frequency / cfg.downsample_factor
        L = int((cfg.window_end - cfg.window_begin) * eff_freq)
        f = torch.fft.rfftfreq(L, d=1.0 / eff_freq)     # (F,) in Hz, F = L//2 + 1
        lo, hi = cfg.normalize_range
        self.register_buffer('_f',       f)
        self.register_buffer('_Mc_min',  torch.tensor(vs['chirp_mass'][0], dtype=torch.float32))
        self.register_buffer('_Mc_max',  torch.tensor(vs['chirp_mass'][1], dtype=torch.float32))
        self.register_buffer('_norm_lo', torch.tensor(lo,  dtype=torch.float32))
        self.register_buffer('_norm_hi', torch.tensor(hi,  dtype=torch.float32))

    def _snr_loss(self, X_sequence: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Frequency-domain de-chirp matched filter loss, maximized over t_coal.

        Unnormalizes the predicted chirp mass, builds the leading-order PN phase
        Ψ(f, Mc), de-chirps the observed strain in the Fourier domain, and measures
        how sharply the result peaks in time (= matched filter SNR).

        Args:
            X_sequence : (B, L, n_ifos) — already transposed
            mean        : (B, n_vars)   — normalized model output, grad-enabled
        Returns:
            scalar ∈ [0, 1]
        """
        B, L, _ = X_sequence.shape

        # Unnormalize Mc: model output (normalized) → physical (M_sun)
        Mc_norm = mean[:, 0]    # (B,) — chirp_mass is target_variables[0]
        Mc = self._Mc_min + (self._Mc_max - self._Mc_min) * \
             (Mc_norm - self._norm_lo) / (self._norm_hi - self._norm_lo)   # (B,)

        # PN phase: Ψ(f, Mc) = (3/128)(π Mc_sec f)^{-5/3}
        # 1 M_sun in geometric units = G M_sun / c^3 = 4.9255e-6 s
        Mc_sec = Mc * 4.9255e-6                             # (B,)
        f      = self._f                                    # (F,)
        mask   = f >= self.F_MIN                            # (F,) bool — active freq band
        f_safe = f.clamp(min=self.F_MIN)                   # (F,) avoid singularity
        psi    = (3.0 / 128.0) * \
                 (math.pi * Mc_sec[:, None] * f_safe[None, :]).pow(-5/3)   # (B, F)

        # Coherent IFO sum → RFFT → mask → de-chirp → IRFFT
        x_sum     = X_sequence.sum(dim=-1)                  # (B, L) real
        X_f       = torch.fft.rfft(x_sum, n=L)             # (B, F) complex
        X_f_band  = X_f * mask.float()                     # (B, F) zero below F_MIN
        dechirped = X_f_band * torch.polar(torch.ones_like(psi), -psi)     # (B, F) complex
        z         = torch.fft.irfft(dechirped, n=L)        # (B, L) real

        # Matched filter SNR: Cauchy-Schwarz normalized → rho ∈ [0, 1] by construction
        peak = z.abs().amax(dim=1)                          # (B,)
        S_xx = X_f_band.abs().pow(2).sum(dim=1)            # (B,) strain power in band
        S_hh = mask.float().sum()                          # scalar: template power (|H|=1)
        rho  = (peak * L / (S_xx * S_hh).sqrt().clamp(min=1e-8)).clamp(max=1.0)  # (B,)

        return (1.0 - rho).mean()

    def compute_loss(self, batch):
        X_sequence, y_target, z_observed = batch
        X_sequence  = X_sequence.transpose(2, 1)            # (B, L, n_ifos)
        outputs     = self(X_sequence)                      # (B, d_output)
        mean        = outputs[:, :self.n_vars]
        var         = self.var_activation(outputs[:, self.n_vars:])
        mse_metric  = torch.nn.MSELoss(reduction='none')
        y_indiv_mse = mse_metric(mean, y_target).T.mean(dim=1)
        nll_loss    = self.criterion(mean, y_target, var)
        snr_loss    = self._snr_loss(X_sequence, mean)      # grad flows: mean → Mc → Ψ → rho
        loss        = nll_loss + self.LAMBDA_SNR * snr_loss
        return loss, y_indiv_mse, var, mean, y_target, nll_loss, snr_loss

    def training_step(self, batch, batch_idx):
        loss, y_indiv_mse, var, mean, y_target, nll_loss, snr_loss = self.compute_loss(batch)
        log_GaussianNLLLoss(self, 'train', nll_loss, y_indiv_mse, var)
        self.log('train/snr_loss', snr_loss, on_step=False, on_epoch=True, prog_bar=False)
        return {'loss': loss, 'mean': mean.detach(), 'y_target': y_target.detach()}

    def validation_step(self, batch, batch_idx):
        loss, y_indiv_mse, var, mean, y_target, nll_loss, snr_loss = self.compute_loss(batch)
        log_GaussianNLLLoss(self, 'val', nll_loss, y_indiv_mse, var)
        self.log('val/snr_loss', snr_loss, on_step=False, on_epoch=True, prog_bar=False)
        return {'loss': loss, 'mean': mean.detach(), 'y_target': y_target.detach()}

    def test_step(self, batch, batch_idx):
        X_sequence, y_target, z_observed = batch
        X_sequence = X_sequence.transpose(2, 1)             # (B, d_input, L) -> (B, L, d_input)
        outputs    = self(X_sequence)                       # (B, d_output)
        mean  = outputs[:, :self.n_vars]
        var   = self.var_activation(outputs[:, self.n_vars:])
        sigma = torch.sqrt(var)
        return {
            'y_true':  y_target.detach().cpu(),
            'y_pred':  mean.detach().cpu(),
            'y_sigma': sigma.detach().cpu(),
        }
