import torch
import torch.nn.functional as F
from torch import optim

from BNSReg.losses.denoising import BandWeightedSpectralLoss
from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig


class LitModelS4DBandWeighted(LitBaseTask):
    """S4D denoiser with a frequency-band-weighted spectral loss.

    The standard PSD loss weights all frequencies equally, so broadband
    noise residuals overwhelm the narrow GW signal peak.  This task uses
    a soft sigmoid band mask to amplify the gradient in the signal band
    (typically 20–500 Hz for BNS) and suppress out-of-band noise.

    Loss:
        W[f] = sigmoid((f - f_low) / bw) * sigmoid((f_high - f) / bw)
        L    = mean_f W[f] * |FFT(pred)[f] - FFT(target)[f]|²

    Args:
        cfg:         S4D model configuration.
        seq_len:     Window length in samples (window_end - window_begin)
                     * strain_frequency / downsample_factor.
        sample_rate: Effective sample rate in Hz
                     (strain_frequency / downsample_factor).
        f_low:       Lower edge of signal band in Hz.
        f_high:      Upper edge of signal band in Hz.
        bandwidth:   Sigmoid transition width in Hz (smaller = sharper).
    """

    def __init__(
        self,
        cfg:         S4DModelConfig,
        seq_len:     int   = 1024,
        sample_rate: float = 256.0,
        f_low:       float = 20.0,
        f_high:      float = 500.0,
        bandwidth:   float = 10.0,
    ):
        super().__init__()
        self.cfg       = cfg
        self.criterion = BandWeightedSpectralLoss(
            seq_len=seq_len,
            sample_rate=sample_rate,
            f_low=f_low,
            f_high=f_high,
            bandwidth=bandwidth,
        )
        self.model = None
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4ModelSeq2Seq(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        input, target = batch        # (B, n_ifos, L)
        x = input.transpose(1, 2)   # (B, L, d_input)
        t = target.transpose(1, 2)  # (B, L, d_output)
        out = self(x)               # (B, L, d_output)

        loss = self.criterion(out, t)
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Also log unweighted MSE and spectral for comparison
        mse_val  = F.mse_loss(out, t)
        pred_mag = torch.fft.rfft(out, dim=1).abs()
        tgt_mag  = torch.fft.rfft(t,   dim=1).abs()
        spec_val = F.mse_loss(pred_mag, tgt_mag)
        self.log(f'{stage}/mse',      mse_val,  on_step=False, on_epoch=True)
        self.log(f'{stage}/spectral', spec_val, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
