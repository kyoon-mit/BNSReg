import math

import torch
import torch.nn.functional as F
from torch import optim

from BNSReg.losses.denoising import DynamicMixtureLoss
from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d_seq2seq import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig


class LitModelS4DMixtureLoss(LitBaseTask):
    """S4D denoiser with a dynamically-scheduled MSE + spectral mixture loss.

    The loss is:
        L = alpha(t) * MSE(pred, target)
          + (1 - alpha(t)) * MSE(|FFT(pred)|, |FFT(target)|)

    where alpha(t) is updated each epoch according to the chosen schedule.
    Starting pure-spectral (alpha=0) lets the model first find the correct
    frequency envelope; ramping toward MSE (alpha=1) then forces fine
    waveform alignment.

    Args:
        cfg:            S4D model configuration.
        alpha_start:    Initial alpha value (default 0.0 = pure spectral).
        alpha_end:      Final alpha value   (default 1.0 = pure MSE).
        alpha_schedule: One of 'constant', 'linear', 'cosine'.
        alpha_epochs:   Number of epochs over which to anneal alpha.
    """

    def __init__(
        self,
        cfg:            S4DModelConfig,
        alpha_start:    float = 0.0,
        alpha_end:      float = 1.0,
        alpha_schedule: str   = 'linear',
        alpha_epochs:   int   = 1000,
    ):
        super().__init__()
        self.cfg            = cfg
        self.alpha_start    = alpha_start
        self.alpha_end      = alpha_end
        self.alpha_schedule = alpha_schedule
        self.alpha_epochs   = alpha_epochs
        self.criterion      = DynamicMixtureLoss(alpha=alpha_start)
        self.model          = None
        self.configure_model()

    # ------------------------------------------------------------------
    # Alpha scheduling
    # ------------------------------------------------------------------

    def _alpha(self, epoch: int) -> float:
        if self.alpha_schedule == 'constant':
            return float(self.alpha_start)
        progress = min(epoch / max(self.alpha_epochs, 1), 1.0)
        if self.alpha_schedule == 'linear':
            t = progress
        elif self.alpha_schedule == 'cosine':
            t = (1.0 - math.cos(math.pi * progress)) / 2.0
        else:
            raise ValueError(f"Unknown alpha_schedule '{self.alpha_schedule}'. "
                             "Choose 'constant', 'linear', or 'cosine'.")
        return float(self.alpha_start + t * (self.alpha_end - self.alpha_start))

    def on_train_epoch_start(self):
        alpha = self._alpha(self.trainer.current_epoch)
        self.criterion.alpha = alpha
        self.log('train/alpha', alpha, on_epoch=True, prog_bar=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4ModelSeq2Seq(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _step(self, batch, stage: str):
        input, target = batch        # (B, n_ifos, L)
        x = input.transpose(1, 2)   # (B, L, d_input)
        t = target.transpose(1, 2)  # (B, L, d_output)
        out = self(x)               # (B, L, d_output)

        loss = self.criterion(out, t)
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log component losses separately for diagnostics
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
