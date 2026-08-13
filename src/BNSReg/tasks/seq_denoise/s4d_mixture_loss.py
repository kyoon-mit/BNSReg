import math

import torch
import torch.nn.functional as F
from torch import optim

from BNSReg.losses.denoising import DynamicMixtureLoss
from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig


class LitModelS4DMixtureLoss(LitBaseTask):
    """S4D denoiser with a dynamically-scheduled MSE + spectral mixture loss.

    The loss is:
        L = alpha(t) * MSE(pred, target)
          + (1 - alpha(t)) * spectral_loss(|FFT(pred)|, |FFT(target)|)

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
        density:        If True (default), normalize the MSE and spectral terms
                         by their own target energy before mixing, so alpha
                         reflects a true relative weight regardless of the
                         raw scale mismatch between the two domains.
        time_domain_loss: Which discrepancy the time-domain term uses: 'mse',
                         'mae', 'rmse', 'huber', 'smooth_l1' ('smae'), or
                         'logcosh'. See BNSReg.losses.denoising for what each
                         one does.
        spectral_loss:  What the frequency-domain term compares. 'mse'
                         (default) or 'msle', the latter comparing
                         log(|FFT| + log_floor) to compress the dynamic range.
                         Any time_domain_loss name also works and applies that
                         discrepancy to plain |FFT|.
        log_floor:      Floor inside the log for 'msle':
                         log(|FFT| + log_floor). Controls how much near-zero
                         bins get compressed: a smaller floor weights quiet
                         bins more heavily. Ignored unless spectral_loss is
                         'msle'.
        huber_delta:    Transition point for 'huber' and 'smooth_l1', shared
                         by both terms.
    """

    def __init__(
        self,
        cfg:              S4DModelConfig,
        alpha_start:      float = 0.0,
        alpha_end:        float = 1.0,
        alpha_schedule:   str   = 'linear',
        alpha_epochs:     int   = 1000,
        density:          bool  = True,
        time_domain_loss: str   = 'mse',
        spectral_loss:    str   = 'mse',
        log_floor:        float = 1e-3,
        huber_delta:      float = 1.0,
    ):
        super().__init__()
        # All init args land in self.hparams and are restored from checkpoints;
        # read them from there rather than shadowing each one on self.
        self.save_hyperparameters()
        self.cfg       = cfg
        self.criterion = DynamicMixtureLoss(
            alpha=alpha_start,
            density=density,
            time_loss=time_domain_loss,
            spectral_loss=spectral_loss,
            log_floor=log_floor,
            huber_delta=huber_delta,
        )
        self.model     = None
        self.configure_model()

    # ------------------------------------------------------------------
    # Alpha scheduling
    # ------------------------------------------------------------------

    def _alpha(self, epoch: int) -> float:
        hp = self.hparams
        if hp.alpha_schedule == 'constant':
            return float(hp.alpha_start)
        progress = min(epoch / max(hp.alpha_epochs, 1), 1.0)
        if hp.alpha_schedule == 'linear':
            t = progress
        elif hp.alpha_schedule == 'cosine':
            t = (1.0 - math.cos(math.pi * progress)) / 2.0
        else:
            raise ValueError(f"Unknown alpha_schedule '{hp.alpha_schedule}'. "
                             "Choose 'constant', 'linear', or 'cosine'.")
        return float(hp.alpha_start + t * (hp.alpha_end - hp.alpha_start))

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
        # S4ModelSeq2Seq.forward takes (B, d_input, L) and transposes internally,
        # unlike S4Model.forward which takes (B, L, d_input); pass the batch as is.
        noisy, target = batch       # (B, n_ifos, L)
        out = self(noisy)           # (B, d_output, L)

        # DynamicMixtureLoss expects (B, L, n_ifos); transpose for the loss only.
        out_t = out.transpose(1, 2)
        t = target.transpose(1, 2)

        loss = self.criterion(out_t, t)
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log the two terms separately for diagnostics. Reuse the criterion's
        # own _term so these always match what is actually optimized.
        mse_val  = self.criterion._term(self.hparams.time_domain_loss, out_t, t)
        pred_mag = torch.fft.rfft(out_t, dim=1).abs()
        tgt_mag  = torch.fft.rfft(t,   dim=1).abs()
        if self.hparams.spectral_loss == 'msle':
            pred_mag = torch.log(pred_mag + self.hparams.log_floor)
            tgt_mag  = torch.log(tgt_mag  + self.hparams.log_floor)
            spec_val = self.criterion._term('mse', pred_mag, tgt_mag)
        else:
            spec_val = self.criterion._term(
                self.hparams.spectral_loss, pred_mag, tgt_mag
            )
        self.log(f'{stage}/time_domain_mse_loss', mse_val,  on_step=False, on_epoch=True)
        self.log(f'{stage}/spectral_loss',        spec_val, on_step=False, on_epoch=True)

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
