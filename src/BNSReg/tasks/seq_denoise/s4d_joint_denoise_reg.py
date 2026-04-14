import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d_seq2seq import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig


class LitModelS4DJointDenoiseReg(LitBaseTask):
    """Joint denoising + regression via shared S4D encoder with two heads.

    Architecture:
        shared backbone (S4ModelSeq2Seq, minus its own decoder)
          ├─ reconstruction head: Linear(d_model, n_ifos) → (B, L, n_ifos)
          │      MSE vs. clean signal
          └─ regression head:    mean-pool → Linear(d_model, n_vars*2) → (B, n_vars*2)
                 GaussianNLL vs. physical parameters (mean + log-variance)

    Staged training:
        The regression head is gated off for the first ``warmup_epochs``
        epochs (loss weight = 0), so the model first learns reconstruction.
        After warmup the regression weight ramps linearly to ``max_reg_weight``
        over ``reg_ramp_epochs`` epochs, forcing the shared features to
        preserve the information needed to predict physical parameters.

    Why this prevents collapse:
        Even if reconstruction alone collapses toward zero, the regression
        gradient requires non-zero feature activations correlated with the
        chirp signal structure — creating an opposing gradient that prevents
        the model from erasing the signal.

    Args:
        cfg:              S4D backbone configuration.
        n_vars:           Number of regression target variables.
        warmup_epochs:    Epochs of pure reconstruction before regression starts.
        max_reg_weight:   Final regression loss weight.
        reg_ramp_epochs:  Epochs to ramp from 0 to max_reg_weight after warmup.

    Data:
        Requires LitBNSDataJointDenoise; batches are
        (x_noisy, x_clean, y_target, z_observed).
    """

    def __init__(
        self,
        cfg:             S4DModelConfig,
        n_vars:          int   = 1,
        warmup_epochs:   int   = 100,
        max_reg_weight:  float = 1.0,
        reg_ramp_epochs: int   = 200,
    ):
        super().__init__()
        self.cfg             = cfg
        self.n_vars          = n_vars
        self.warmup_epochs   = warmup_epochs
        self.max_reg_weight  = max_reg_weight
        self.reg_ramp_epochs = reg_ramp_epochs

        self.backbone       = None
        self.rec_head       = None
        self.reg_head       = None
        self.gaussnll       = nn.GaussianNLLLoss(reduction='mean', full=False, eps=1e-6)
        self.configure_model()

    # ------------------------------------------------------------------
    # Regression weight schedule
    # ------------------------------------------------------------------

    def _reg_weight(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        ramp = min(
            (epoch - self.warmup_epochs) / max(self.reg_ramp_epochs, 1),
            1.0,
        )
        return float(self.max_reg_weight * ramp)

    def on_train_epoch_start(self):
        w = self._reg_weight(self.trainer.current_epoch)
        self.log('train/reg_weight', w, on_epoch=True, prog_bar=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def configure_model(self):
        if self.backbone is not None:
            return
        d_model = self.cfg.d_model
        n_ifos  = self.cfg.d_input

        self.backbone = S4ModelSeq2Seq(**self.cfg.model_kwargs())
        self.backbone = torch.compile(self.backbone)

        # Reconstruction head replaces the backbone's own decoder
        self.rec_head = nn.Linear(d_model, n_ifos)

        # Regression head: mean-pool over time, then predict mean + log-var
        self.reg_head = nn.Linear(d_model, self.n_vars * 2)

    def forward(self, x):
        """Standard forward: returns only the reconstruction output.

        Args:
            x: (B, L, d_input)
        Returns:
            rec: (B, L, n_ifos)
        """
        _, features = self.backbone(x, return_features=True)  # (B, L, d_model)
        return self.rec_head(features)                         # (B, L, n_ifos)

    def _forward_both(self, x):
        """Forward returning both heads.

        Returns:
            rec:  (B, L, n_ifos)
            mean: (B, n_vars)
            var:  (B, n_vars)  — positive, via softplus
        """
        _, features = self.backbone(x, return_features=True)  # (B, L, d_model)
        rec         = self.rec_head(features)                  # (B, L, n_ifos)
        pooled      = features.mean(dim=1)                     # (B, d_model)
        reg_out     = self.reg_head(pooled)                    # (B, n_vars*2)
        mean        = reg_out[:, :self.n_vars]
        var         = F.softplus(reg_out[:, self.n_vars:])
        return rec, mean, var

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _step(self, batch, stage: str):
        x_noisy, x_clean, y_target, _ = batch  # (B, n_ifos, L), (B, n_ifos, L), (B, n_vars), …

        x   = x_noisy.transpose(1, 2)          # (B, L, d_input)
        t   = x_clean.transpose(1, 2)          # (B, L, n_ifos)

        rec, mean, var = self._forward_both(x)

        loss_rec = F.mse_loss(rec, t)
        loss_reg = self.gaussnll(mean, y_target, var)

        reg_w    = self._reg_weight(self.trainer.current_epoch)
        loss     = loss_rec + reg_w * loss_reg

        self.log(f'{stage}/loss',     loss,     on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{stage}/loss_rec', loss_rec, on_step=False, on_epoch=True)
        self.log(f'{stage}/loss_reg', loss_reg, on_step=False, on_epoch=True)

        for i in range(self.n_vars):
            self.log(f'{stage}/sigma_{i}',
                     var[:, i].mean().sqrt(),
                     on_step=False, on_epoch=True)

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
