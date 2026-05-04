import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d_seq2seq import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig
from BNSReg.callbacks.log_metric import log_GaussianNLLLoss
from BNSReg.utils.schedulers import WarmupCosineAnnealingWarmRestarts

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
        base_lr:          Base learning rate for default parameter group.
        weight_decay:     AdamW weight decay.
        T_0:              CosineAnnealingWarmRestarts period.
        T_mult:           Period multiplier after each restart.
        eta_min:          Minimum LR at cosine trough.
        warmup_start_factor: LR scale at start of warmup (relative to base_lr).

    Data:
        Requires LitBNSDataJointDenoise; batches are
        (x_noisy, x_clean, y_target, z_observed).
    """

    def __init__(
        self,
        cfg:                  S4DModelConfig,
        n_vars:               int   = 1,
        warmup_epochs:        int   = 100,
        max_reg_weight:       float = 1.0,
        reg_ramp_epochs:      int   = 200,
        base_lr:              float = 1e-3,
        weight_decay:         float = 1e-2,
        T_0:                  int   = 16,
        T_mult:               int   = 1,
        eta_min:              float = 1e-7,
        warmup_start_factor:  float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.backbone = None
        self.rec_head = None
        self.reg_head = None
        self.gaussnll = nn.GaussianNLLLoss(reduction='mean', full=False, eps=1e-6)
        self.configure_model()

    # ------------------------------------------------------------------
    # Regression weight schedule
    # ------------------------------------------------------------------

    def _reg_weight(self, epoch: int) -> float:
        if epoch < self.hparams.warmup_epochs:
            return 0.0
        ramp = min(
            (epoch - self.hparams.warmup_epochs) / max(self.hparams.reg_ramp_epochs, 1),
            1.0,
        )
        return float(self.hparams.max_reg_weight * ramp)

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
        self.reg_head = nn.Linear(d_model, self.hparams.n_vars * 2)

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
        mean        = reg_out[:, :self.hparams.n_vars]
        var         = F.softplus(reg_out[:, self.hparams.n_vars:])
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

        log_GaussianNLLLoss(
            task=self,
            stage=stage,
            loss=loss_reg,
            indiv_mse=F.mse_loss(mean, y_target, reduction='none').mean(dim=0),
            variance=var,
        )
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def configure_optimizers(self):
        all_params     = list(self.parameters())
        default_params = [p for p in all_params if not hasattr(p, '_optim')]
        optim_params   = [p for p in all_params if     hasattr(p, '_optim')]

        param_groups = [{'params': default_params, 'lr': self.hparams.base_lr,
                         'weight_decay': self.hparams.weight_decay}]

        hps = [getattr(p, '_optim') for p in optim_params]
        unique_hps = [dict(s) for s in sorted(set(frozenset(hp.items()) for hp in hps))]
        for hp in unique_hps:
            group = {
                'params': [p for p in optim_params if getattr(p, '_optim') == hp],
                'lr': hp.get('lr', self.hparams.base_lr),
            }
            group.update(hp)
            param_groups.append(group)

        optimizer = optim.AdamW(param_groups)
        scheduler = WarmupCosineAnnealingWarmRestarts(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            T_0=self.hparams.T_0,
            T_mult=self.hparams.T_mult,
            eta_min=self.hparams.eta_min,
            warmup_start_factor=self.hparams.warmup_start_factor,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
