import math

import torch
import torch.optim as optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4Model
from BNSReg.core.config import S4DModelConfig
from BNSReg.losses.spherical import CosineSkyLoss, sky_angles_to_vec, ring_distance, H1L1_BASELINE
from BNSReg.utils.schedulers import WarmupCosineAnnealingWarmRestarts


class LitModelS4DSkyCosineLoss(LitBaseTask):
    """S4D model for sky localization (dec, phi) via cosine distance on the sphere.

    d_output must be 3 — the model predicts a raw (x, y, z) direction vector
    which is L2-normalised internally before computing the loss.
    """

    def __init__(
        self,
        model_cfg: S4DModelConfig,
        base_lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 10,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-7,
        warmup_start_factor: float = 1e-2,
    ):
        super().__init__()
        if model_cfg.d_output != 3:
            raise ValueError(f'{model_cfg.d_output=} must be 3 (x, y, z direction vector).')
        self.save_hyperparameters()
        self.cfg = model_cfg
        self.criterion = CosineSkyLoss()
        self.register_buffer('baseline', H1L1_BASELINE)
        self.model = None
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4Model(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def configure_optimizers(self):
        all_params = list(self.parameters())
        default_params = [p for p in all_params if not hasattr(p, '_optim')]
        optim_params  = [p for p in all_params if     hasattr(p, '_optim')]

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

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):
        X_sequence, y_target, z_observed = batch
        X_sequence = X_sequence.transpose(2, 1)  # (B, d_input, L) -> (B, L, d_input)
        pred = self(X_sequence)  # (B, 3)

        loss = self.criterion(pred, y_target)

        v_pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        v_true = sky_angles_to_vec(y_target[:, 0], y_target[:, 1])
        cos_sim = (v_pred * v_true).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        angular_error_deg = torch.acos(cos_sim).mean() * (180.0 / math.pi)
        ring_dist = ring_distance(v_pred, v_true, self.baseline)

        return loss, angular_error_deg, ring_dist, pred

    def training_step(self, batch, batch_idx):
        X_sequence, y_target, _ = batch
        loss, angular_error_deg, ring_dist, pred = self.compute_loss(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/angular_error_deg', angular_error_deg, on_step=False, on_epoch=True)
        self.log('train/ring_distance', ring_dist, on_step=False, on_epoch=True)
        return {'loss': loss, 'mean': pred.detach(), 'y_target': y_target.detach()}

    def validation_step(self, batch, batch_idx):
        X_sequence, y_target, _ = batch
        loss, angular_error_deg, ring_dist, pred = self.compute_loss(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/angular_error_deg', angular_error_deg, on_step=False, on_epoch=True)
        self.log('val/ring_distance', ring_dist, on_step=False, on_epoch=True)
        return {'loss': loss, 'mean': pred.detach(), 'y_target': y_target.detach()}

    def test_step(self, batch, batch_idx):
        X_sequence, y_target, z_observed = batch
        X_sequence = X_sequence.transpose(2, 1)
        pred = self(X_sequence)
        v_pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        return {
            'y_true':     y_target.detach().cpu(),
            'y_pred':     v_pred.detach().cpu(),
            'z_observed': z_observed.detach().cpu(),
        }

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'grad_norm/{name}', param.grad.norm(), on_step=False, on_epoch=True)
            if 'log_A_real' in name:
                self.log(f'ssm/A_real_mean/{name}', -param.exp().mean(), on_step=False, on_epoch=True)
                self.log(f'ssm/A_real_max/{name}', -param.exp().max(), on_step=False, on_epoch=True)
            if 'log_dt' in name:
                self.log(f'ssm/dt_mean/{name}', param.exp().mean(), on_step=False, on_epoch=True)
                self.log(f'ssm/dt_max/{name}', param.exp().max(), on_step=False, on_epoch=True)
