import torch
import torch.nn.functional as F
import torch.optim as optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.linoss import LinOSSModel
from BNSReg.core.config import LinOSSModelConfig
from BNSReg.callbacks.log_metric import log_GaussianNLLLoss
from BNSReg.losses.beta_nll import BetaNLLLoss
from BNSReg.utils.schedulers import WarmupCosineAnnealingWarmRestarts


class LitModelLinOSSGaussianNLLLoss(LitBaseTask):
    def __init__(
        self,
        model_cfg: LinOSSModelConfig,
        beta_nll: float = 0.5,
        lambda_spread: float = 0.0,
        base_lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 10,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-7,
        warmup_start_factor: float = 1e-2,
    ):
        super().__init__()
        if model_cfg.d_output % 2 != 0:
            raise ValueError(
                f'{model_cfg.d_output=} must be even and twice the number of regressed variables.'
            )
        self.save_hyperparameters()
        self.cfg = model_cfg
        self.n_vars = int(self.cfg.d_output / 2)
        self.criterion = BetaNLLLoss(beta=beta_nll, reduction='mean')
        self.var_activation = torch.nn.Softplus()
        self.model = None
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = LinOSSModel(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.base_lr,
            weight_decay=self.hparams.weight_decay,
        )
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
        outputs = self(X_sequence)               # (B, d_output)
        mean = outputs[:, :self.n_vars]
        var = self.var_activation(outputs[:, self.n_vars:])
        mse_metric = torch.nn.MSELoss(reduction='none')
        y_indiv_mse = mse_metric(mean, y_target).T.mean(dim=1)  # (n_vars,)
        nll = self.criterion(mean, y_target, var)

        spread = F.softplus(y_target.detach().var(dim=0) - mean.var(dim=0)).mean()

        loss = nll + self.hparams.lambda_spread * spread
        return loss, nll, spread, y_indiv_mse, var, mean, y_target

    def training_step(self, batch, batch_idx):
        loss, nll, spread, y_indiv_mse, var, mean, y_target = self.compute_loss(batch)
        log_GaussianNLLLoss(self, 'train', nll, y_indiv_mse, var)
        self.log('train/spread_penalty', spread, on_step=False, on_epoch=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        return {'loss': loss, 'mean': mean.detach(), 'y_target': y_target.detach()}

    def validation_step(self, batch, batch_idx):
        loss, nll, spread, y_indiv_mse, var, mean, y_target = self.compute_loss(batch)
        log_GaussianNLLLoss(self, 'val', nll, y_indiv_mse, var)
        self.log('val/spread_penalty', spread, on_step=False, on_epoch=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        return {'loss': loss, 'mean': mean.detach(), 'y_target': y_target.detach()}

    def test_step(self, batch, batch_idx):
        X_sequence, y_target, z_observed = batch
        X_sequence = X_sequence.transpose(2, 1)
        outputs = self(X_sequence)
        mean = outputs[:, :self.n_vars]
        var = self.var_activation(outputs[:, self.n_vars:])
        sigma = torch.sqrt(var)
        return {
            'y_true':     y_target.detach().cpu(),
            'y_pred':     mean.detach().cpu(),
            'y_sigma':    sigma.detach().cpu(),
            'z_observed': z_observed.detach().cpu(),
        }

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'grad_norm/{name}', param.grad.norm(), on_step=False, on_epoch=True)
