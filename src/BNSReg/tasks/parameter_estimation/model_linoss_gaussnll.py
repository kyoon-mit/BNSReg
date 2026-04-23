import torch

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.linoss import LinOSSModel
from BNSReg.core.config import LinOSSModelConfig
from BNSReg.callbacks.log_metric import log_GaussianNLLLoss
from BNSReg.losses.beta_nll import BetaNLLLoss


class LitModelLinOSSGaussianNLLLoss(LitBaseTask):
    def __init__(self, model_cfg: LinOSSModelConfig, beta_nll: float = 0.5):
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X_sequence, y_target, z_observed = batch
        X_sequence = X_sequence.transpose(2, 1)
        outputs = self(X_sequence)
        mean = outputs[:, :self.n_vars]
        var = self.var_activation(outputs[:, self.n_vars:])
        mse_metric = torch.nn.MSELoss(reduction='none')
        y_indiv_mse = mse_metric(mean, y_target).T.mean(dim=1)
        loss = self.criterion(mean, y_target, var)
        log_GaussianNLLLoss(self, 'train', loss, y_indiv_mse, var)
        return {'loss': loss, 'mean': mean.detach(), 'y_target': y_target.detach()}

    def validation_step(self, batch, batch_idx):
        X_sequence, y_target, z_observed = batch
        X_sequence = X_sequence.transpose(2, 1)
        outputs = self(X_sequence)
        mean = outputs[:, :self.n_vars]
        var = self.var_activation(outputs[:, self.n_vars:])
        mse_metric = torch.nn.MSELoss(reduction='none')
        y_indiv_mse = mse_metric(mean, y_target).T.mean(dim=1)
        loss = self.criterion(mean, y_target, var)
        log_GaussianNLLLoss(self, 'val', loss, y_indiv_mse, var)
        return {'loss': loss, 'mean': mean.detach(), 'y_target': y_target.detach()}

    def test_step(self, batch, batch_idx):
        X_sequence, y_target, z_observed = batch
        X_sequence = X_sequence.transpose(2, 1)  # (B, d_input, L) -> (B, L, d_input)
        outputs = self(X_sequence)               # (B, d_output)
        mean = outputs[:, :self.n_vars]
        var = self.var_activation(outputs[:, self.n_vars:])
        sigma = torch.sqrt(var)
        return {
            'y_true':  y_target.detach().cpu(),
            'y_pred':  mean.detach().cpu(),
            'y_sigma': sigma.detach().cpu(),
        }
