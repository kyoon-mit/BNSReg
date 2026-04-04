import torch

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4Model
from BNSReg.core.config import S4DModelConfig
from BNSReg.callbacks.log_metric import log_GaussianNLLLoss

class LitModelS4DGaussianNLLLoss(LitBaseTask):
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

    def compute_loss(self, batch):
        X_sequence, y_target, z_observed = batch
        X_sequence = X_sequence.transpose(2, 1)  # (B, d_input, L) -> (B, L, d_input)
        outputs = self(X_sequence) # (B, d_output)
        mean = outputs[:,:self.n_vars]
        var = self.var_activation(outputs[:,self.n_vars:]) # (B, d_output)
        # TODO: modularize this as a callback in cfg
        mse_metric = torch.nn.MSELoss(reduction='none')
        y_indiv_mse = mse_metric(mean, y_target).T.mean(dim=1) # (d_output,)
        return self.criterion(mean, y_target, var), y_indiv_mse, var

    def training_step(self, batch, batch_idx):
        loss, y_indiv_mse, var = self.compute_loss(batch)
        log_GaussianNLLLoss(self, 'train', loss, y_indiv_mse, var)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_indiv_mse, var = self.compute_loss(batch)
        log_GaussianNLLLoss(self, 'val', loss, y_indiv_mse, var)
        return loss

    def test_step(self, batch, batch_idx):
        X_sequence, y_target, z_observed = batch
        X_sequence = X_sequence.transpose(2, 1)  # (B, d_input, L) -> (B, L, d_input)
        outputs = self(X_sequence)  # (B, d_output)
        mean  = outputs[:, :self.n_vars]
        var   = self.var_activation(outputs[:, self.n_vars:])
        sigma = torch.sqrt(var)
        return {
            'y_true':  y_target.detach().cpu(),
            'y_pred':  mean.detach().cpu(),
            'y_sigma': sigma.detach().cpu(),
        }



