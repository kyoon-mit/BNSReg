import torch

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4Model
from BNSReg.core.config import S4DModelConfig

import pandas as pd

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

        self.csv_fname = 'model_s4d_gaussnll_test.csv'

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
        var = self.var_activation(outputs[:,self.n_vars:])
        # TODO: modularize this as a callback in cfg
        mse_metric = torch.nn.MSELoss(reduction='none')
        y_indiv_mse = mse_metric(mean, y_target).T.mean(dim=1) # (d_output,)
        return self.criterion(mean, y_target, var), y_indiv_mse, var

    def training_step(self, batch, batch_idx):
        loss, y_indiv_mse, var = self.compute_loss(batch)
        # TODO: modularize as callback
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for i in range(len(y_indiv_mse)):
            self.log(f'train/mse/var_{i}', y_indiv_mse[i], on_step=False, on_epoch=True)
            self.log(f'train/sigma_{i}', torch.sqrt(var[i]), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_indiv_mse, var = self.compute_loss(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for i in range(len(y_indiv_mse)):
            self.log(f'val/mse/var_{i}', y_indiv_mse[i], on_step=False, on_epoch=True)
            self.log(f'val/sigma_{i}', torch.sqrt(var[i]), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        X_sequence, y_target, z_observed = batch
        outputs = self(X_sequence) # (B, d_output)
        # TODO: modularize as callback
        df = pd.DataFrame(outputs)
        try:
            df.to_csv(self.csv_fname, mode='a', index=False, header=False)
            print(f'Appended data using pandas to {self.csv_fname}.')
        except FileNotFoundError:
            print(f'File not found, creating a new file in {self.csv_fname}.')
            df.to_csv(self.csv_fname, mode='w', index=False, header=True)



