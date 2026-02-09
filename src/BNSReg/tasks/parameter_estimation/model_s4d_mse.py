import torch
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4Model
from BNSReg.core.config import S4DModelConfig

class LitModelS4DMSE(LitBaseTask):
    def __init__(self, cfg: S4DModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.criterion = torch.nn.MSELoss(reduction='mean')
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
        # TODO: modularize this as a callback in cfg
        mse_metric = torch.nn.MSELoss(reduction='none')
        y_indiv_mse = mse_metric(outputs, y_target).T.mean(dim=1) # (d_output,)
        return self.criterion(outputs, y_target), y_indiv_mse

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }

    def training_step(self, batch, batch_idx):
        loss, y_indiv_mse = self.compute_loss(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for i in range(len(y_indiv_mse)):
            self.log(f'train/mse/var_{i}', y_indiv_mse[i], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_indiv_mse = self.compute_loss(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for i in range(len(y_indiv_mse)):
            self.log(f'val/mse/var_{i}', y_indiv_mse[i], on_step=False, on_epoch=True)
        return loss