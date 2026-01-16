import torch
from torch import optim
from lightning.pytorch import LightningModule

from BNSReg.models.s4d import S4Model
from BNSReg.core.config import S4DModelConfig

class LitModelS4DMSE(LightningModule):
    def __init__(self, cfg: S4DModelConfig):
        super().__init__()
        self.cfg = cfg
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.model = None

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
        outputs = self(X_sequence)
        return self.criterion(outputs, y_target)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }

    # def configure_callbacks(self):

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # Callback: write loss, y_target, z_observed to file
        return loss