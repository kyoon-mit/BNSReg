import torch
import torch.nn.functional as F
from torch import nn, optim
from lightning.pytorch import LightningModule

from BNSReg.models.s4d import S4D

class LitS4DM(LightningModule):
    def __init__(
        self,
        d_input,
        d_output,
        variables: list[str],
        loss='',
        prenorm=False,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        lr=0.01,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        # Do not set self.device directly; Lightning will manage devices.
        # If you need device info, query `self.device` during `forward` or later.
        dropout_fn = nn.Dropout1d
        self.encoder = nn.Linear(d_input, d_model)
        self.decoder = nn.Linear(d_model, d_output)
        self.n_outparams = d_output # placeholder
        # Stack S4 layers as residual blocks
        self.s4_layers, self.norms, self.dropouts =\
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=lr)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        self.criterion = torch.nn.MSELoss(reduction='mean')
        # self.criterion = torch.nn.KLDivLoss(reduction='mean')

    def __loss__(self, batch):
        h1, l1, params, idx = batch
        device = h1.device
        inputs = torch.stack([h1.to(device), l1.to(device)], dim=2)
        targets = torch.stack([params[v] for v in self.hparams.variables], dim=1)
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.hparams.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.hparams.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return optimizer, scheduler

    def on_train_start(self):
        # free any leftover GPU memory on startup
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        # Clear GPU cache
        # torch.cuda.empty_cache()
        loss = self.__loss__(batch)
        self.log("train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            reduce_fx='mean',
            logger=True,
            prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Clear GPU cache
        # torch.cuda.empty_cache()
        loss = self.__loss__(batch)
        self.log("val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            reduce_fx='mean',
            logger=True,
            prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Clear GPU cache
        # torch.cuda.empty_cache()
        loss = self.__loss__(batch)
        self.log("test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            reduce_fx='mean',
            logger=True,
            prog_bar=True)
        return

    def predict_step(self, batch, batch_idx):
        # Clear GPU cache
        # torch.cuda.empty_cache()
        self.test_step(batch, batch_idx)
        return