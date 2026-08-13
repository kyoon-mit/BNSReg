import torch
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d_seq2seq import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig

class LitModelS4DAE(LitBaseTask):
    def __init__(self, cfg: S4DModelConfig):
        super().__init__()
        self.cfg = cfg
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.model = None
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4ModelSeq2Seq(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        input, target = batch        # (B, n_ifos, L)
        x = input.transpose(1, 2)   # (B, L, d_input)
        t = target.transpose(1, 2)  # (B, L, d_output)
        out = self(x)               # (B, L, d_output)

        loss = self.criterion(out, t)
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        n_ifos = out.shape[-1]
        for i in range(n_ifos):
            mse_i = self.criterion(out[..., i], t[..., i])
            self.log(f'{stage}/mse/ifo_{i}', mse_i, on_step=False, on_epoch=True)

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
