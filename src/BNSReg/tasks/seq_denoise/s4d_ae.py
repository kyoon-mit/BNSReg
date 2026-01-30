import torch
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4Model
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
        else:
            self.model = S4Model(**self.cfg.model_kwargs())
            self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):
        input, target = batch   # (B, n_ifos, L), (B, n_ifos, L)
        
        input = input.transpose(1, 2) # S4Model expected to see (B, L, d_input)
        h1_injected = input[:, :, 0].unsqueeze(-1) # (B, L) -> (B, L, 1)
        l1_injected = input[:, :, 1].unsqueeze(-1)
        h1_sig_only = target[:, 0, :] # (B, L) = (B, d_output) set in config
        l1_sig_only = target[:, 1, :]

        h1_out = self(h1_injected) # (B, d_output)
        l1_out = self(l1_injected)

        loss_h1 = self.criterion(h1_out, h1_sig_only)
        loss_l1 = self.criterion(l1_out, l1_sig_only)
        return (loss_h1 + loss_l1) / 2

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }