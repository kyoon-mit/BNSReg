import torch
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.conv_ae import ConvAE
from BNSReg.core.config import ConvAEModelConfig
    
class LitModelSeqAE(LitBaseTask):
    def __init__(self, cfg: ConvAEModelConfig):
        super().__init__()
        self.cfg = cfg
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.model = None

    def configure_model(self):
        if self.model is not None:
            return
        else:
            self.model = ConvAE(**self.cfg.model_kwargs())
            self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, batch):
        input, target = batch   # (B, n_ifos, L), (B, n_ifos, L)
        
        h1_injected = input[:, 0, :]
        l1_injected = input[:, 1, :]
        h1_sig_only = target[:, 0, :]
        l1_sig_only = target[:, 1, :]

        h1_out = self(h1_injected)
        l1_out = self(l1_injected)

        loss_h1 = self.criterion(h1_out, h1_sig_only.unsqueeze(1))
        loss_l1 = self.criterion(l1_out, l1_sig_only.unsqueeze(1))
        return (loss_h1 + loss_l1) / 2

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }