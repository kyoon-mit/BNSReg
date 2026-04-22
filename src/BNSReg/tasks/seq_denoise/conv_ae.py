import torch
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.conv_ae import ConvAE, ConvAEAP, ConvAttentionAEAP
from BNSReg.core.config import ConvAEModelConfig, ConvAEAPModelConfig, ConvAttentionAEAPModelConfig
    
class LitModelConvAE(LitBaseTask):
    def __init__(self, model_cfg: ConvAEModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = model_cfg
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.model = None
        self.configure_model()

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
        
        n_ifos = input.shape[1]
        total_loss = 0

        for i in range(n_ifos):
            output = self(input[:, i, :])
            total_loss += self.criterion(output, target[:, i, :].unsqueeze(1))

        return total_loss / n_ifos

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
    
class LitModelConvAEAP(LitModelConvAE):
    def __init__(self, model_cfg: ConvAEAPModelConfig):
        super().__init__(model_cfg)

    def configure_model(self):
        if self.model is not None:
            return
        else:
            self.model = ConvAEAP(**self.cfg.model_kwargs())
            self.model = torch.compile(self.model)

class LitModelConvAttentionAEAP(LitModelConvAE):
    def __init__(self, model_cfg: ConvAttentionAEAPModelConfig):
        super().__init__(model_cfg)

    def configure_model(self):
        if self.model is not None:
            return
        else:
            self.model = ConvAttentionAEAP(**self.cfg.model_kwargs())
            self.model = torch.compile(self.model)