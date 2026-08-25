import torch
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig
from BNSReg.utils.optim_groups import s4_param_groups
from BNSReg.utils.schedulers import build_lr_scheduler

class LitModelS4DMSE(LitBaseTask):
    def __init__(
        self,
        cfg: S4DModelConfig,
        base_lr: float = 1e-3,
        weight_decay: float = 1e-2,
        scheduler: str = 'exponential',
        scheduler_kwargs: dict | None = None,
        monitor: str = 'val/loss',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.model = None
        # Built eagerly (not left to Lightning's configure_model hook) because
        # LitDenoiserBaseTask._load_denoiser instantiates this class directly and
        # then calls load_state_dict on it.
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4ModelSeq2Seq(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        # S4ModelSeq2Seq.forward takes (B, d_input, L) and transposes internally,
        # unlike S4Model.forward which takes (B, L, d_input) — pass the batch as is.
        noisy, target = batch       # (B, n_ifos, L)
        out = self(noisy)           # (B, d_output, L)

        loss = self.criterion(out, target)
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Per-output-channel MSE in one reduction; d_output == n_ifos here.
        mse_per_channel = (out - target).pow(2).mean(dim=(0, 2))  # (d_output,)
        for i, mse_i in enumerate(mse_per_channel):
            self.log(f'{stage}/mse/ifo_{i}', mse_i, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def configure_optimizers(self):
        # Keep the SSM kernel parameters out of weight decay and honour the
        # per-kernel lr from S4DModelConfig.lr; see s4_param_groups.
        optimizer = optim.AdamW(
            s4_param_groups(self, self.hparams.base_lr, self.hparams.weight_decay)
        )
        sched_config = build_lr_scheduler(
            self.hparams.scheduler,
            optimizer,
            self.hparams.scheduler_kwargs,
            monitor=self.hparams.monitor,
        )
        if sched_config is None:
            return optimizer
        return {'optimizer': optimizer, 'lr_scheduler': sched_config}
