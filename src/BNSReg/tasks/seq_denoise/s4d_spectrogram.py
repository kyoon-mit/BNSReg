import torch
from torch import optim

from BNSReg.losses.denoising import SpectrogramLoss
from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig
from BNSReg.utils.optim_groups import s4_param_groups
from BNSReg.utils.schedulers import build_lr_scheduler


class LitModelS4DSpectrogram(LitBaseTask):
    """S4D denoiser trained on STFT magnitudes.

    See SpectrogramLoss for what the terms measure. Both are magnitude only,
    so neither constrains phase or time alignment on its own.

    Args:
        cfg:           S4D model configuration.
        n_fft:         STFT window length in samples.
        hop_length:    samples between consecutive STFT frames.
        loss_type:     'convergence', 'log_magnitude', or 'composite'.
        eps:           floor inside the log and on the convergence
                        denominator.
        base_lr:       AdamW learning rate for non-kernel parameters.
        weight_decay:  AdamW weight decay for non-kernel parameters.
        scheduler:     name passed to build_lr_scheduler.
        monitor:       metric the scheduler and checkpointing watch.
    """

    def __init__(
        self,
        cfg: S4DModelConfig,
        n_fft: int = 256,
        hop_length: int = 64,
        loss_type: str = 'composite',
        eps: float = 1e-8,
        base_lr: float = 1e-3,
        weight_decay: float = 1e-2,
        scheduler: str = 'exponential',
        scheduler_kwargs: dict | None = None,
        monitor: str = 'val/loss',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.criterion = SpectrogramLoss(
            n_fft=n_fft,
            hop_length=hop_length,
            loss_type=loss_type,
            eps=eps,
        )
        self.model = None
        # Built eagerly (not left to Lightning's configure_model hook) because
        # LitDenoiserBaseTask._load_denoiser instantiates this class directly
        # and then calls load_state_dict on it.
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4ModelSeq2Seq(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        # S4ModelSeq2Seq.forward takes (B, d_input, L) and transposes
        # internally, unlike S4Model.forward; pass the batch as is.
        noisy, target = batch       # (B, n_ifos, L)
        out = self(noisy)           # (B, d_output, L)

        # SpectrogramLoss expects (B, L, n_ifos); transpose for the loss only.
        terms = self.criterion._compute_loss(
            out.transpose(1, 2), target.transpose(1, 2)
        )
        loss = sum(terms.values())
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True)
        for name, value in terms.items():
            self.log(f'{stage}/{name}', value, on_step=False, on_epoch=True)

        # Time-domain MSE is not optimized here; logged so runs using a
        # different loss stay comparable on a common yardstick.
        self.log(f'{stage}/time_domain_mse', (out - target).pow(2).mean(),
                 on_step=False, on_epoch=True)

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
            s4_param_groups(
                self, self.hparams.base_lr, self.hparams.weight_decay
            )
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
