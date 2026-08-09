from lightning.pytorch.callbacks import Callback

from BNSReg.utils.schedulers import WarmupCosineAnnealingWarmRestarts


class WarmupCosineSchedulerCallback(Callback):
    """Applies WarmupCosineAnnealingWarmRestarts to the model's optimizer externally.

    Replaces whatever lr_scheduler configure_optimizers() returned (e.g. ExponentialLR)
    so that models without scheduler hyperparams in their constructor can still use a
    proper warmup+cosine schedule configured from the YAML callbacks section.
    """

    def __init__(
        self,
        base_lr: float,
        warmup_epochs: int,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 1e-7,
        warmup_start_factor: float = 0.01,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        self._scheduler = None

    def on_fit_start(self, trainer, pl_module):
        opt = trainer.optimizers[0]
        for pg in opt.param_groups:
            pg['lr'] = self.base_lr
            pg['initial_lr'] = self.base_lr
        self._scheduler = WarmupCosineAnnealingWarmRestarts(
            opt,
            warmup_epochs=self.warmup_epochs,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
            warmup_start_factor=self.warmup_start_factor,
        )
        if trainer.lr_schedulers:
            trainer.lr_schedulers[0]['scheduler'] = self._scheduler

    def on_train_epoch_end(self, trainer, pl_module):
        if self._scheduler is not None:
            self._scheduler.step()
