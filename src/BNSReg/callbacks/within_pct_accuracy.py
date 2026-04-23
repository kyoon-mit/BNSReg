import torch
from lightning.pytorch.callbacks import Callback

THRESHOLDS = (50, 10, 5, 1)


class WithinPctAccuracyCallback(Callback):
    """
    Logs the fraction of samples whose prediction falls within x% of the true
    value, for x in (50, 10, 5, 1), at the end of each train/val epoch.

    Unnormalizes predictions and targets using var_scales from the datamodule
    before computing relative error, so percentages are in physical units.

    Metrics logged: {stage}/within_{x}pct/out_{i} for each variable i.
    """

    def on_train_start(self, trainer, pl_module):
        self._read_scales(trainer)

    def on_validation_start(self, trainer, pl_module):
        self._read_scales(trainer)

    def _read_scales(self, trainer):
        cfg = trainer.datamodule.cfg
        self.normalize          = cfg.normalize_variables
        self.normalize_range    = tuple(cfg.normalize_range)
        self.target_variables   = list(cfg.target_variables)
        self.var_scales         = dict(
            getattr(trainer.datamodule.train_dataset, 'var_scales', {})
        )

    def _unnorm(self, arr: torch.Tensor) -> torch.Tensor:
        lo, hi = self.normalize_range
        out = arr.clone()
        for i, var in enumerate(self.target_variables):
            if var in self.var_scales:
                vmin, vmax = self.var_scales[var]
                out[:, i] = vmin + (vmax - vmin) * (arr[:, i] - lo) / (hi - lo)
        return out

    def _log(self, pl_module, outputs, stage: str):
        if outputs is None or not isinstance(outputs, dict) or 'mean' not in outputs:
            return
        mean     = outputs['mean']      # (B, n_vars)
        y_target = outputs['y_target']  # (B, n_vars)

        if self.normalize and self.var_scales:
            mean     = self._unnorm(mean)
            y_target = self._unnorm(y_target)

        eps     = 1e-8
        rel_err = (mean - y_target).abs() / (y_target.abs() + eps)  # (B, n_vars)

        for pct in THRESHOLDS:
            within = (rel_err <= pct / 100.0).float().mean(dim=0)  # (n_vars,)
            for i, acc in enumerate(within):
                pl_module.log(f'{stage}/within_{pct}pct/out_{i}', acc,
                              on_step=False, on_epoch=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log(pl_module, outputs, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._log(pl_module, outputs, 'val')
