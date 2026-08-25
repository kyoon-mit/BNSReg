import math
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineAnnealingWarmRestarts(LRScheduler):
    """
    Linear warmup followed by CosineAnnealingWarmRestarts.

    For the first ``warmup_epochs`` steps, LR ramps linearly from
    ``base_lr * warmup_start_factor`` to ``base_lr``.
    After warmup, CosineAnnealingWarmRestarts takes over with cycle
    length ``T_0``, multiplier ``T_mult``, and floor ``eta_min``.

    Each restart peak is scaled by ``peak_decay ** n``, where ``n`` is the
    zero-based cycle index. With ``peak_decay=0.8`` the peaks are
    ``base_lr``, ``0.8 * base_lr``, ``0.64 * base_lr``, ... The default of
    1.0 leaves every restart at full height.

    Args:
        optimizer:           Wrapped optimizer.
        warmup_epochs:       Number of linear-warmup epochs.
        T_0:                 Length (epochs) of the first cosine cycle.
        T_mult:              Cycle-length multiplier after each restart.
        eta_min:             Minimum LR floor during cosine phase.
        warmup_start_factor: LR at epoch 0 as a fraction of base_lr.
        peak_decay:          Multiplicative decay of the LR peak per restart,
                             in (0, 1]. 1.0 disables peak decay.
        last_epoch:          Last epoch index (-1 = start fresh).

    Raises:
        ValueError: if ``peak_decay`` is outside (0, 1].
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 1e-8,
        warmup_start_factor: float = 0.01,
        peak_decay: float = 1.0,
        last_epoch: int = -1,
    ):
        if not 0.0 < peak_decay <= 1.0:
            raise ValueError(f'peak_decay must be in (0, 1], got {peak_decay}')
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        self.peak_decay = peak_decay
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch

        if e < self.warmup_epochs:
            # Linear warmup
            alpha = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * e / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]

        # Cosine annealing with warm restarts (offset by warmup_epochs)
        t = e - self.warmup_epochs
        T_cur, T_i, cycle = self._cosine_position(t)
        # Peak of this cycle, floored at eta_min so a long run cannot invert
        # the cosine once peak_decay ** cycle drops below the floor.
        decay = self.peak_decay ** cycle
        return [
            self.eta_min + (peak - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            for peak in (max(base_lr * decay, self.eta_min) for base_lr in self.base_lrs)
        ]

    def _cosine_position(self, t: int) -> tuple[int, int, int]:
        """Return (T_cur, T_i, cycle): position within, length of, and index of
        the current cosine cycle."""
        T_i = self.T_0
        cycle = 0
        while t >= T_i:
            t -= T_i
            T_i *= self.T_mult
            cycle += 1
        return t, T_i, cycle


# Registry of selectable LR schedulers. Keys are the strings accepted by
# `build_lr_scheduler`; values are (factory, default kwargs).
_SCHEDULERS: dict[str, tuple[type, dict]] = {
    'exponential':   (lr_scheduler.ExponentialLR,               {'gamma': 0.99}),
    'cosine':        (lr_scheduler.CosineAnnealingLR,           {'T_max': 100, 'eta_min': 1e-8}),
    'cosine_restart': (lr_scheduler.CosineAnnealingWarmRestarts, {'T_0': 16, 'T_mult': 2, 'eta_min': 1e-8}),
    'warmup_cosine': (WarmupCosineAnnealingWarmRestarts,
                      {'warmup_epochs': 0, 'T_0': 16, 'T_mult': 2, 'eta_min': 1e-8,
                       'warmup_start_factor': 0.01, 'peak_decay': 1.0}),
    'step':          (lr_scheduler.StepLR,                      {'step_size': 30, 'gamma': 0.1}),
    'plateau':       (lr_scheduler.ReduceLROnPlateau,           {'mode': 'min', 'factor': 0.5, 'patience': 5}),
}


def build_lr_scheduler(
    name: str,
    optimizer,
    kwargs: dict | None = None,
    monitor: str = 'val/loss',
) -> dict | None:
    """Build a Lightning `lr_scheduler` config from a scheduler name.

    Args:
        name:      one of `_SCHEDULERS`, or 'none' for a constant LR.
        optimizer: the optimizer to wrap.
        kwargs:    overrides merged onto the scheduler's defaults.
        monitor:   metric watched by 'plateau'; ignored by the others.

    Returns:
        The dict expected under `configure_optimizers()['lr_scheduler']`, or
        None when `name == 'none'`.

    Raises:
        ValueError: if `name` is not a known scheduler.
    """
    if name == 'none':
        return None
    if name not in _SCHEDULERS:
        raise ValueError(
            f"unknown scheduler '{name}'; choose from {sorted(_SCHEDULERS)} or 'none'"
        )

    factory, defaults = _SCHEDULERS[name]
    scheduler = factory(optimizer, **{**defaults, **(kwargs or {})})

    config = {'scheduler': scheduler, 'interval': 'epoch'}
    # ReduceLROnPlateau steps on a metric, so Lightning needs to know which one.
    if name == 'plateau':
        config['monitor'] = monitor
    return config
