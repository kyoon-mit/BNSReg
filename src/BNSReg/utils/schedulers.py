import math
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineAnnealingWarmRestarts(LRScheduler):
    """
    Linear warmup followed by CosineAnnealingWarmRestarts.

    For the first ``warmup_epochs`` steps, LR ramps linearly from
    ``base_lr * warmup_start_factor`` to ``base_lr``.
    After warmup, CosineAnnealingWarmRestarts takes over with cycle
    length ``T_0``, multiplier ``T_mult``, and floor ``eta_min``.

    Args:
        optimizer:           Wrapped optimizer.
        warmup_epochs:       Number of linear-warmup epochs.
        T_0:                 Length (epochs) of the first cosine cycle.
        T_mult:              Cycle-length multiplier after each restart.
        eta_min:             Minimum LR floor during cosine phase.
        warmup_start_factor: LR at epoch 0 as a fraction of base_lr.
        last_epoch:          Last epoch index (-1 = start fresh).
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 1e-8,
        warmup_start_factor: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch

        if e < self.warmup_epochs:
            # Linear warmup
            alpha = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * e / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]

        # Cosine annealing with warm restarts (offset by warmup_epochs)
        t = e - self.warmup_epochs
        T_cur, T_i = self._cosine_position(t)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def _cosine_position(self, t: int) -> tuple[int, int]:
        """Return (T_cur, T_i): position within and length of the current cosine cycle."""
        T_i = self.T_0
        while t >= T_i:
            t -= T_i
            T_i *= self.T_mult
        return t, T_i
