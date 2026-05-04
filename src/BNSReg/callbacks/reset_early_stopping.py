from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.callbacks import EarlyStopping


class ResetEarlyStoppingCallback(Callback):
    """Reset EarlyStopping state after checkpoint loading.

    Lightning's EarlyStopping.load_state_dict() overwrites both wait_count
    and patience from the checkpoint, so a resumed run silently inherits the
    old patience even if the YAML specifies a new value.  Add this callback
    alongside EarlyStopping to start fresh with the YAML-specified patience.

    Args:
        patience: the patience value to enforce, overriding the checkpoint.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                cb.wait_count = 0
                cb.stopped_epoch = 0
                cb.patience = self.patience
