import lightning as L
from torch.utils.data import DataLoader

from BNSReg.core.config import BNSDataModuleConfig
from BNSReg.dataset.triplet_dataset import BNSDatasetTriplet


class LitBNSDataTriplet(L.LightningDataModule):
    """DataModule wrapping BNSDatasetTriplet for the contrastive denoising task.

    Expects cfg.bkg_only_data_key to be set in the YAML.
    Returns batches of (x_injected, x_signal, x_background).
    """

    def __init__(self, data_cfg: BNSDataModuleConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = data_cfg

    def setup(self, stage: str | None = None):
        if stage == 'fit':
            self.train_dataset = BNSDatasetTriplet('train', self.cfg)
            self.val_dataset   = BNSDatasetTriplet('val',   self.cfg)
        elif stage in ('test', 'predict'):
            self.test_dataset  = BNSDatasetTriplet('test',  self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.dataloader_kwargs('train'))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.dataloader_kwargs('val'))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs('test'))

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs('test'))
