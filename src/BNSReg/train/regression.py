import lightning as L
from torch.utils.data import DataLoader

from BNSReg.core.config import BNSDataModuleConfig
from BNSReg.dataset.regression import BNSDatasetRegression

class LitBNSDataModuleCallback(LitBNSDataModule): # TODO: move to another place
    def __init__(self, **kwargs):
        cfg = BNSDataModuleConfig(**kwargs)
        super().__init__(cfg)

class LitBNSDataModule(L.LightningDataModule):
    def __init__(self, cfg: BNSDataModuleConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # instantiate train and val datasets separately
            self.train_dataset = BNSDatasetRegression(**self.cfg.dataset_kwargs('train'))
            self.val_dataset = BNSDatasetRegression(**self.cfg.dataset_kwargs('val'))

        # Assign test dataset for use in dataloader
        elif stage in ("test", "predict"):
            self.test_dataset = BNSDatasetRegression(**self.cfg.dataset_kwargs('test'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.dataloader_kwargs('train'))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.dataloader_kwargs('val'))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs('test'))
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs('test'))