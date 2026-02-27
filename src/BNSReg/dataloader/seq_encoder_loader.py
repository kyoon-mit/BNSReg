import lightning as L
from torch.utils.data import DataLoader

from BNSReg.core.config import BNSDataModuleConfig
from BNSReg.dataset.seq_encoder_dataset import BNSDatasetSeqEncoder

class LitBNSDataSeqEncoder(L.LightningDataModule):
    def __init__(self, data_cfg: BNSDataModuleConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = data_cfg

    def setup(self, stage: str | None = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # instantiate train and val datasets separately
            self.train_dataset = BNSDatasetSeqEncoder('train', self.cfg)
            self.val_dataset = BNSDatasetSeqEncoder('val', self.cfg)

        # Assign test dataset for use in dataloader
        elif stage in ("test", "predict"):
            self.test_dataset = BNSDatasetSeqEncoder('test', self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.dataloader_kwargs('train'))
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.dataloader_kwargs('val'))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs('test'))
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs('test'))