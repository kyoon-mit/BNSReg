import lightning as L
from torch.utils.data import DataLoader

from BNSReg.core.config import BNSDataModuleRegressionConfig
from BNSReg.dataset.joint_denoise_dataset import BNSDatasetJointDenoise


class LitBNSDataJointDenoise(L.LightningDataModule):
    """DataModule wrapping BNSDatasetJointDenoise for joint denoising + regression.

    Returns batches of (x_noisy, x_clean, y_target, z_observed).
    Requires a BNSDataModuleRegressionConfig (target_variables + sig_only_data_key).
    """

    def __init__(self, data_cfg: BNSDataModuleRegressionConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = data_cfg

    def setup(self, stage: str | None = None):
        if stage == 'fit':
            self.train_dataset = BNSDatasetJointDenoise('train', self.cfg)
            self.val_dataset   = BNSDatasetJointDenoise('val',   self.cfg)
        elif stage in ('test', 'predict'):
            self.test_dataset  = BNSDatasetJointDenoise('test',  self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.dataloader_kwargs('train'))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.cfg.dataloader_kwargs('val'))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs('test'))

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs('test'))
