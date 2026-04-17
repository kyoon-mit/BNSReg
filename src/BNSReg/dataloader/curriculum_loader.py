from dataclasses import fields

import lightning as L
from torch.utils.data import DataLoader

from BNSReg.core.config import BNSDataModuleCurriculumConfig, BNSDataModuleRegressionConfig
from BNSReg.dataset.regression_curriculum_dataset import BNSDatasetMixture
from BNSReg.dataset.regression_dataset import BNSDatasetRegression


def _regression_cfg(
    cfg: BNSDataModuleCurriculumConfig, train_file: str
) -> BNSDataModuleRegressionConfig:
    """Build a BNSDataModuleRegressionConfig from a curriculum config with a specific train_file."""
    parent_field_names = {f.name for f in fields(BNSDataModuleRegressionConfig)}
    kwargs = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name in parent_field_names}
    kwargs["train_file"] = train_file
    return BNSDataModuleRegressionConfig(**kwargs)


class LitBNSDataCurriculum(L.LightningDataModule):
    """Curriculum DataModule: mixes training files by per-stage weights.

    Stage transitions are driven by epoch number. At the start of each epoch,
    the mixture weights over train_files are updated according to stage_schedule
    and stage_weights in the config. Setting a weight to 0 gives hard switching
    (equivalent to Option A behaviour).

    Example config for hard switching across 3 SNR bins:
        train_files: [high_snr.h5, med_snr.h5, low_snr.h5]
        stage_schedule: [0, 100, 200]
        stage_weights: [[1,0,0], [0,1,0], [0,0,1]]

    Example config for smooth mixing:
        stage_weights: [[0.8,0.15,0.05], [0.4,0.4,0.2], [0.2,0.4,0.4]]
    """

    def __init__(self, data_cfg: BNSDataModuleCurriculumConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = data_cfg

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            bin_datasets = [
                BNSDatasetRegression("train", _regression_cfg(self.cfg, f))
                for f in self.cfg.train_files
            ]
            self.train_dataset = BNSDatasetMixture(bin_datasets, list(self.cfg.stage_weights[0]))
            self.val_dataset = BNSDatasetRegression("val", self.cfg)

        elif stage in ("test", "predict"):
            self.test_dataset = BNSDatasetRegression("test", self.cfg)

    def on_train_epoch_start(self) -> None:
        epoch = self.trainer.current_epoch
        stage_idx = 0
        for i, start_epoch in enumerate(self.cfg.stage_schedule):
            if epoch >= start_epoch:
                stage_idx = i
        new_weights = list(self.cfg.stage_weights[stage_idx])
        self.train_dataset.set_weights(new_weights)
        self.print(f"Curriculum stage {stage_idx} (epoch {epoch}): weights={new_weights}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.cfg.dataloader_kwargs("train"))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.cfg.dataloader_kwargs("val"))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs("test"))

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self.cfg.dataloader_kwargs("test"))
