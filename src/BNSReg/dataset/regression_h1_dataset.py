import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset, Stage
from BNSReg.core.config import BNSDataModuleRegressionConfig
from BNSReg.utils.config_tools import str_to_dtype

class BNSDatasetRegressionH1(BNSBaseDataset):
    def __init__(self, stage: Stage, cfg: BNSDataModuleRegressionConfig):
        super().__init__(stage, cfg)
        self._set_index()

    def __getitem__(self, idx):
        f = self._get_file()

        seq = f[self.cfg.injected_data_key][idx, 0:1, self.start_idx:self.end_idx:self.cfg.downsample_factor]

        X_sequence = torch.as_tensor(seq, dtype=str_to_dtype(self.cfg.strain_precision))
        y_target = self._get_vars(f, self.cfg.target_variables, idx, dtype=str_to_dtype(self.cfg.variables_precision))
        z_observed = self._get_vars(f, self.cfg.observed_variables, idx, dtype=str_to_dtype(self.cfg.variables_precision))
        
        return X_sequence, y_target, z_observed