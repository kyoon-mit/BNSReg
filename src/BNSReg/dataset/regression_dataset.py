import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset
from BNSReg.utils.config_tools import str_to_dtype

class BNSDatasetRegression(BNSBaseDataset):
    def __getitem__(self, idx):
        f = self._get_file()

        start_idx = self.cfg.window_begin * self.cfg.strain_frequency
        end_idx = self.cfg.window_end * self.cfg.strain_frequency

        seq = f['data'][idx, :, start_idx:end_idx:self.cfg.downsample_factor]

        X_sequence = torch.as_tensor(seq, dtype=str_to_dtype(self.cfg.strain_precision))
        y_target = self._get_vars(f, self.cfg.target_variables, idx, dtype=str_to_dtype(self.cfg.variables_precision))
        z_observed = self._get_vars(f, self.cfg.observed_variables, idx, dtype=str_to_dtype(self.cfg.variables_precision))
        
        return X_sequence, y_target, z_observed