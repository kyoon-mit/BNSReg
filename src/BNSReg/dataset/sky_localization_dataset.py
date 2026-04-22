

##### NECESSARY ?? #####

import h5py
from typing import Literal

import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset, Stage
from BNSReg.core.config import BNSDataModuleRegressionConfig
from BNSReg.utils.config_tools import str_to_dtype

Stage = Literal['train', 'test', 'val']

class SkyLocalizationDataset(BNSBaseDataset):
    def __init__(self, stage: Stage, cfg: BNSDataModuleRegressionConfig):
        self.cfg = cfg
        self.stage = stage
        self._f = None
        self.start_idx = None
        self.end_idx = None

        try:
            self.file_path = {
                'train': self.cfg.train_file,
                'val': self.cfg.val_file,
                'test': self.cfg.test_file,
            }[stage]
        except KeyError:
            raise ValueError(f'Invalid stage: {stage}')

        with h5py.File(self.file_path, 'r') as f:
            self.n_samples, self.n_ifos, self.full_len = f[self.cfg.sig_only_data_key].shape

        if not (self.cfg.strain_duration * self.cfg.strain_frequency) == self.full_len:
            raise ValueError(f'Length of sequence = {self.full_len} does not match ' +
            f'strain_duration ({self.cfg.strain_duration}) * strain_frequency ({self.cfg.strain_frequency}) provided in the configuration.')

    def __getitem__(self, idx):
        f = self._get_file()

        L1_sequence = f[self.cfg.injected_data_key][idx, 0:1, self.start_idx:self.end_idx:self.cfg.downsample_factor]
        H1_sequence = f[self.cfg.injected_data_key][idx, 1:2, self.start_idx:self.end_idx:self.cfg.downsample_factor]

        L1_sequence = torch.as_tensor(L1_sequence, dtype=str_to_dtype(self.cfg.strain_precision))
        H1_sequence = torch.as_tensor(H1_sequence, dtype=str_to_dtype(self.cfg.strain_precision))
        y_target = self._get_vars(f, self.cfg.target_variables, idx, dtype=str_to_dtype(self.cfg.variables_precision))
        z_observed = self._get_vars(f, self.cfg.observed_variables, idx, dtype=str_to_dtype(self.cfg.variables_precision))
        
        return L1_sequence, H1_sequence, y_target, z_observed