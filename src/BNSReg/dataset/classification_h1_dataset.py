import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset, Stage
from BNSReg.core.config import BNSDataModuleClassificationConfig
from BNSReg.utils.config_tools import str_to_dtype

class BNSDatasetClassificationH1(BNSBaseDataset):
    def __init__(self, stage: Stage, cfg: BNSDataModuleClassificationConfig):
        super().__init__(stage, cfg)
        self._set_index()

    def __getitem__(self, idx):
        f = self._get_file()

        injected_seq = self._read_strain(f, self.cfg.injected_data_key, idx, slice(0, 1))
        bkg_only_seq = self._read_strain(f, self.cfg.bkg_only_data_key, idx, slice(0, 1))

        injected_seq = torch.as_tensor(injected_seq, dtype=str_to_dtype(self.cfg.strain_precision))
        bkg_only_seq = torch.as_tensor(bkg_only_seq, dtype=str_to_dtype(self.cfg.strain_precision))
        z_observed = self._get_vars(f, self.cfg.observed_variables, idx, dtype=str_to_dtype(self.cfg.variables_precision))
        
        return injected_seq, bkg_only_seq, z_observed