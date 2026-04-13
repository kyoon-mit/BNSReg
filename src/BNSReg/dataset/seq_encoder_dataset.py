import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset, Stage
from BNSReg.core.config import BNSDatasetConfig
from BNSReg.utils.config_tools import str_to_dtype

class BNSDatasetSeqEncoder(BNSBaseDataset):
    def __init__(self, stage: Stage, cfg: BNSDatasetConfig):
        super().__init__(stage, cfg)
        self._set_index()

    def __getitem__(self, idx):
        f = self._get_file()

        injected_seq = self._read_strain(f, self.cfg.injected_data_key, idx)
        sig_only_seq = self._read_strain(f, self.cfg.sig_only_data_key, idx)

        dtype = str_to_dtype(self.cfg.strain_precision)
        input = torch.as_tensor(injected_seq, dtype=dtype)
        target = torch.as_tensor(sig_only_seq, dtype=dtype)

        return input, target