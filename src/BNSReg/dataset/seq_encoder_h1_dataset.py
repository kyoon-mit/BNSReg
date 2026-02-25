import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset, Stage
from BNSReg.core.config import BNSDatasetConfig
from BNSReg.utils.config_tools import str_to_dtype

class BNSDatasetSeqEncoderH1(BNSBaseDataset):
    def __init__(self, stage: Stage, cfg: BNSDatasetConfig):
        super().__init__(stage, cfg)
        self._set_index()

    def __getitem__(self, idx):
        f = self._get_file()

        injected_seq = f[self.cfg.injected_data_key][idx, 0:1, self.start_idx:self.end_idx:self.cfg.downsample_factor]
        sig_only_seq = f[self.cfg.sig_only_data_key][idx, 0:1, self.start_idx:self.end_idx:self.cfg.downsample_factor]

        input = torch.as_tensor(injected_seq, dtype=str_to_dtype(self.cfg.strain_precision))
        target = torch.as_tensor(sig_only_seq, dtype=str_to_dtype(self.cfg.strain_precision))
        
        return input, target