import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset
from BNSReg.utils.config_tools import str_to_dtype

class BNSDatasetSeqEncoder(BNSBaseDataset):
    def __getitem__(self, idx):
        f = self._get_file()

        start_idx = self.cfg.window_begin * self.cfg.strain_frequency
        end_idx = self.cfg.window_end * self.cfg.strain_frequency

        injected_seq = f['injected_data'][idx, :, start_idx:end_idx:self.cfg.downsample_factor]
        sig_only_seq = f['sig_only_data'][idx, :, start_idx:end_idx:self.cfg.downsample_factor]

        input = torch.as_tensor(injected_seq, dtype=str_to_dtype(self.cfg.strain_precision))
        target = torch.as_tensor(sig_only_seq, dtype=str_to_dtype(self.cfg.strain_precision))
        
        return input, target