import h5py

import torch
from torch.utils.data import Dataset

from BNSReg.core.config import BNSDatasetConfig

class BNSBaseDataset(Dataset):
    def __init__(self, cfg: BNSDatasetConfig):
        self.cfg = cfg
        self._f = None

        with h5py.File(self.cfg.hdf5_path, 'r') as f:
            self.n_samples, self.n_ifos, self.full_len = f['data'].shape

        if not (self.cfg.strain_duration * self.cfg.strain_frequency) == self.full_len:
            raise ValueError(f'Length of sequence = {self.full_len} does not match ' +
            f'strain_duration ({self.cfg.strain_duration}) * strain_frequency ({self.cfg.strain_frequency}) provided in the configuration.')
    
    def _get_file(self):
        if self._f is None:
            self._f = h5py.File(self.cfg.hdf5_path, 'r')
        return self._f
    
    def _get_vars(self, f, keys, idx, dtype=torch.float16):
        tensor_list = []
        for k in keys:
            tensor_list.append(torch.as_tensor(f[k][idx], dtype=dtype))
        return torch.stack(tensor_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        raise NotImplementedError