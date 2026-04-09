import os
import h5py
from typing import Literal

import torch
from torch.utils.data import Dataset

from BNSReg.core.config import BNSDatasetConfig

Stage = Literal['train', 'test', 'val']

class BNSBaseDataset(Dataset):
    def __init__(self, stage: Stage, cfg: BNSDatasetConfig):
        self.cfg = cfg
        self.stage = stage
        self._f = None
        self._f_pid = None  # PID that opened self._f; reopen on fork
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
            self.n_samples, self.n_ifos, self.full_len = f[self.cfg.injected_data_key].shape

        if not (self.cfg.strain_duration * self.cfg.strain_frequency) == self.full_len:
            raise ValueError(f'Length of sequence = {self.full_len} does not match ' +
            f'strain_duration ({self.cfg.strain_duration}) * strain_frequency ({self.cfg.strain_frequency}) provided in the configuration.')
    
    def _get_file(self):
        pid = os.getpid()
        if self._f is None or self._f_pid != pid:
            if self._f is not None:
                try:
                    self._f.close()
                except Exception:
                    pass
            self._f = h5py.File(
                self.file_path, 'r',
                rdcc_nbytes=self.cfg.rdcc_nbytes,
                rdcc_nslots=self.cfg.rdcc_nslots,
                rdcc_w0=self.cfg.rdcc_w0,
            )
            self._f_pid = pid
        return self._f
    
    def _compute_var_minmax(self, file_path: str, keys: tuple[str, ...]) -> dict[str, tuple[float, float]]:
        stats: dict[str, tuple[float, float]] = {}
        with h5py.File(file_path, 'r') as f:
            for k in keys:
                data = f[k][:]
                stats[k] = (float(data.min()), float(data.max()))
        return stats

    def _get_vars(self, f: h5py.File, keys: tuple[str, ...], idx: int, dtype: torch.dtype) -> torch.Tensor:
        tensor_list = []
        if keys:
            for k in keys:
                tensor_list.append(torch.as_tensor(f[k][idx], dtype=dtype))
            return torch.stack(tensor_list)
        else:
            return torch.empty(0, dtype=dtype)
    
    def _set_index(self):
        if self.start_idx and self.end_idx:
            return

        start_idx = self.cfg.window_begin * self.cfg.strain_frequency
        end_idx = self.cfg.window_end * self.cfg.strain_frequency

        if start_idx.is_integer() and end_idx.is_integer():
            self.start_idx, self.end_idx = int(start_idx), int(end_idx)
        else:
            raise ValueError('Sequence data slice index values are not integers. Please check.\n'\
                            f'start_idx = window_begin ({self.cfg.window_begin}) * strain_frequency ({self.cfg.strain_frequency})\n'\
                            f'end_idx = window_end ({self.cfg.window_end}) * strain_frequency ({self.cfg.strain_frequency})\n')
        return

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        raise NotImplementedError