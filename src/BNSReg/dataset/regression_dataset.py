import warnings
import h5py
import numpy as np
import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset, Stage
from BNSReg.core.config import BNSDataModuleRegressionConfig
from BNSReg.utils.config_tools import str_to_dtype

class BNSDatasetRegression(BNSBaseDataset):
    def __init__(self, stage: Stage, cfg: BNSDataModuleRegressionConfig) -> None:
        super().__init__(stage, cfg)
        self._set_index()

        self.var_scales: dict[str, tuple[float, float]] = {}  # key -> (min, max)
        all_vars = tuple(cfg.target_variables) + tuple(cfg.observed_variables)
        if cfg.normalize_variables:
            self.var_scales = self._compute_var_minmax(cfg.train_file, all_vars)

        # Build index of valid (non-NaN) samples
        self._valid_indices = self._build_valid_index(all_vars)

    def _build_valid_index(self, keys: tuple[str, ...]) -> np.ndarray:
        with h5py.File(self.file_path, 'r') as f:
            stacked = np.stack([f[k][:] for k in keys], axis=1)  # (N, n_vars)
        valid = np.where(~np.any(np.isnan(stacked), axis=1))[0]
        n_skipped = self.n_samples - len(valid)
        if n_skipped > 0:
            warnings.warn(
                f'[{self.__class__.__name__}] {self.file_path}: skipping {n_skipped} sample(s) '
                f'({100*n_skipped/self.n_samples:.3f}%) with NaN in {list(keys)}.',
                UserWarning,
                stacklevel=2,
            )
        return valid

    def __len__(self) -> int:
        return len(self._valid_indices)

    def _normalize(self, tensor: torch.Tensor, keys: tuple[str, ...]) -> torch.Tensor:
        if not keys:
            return tensor
        lo, hi = self.cfg.normalize_range
        mins = torch.tensor([self.var_scales[k][0] for k in keys], dtype=tensor.dtype)
        maxs = torch.tensor([self.var_scales[k][1] for k in keys], dtype=tensor.dtype)
        return lo + (hi - lo) * (tensor - mins) / (maxs - mins)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self._get_file()
        dtype = str_to_dtype(self.cfg.variables_precision)

        idx = int(self._valid_indices[idx])
        seq = self._read_strain(f, self.cfg.injected_data_key, idx)
        X_sequence = torch.as_tensor(seq, dtype=str_to_dtype(self.cfg.strain_precision))
        y_target = self._get_vars(f, self.cfg.target_variables, idx, dtype=dtype)
        z_observed = self._get_vars(f, self.cfg.observed_variables, idx, dtype=dtype)

        if self.var_scales:
            y_target = self._normalize(y_target, self.cfg.target_variables)
            z_observed = self._normalize(z_observed, self.cfg.observed_variables)

        return X_sequence, y_target, z_observed
