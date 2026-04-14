import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset, Stage
from BNSReg.core.config import BNSDataModuleRegressionConfig
from BNSReg.utils.config_tools import str_to_dtype


class BNSDatasetJointDenoise(BNSBaseDataset):
    """Dataset for joint denoising + regression training.

    Returns (x_noisy, x_clean, y_target, z_observed) so that the task can
    simultaneously train a reconstruction head (x_noisy → x_clean) and a
    regression head (features → physical parameters).

    Requires:
        cfg.injected_data_key  — noisy injected strain
        cfg.sig_only_data_key  — clean signal-only strain
        cfg.target_variables   — physical parameters to regress
        cfg.observed_variables — conditioning observables (e.g. SNR)
    """

    def __init__(self, stage: Stage, cfg: BNSDataModuleRegressionConfig):
        super().__init__(stage, cfg)
        self._set_index()

        self.var_scales: dict[str, tuple[float, float]] = {}
        if cfg.normalize_variables:
            all_vars = tuple(cfg.target_variables) + tuple(cfg.observed_variables)
            self.var_scales = self._compute_var_minmax(cfg.train_file, all_vars)

    def _normalize(self, tensor: torch.Tensor, keys: tuple[str, ...]) -> torch.Tensor:
        if not keys:
            return tensor
        lo, hi = self.cfg.normalize_range
        mins = torch.tensor([self.var_scales[k][0] for k in keys], dtype=tensor.dtype)
        maxs = torch.tensor([self.var_scales[k][1] for k in keys], dtype=tensor.dtype)
        return lo + (hi - lo) * (tensor - mins) / (maxs - mins)

    def __getitem__(self, idx):
        f            = self._get_file()
        strain_dtype = str_to_dtype(self.cfg.strain_precision)
        var_dtype    = str_to_dtype(self.cfg.variables_precision)

        x_noisy = self._read_strain(f, self.cfg.injected_data_key, idx)
        x_clean = self._read_strain(f, self.cfg.sig_only_data_key, idx)

        y_target   = self._get_vars(f, self.cfg.target_variables,   idx, dtype=var_dtype)
        z_observed = self._get_vars(f, self.cfg.observed_variables,  idx, dtype=var_dtype)

        if self.var_scales:
            y_target   = self._normalize(y_target,   self.cfg.target_variables)
            z_observed = self._normalize(z_observed, self.cfg.observed_variables)

        return (
            torch.as_tensor(x_noisy, dtype=strain_dtype),  # (n_ifos, L)
            torch.as_tensor(x_clean, dtype=strain_dtype),  # (n_ifos, L)
            y_target,                                       # (n_vars,)
            z_observed,                                     # (n_observed,)
        )
