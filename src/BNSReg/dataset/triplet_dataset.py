import torch

from BNSReg.dataset.base_dataset import BNSBaseDataset, Stage
from BNSReg.core.config import BNSDatasetConfig
from BNSReg.utils.config_tools import str_to_dtype


class BNSDatasetTriplet(BNSBaseDataset):
    """Dataset returning (injected, clean_signal, background) triplets.

    Used by the contrastive denoising task (LitModelS4DContrastive) to
    provide anchor/positive/negative training signal.

    Requires:
        cfg.injected_data_key  — noisy injected strain
        cfg.sig_only_data_key  — clean signal only
        cfg.bkg_only_data_key  — background-only noise (must be non-empty)

    Returns:
        x_inj  (n_ifos, L): noisy injected strain  → anchor after denoising
        x_sig  (n_ifos, L): clean signal            → positive
        x_bkg  (n_ifos, L): background noise        → negative
    """

    def __init__(self, stage: Stage, cfg: BNSDatasetConfig):
        if not cfg.bkg_only_data_key:
            raise ValueError(
                'BNSDatasetTriplet requires cfg.bkg_only_data_key to be set. '
                'Provide the HDF5 key for background-only (noise) samples.'
            )
        super().__init__(stage, cfg)
        self._set_index()

    def __getitem__(self, idx):
        f     = self._get_file()
        dtype = str_to_dtype(self.cfg.strain_precision)

        x_inj = self._read_strain(f, self.cfg.injected_data_key, idx)
        x_sig = self._read_strain(f, self.cfg.sig_only_data_key, idx)
        x_bkg = self._read_strain(f, self.cfg.bkg_only_data_key, idx)

        return (
            torch.as_tensor(x_inj, dtype=dtype),   # (n_ifos, L)
            torch.as_tensor(x_sig, dtype=dtype),
            torch.as_tensor(x_bkg, dtype=dtype),
        )
