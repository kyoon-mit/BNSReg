from dataclasses import dataclass, fields
from functools import cached_property
import os.path

@dataclass(frozen=True, slots=True)
class BNSDatasetConfig():
    hdf5_path: str

    target_variables: tuple[str, ...]
    observed_variables: tuple[str, ...]

    # strain variables
    strain_frequency: int = 2048  # Hz
    strain_duration: int = 64     # sec
    coalescence_time: int = 63    # sec
    window_begin: int = 0         # sec
    window_end: int = 55          # sec
    downsample_factor: int = 1

    # I/O variables
    strain_precision: str = 'torch.float16'
    variables_precision: str = 'torch.float16'
    rdcc_nbytes: int = 512 * 1024**2,
    rdcc_nslots: int = 50_021,
    rdcc_w0: int = 0.75

    def __post_init__(self):
        valid_keys = {
            'chi1', 'chi2', 'chirp_mass', 'dec', 'distance', 'inclination',
            'mass_1', 'mass_2', 'mass_ratio', 'phi', 'phic', 'psi',
            's1z', 's2z', 'snr'
        }
        targets = set(self.target_variables)
        observed = set(self.observed_variables)

        invalid = (targets | observed) - valid_keys
        if invalid:
            raise ValueError(
                f'Invalid target_variables: {invalid}. Valid options are {valid_keys}.'
            )
        
        overlap = targets & observed
        if overlap:
            raise ValueError(f'Variables cannot be both target and observed: {overlap}.')

@dataclass(frozen=True, slots=True)
class BNSDataModuleConfig(BNSDatasetConfig):
    train_file: str
    test_file: str
    val_file: str
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int
    num_workers: int
    shuffle: bool = True
    random_seed: int = 1234
    prefetch_factor: int = 2
    persistent_workers: bool = True
    hdf5_path: str = '' # base hdf5_path is unused here; keep but make it explicit

    def __post_init__(self):
        super().__post_init__()
        for stage, path in (
            ('Train', self.train_file),
            ('Test', self.test_file),
            ('Validation', self.val_file),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f'{stage} file {path} does not exist.')

        object.__setattr__(self, '_base_dataset_kwargs_cache', None)
        object.__setattr__(self, '_base_dataloader_kwargs_cache', None)

    @property
    def _base_dataset_kwargs(self) -> dict[str, object]:
        cache = self._base_dataset_kwargs_cache
        if cache is None:
            cache = {
                f.name: getattr(self, f.name)
                for f in fields(BNSDatasetConfig)
                if f.name != 'hdf5_path'
            }
            object.__setattr__(self, '_base_dataset_kwargs_cache', cache)
        return cache
    
    @property
    def _base_dataloader_kwargs(self) -> dict[str, object]:
        cache = self._base_dataloader_kwargs_cache
        if cache is None:
            cache = dict(
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                random_seed=self.random_seed,
            )
        object.__setattr__(self, '_base_dataloader_kwargs_cache', cache)
        return cache

    def dataset_kwargs(self, stage: str) -> dict[str, object]:
        if stage == 'train':
            path = self.train_file
        elif stage == 'test':
            path = self.test_file
        elif stage == 'val':
            path = self.val_file
        else:
            raise ValueError(f'Stage={stage} must be one of "train", "test", "val"')
        return {'hdf5_path': path, **self._base_dataset_kwargs}
    
    def dataloader_kwargs(self, stage: str) -> dict[str, object]:
        if stage == 'train':
            batch_size = self.train_batch_size
            shuffle = self.shuffle
        elif stage == 'test':
            batch_size = self.test_batch_size
            shuffle = False
        elif stage == 'val':
            batch_size = self.val_batch_size
            shuffle = False
        else:
            raise ValueError(f'Stage={stage} must be one of "train", "test", "val"')
        return {'batch_size': batch_size, 'shuffle': shuffle, **self._base_dataloader_kwargs}