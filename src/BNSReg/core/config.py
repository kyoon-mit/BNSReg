from dataclasses import dataclass, fields
from typing import Self
import os.path

@dataclass(frozen=True, slots=True, kw_only=True)
class BNSDatasetConfig():
    hdf5_path: str | None = None

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
    rdcc_nbytes: int = 64 * 1024**2
    rdcc_nslots: int = 10_007
    rdcc_w0: int = 0.75

@dataclass(frozen=True, slots=True, kw_only=True)
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

    # internal
    _dataloader_kwargs_cache: dict[str, object] | None = None

    def __post_init__(self):
        # Check if files exist
        for stage, path in (
            ('Train', self.train_file),
            ('Test', self.test_file),
            ('Validation', self.val_file),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f'{stage} file {path} does not exist.')
    
    @property
    def _dataloader_dict(self) -> dict[str, object]:
        cache = self._dataloader_kwargs_cache
        if cache is None:
            cache = dict(
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                random_seed=self.random_seed,
            )
        object.__setattr__(self, '_dataloader_kwargs_cache', cache)
        return cache

    def get_cfg(self, stage: str) -> Self:
        if stage == 'train':
            path = self.train_file
        elif stage == 'test':
            path = self.test_file
        elif stage == 'val':
            path = self.val_file
        else:
            raise ValueError('stage must be train|test|val')

        object.__setattr__(self, 'hdf5_path', path)
        return self

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
        return {'batch_size': batch_size, 'shuffle': shuffle, **self._dataloader_dict}

@dataclass(frozen=True, slots=True, kw_only=True)
class BNSDataModuleRegressionConfig(BNSDataModuleConfig):
    target_variables: tuple[str, ...]
    observed_variables: tuple[str, ...]

    def __post_init__(self):
        # Check if files exist
        for stage, path in (
            ('Train', self.train_file),
            ('Test', self.test_file),
            ('Validation', self.val_file),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f'{stage} file {path} does not exist.')
        
        # Check variables
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
class S4DModelConfig():
    d_input: int
    d_output: int
    d_model: int
    d_state: int
    n_layers: int
    dropout: float

    # kernel arguments
    dt_min: float = 0.001
    dt_max: float = 0.1
    lr: float | None = None

    def model_kwargs(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}