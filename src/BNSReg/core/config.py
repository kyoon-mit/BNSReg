from dataclasses import dataclass, fields
from typing import TypeAlias
import os.path

@dataclass(frozen=True, slots=True, kw_only=True)
class BNSDatasetConfig():
    train_file: str
    test_file: str
    val_file: str
    injected_data_key : str = 'data'
    sig_only_data_key : str = ''
    bkg_only_data_key: str = ''

    # strain variables
    strain_frequency: int = 2048  # Hz
    downsample_factor: int = 1
    strain_duration: float = 64.0   # sec
    coalescence_time: float = 63.0  # sec
    window_begin: float = 0.0       # sec
    window_end: float = 55.0        # sec

    # bandpass filter
    apply_bandpass: bool = False
    bandpass_low: float = 0.0     # Hz; 0 → lowpass only
    bandpass_high: float = 3e4    # Hz; >= Nyquist → highpass only

    # I/O variables
    strain_precision: str = 'torch.float32'
    variables_precision: str = 'torch.float32'
    rdcc_nbytes: int = 64 * 1024**2
    rdcc_nslots: int = 10_007
    rdcc_w0: float = 0.75

    def __post_init__(self) -> None:
        # Check if files exist
        for stage, path in (
            ('Train', self.train_file),
            ('Test', self.test_file),
            ('Validation', self.val_file),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f'{stage} file {path} does not exist.')

@dataclass(frozen=True, slots=True, kw_only=True)
class BNSDataModuleConfig(BNSDatasetConfig):
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int
    num_workers: int = 0
    shuffle: bool = True
    prefetch_factor: int | None = None
    persistent_workers: bool = False

    # internal
    _dataloader_kwargs_cache: dict[str, object] | None = None

    @property
    def _dataloader_dict(self) -> dict[str, object]:
        cache = self._dataloader_kwargs_cache
        if cache is None:
            cache = dict(
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )
        object.__setattr__(self, '_dataloader_kwargs_cache', cache)
        return cache

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

    # normalization: min/max computed from training file at dataset init
    normalize_variables: bool = False
    normalize_range: tuple[float, float] = (-1.0, 1.0)  # [lo, hi] output range

    def __post_init__(self) -> None:
        # Check if files exist
        for stage, path in (
            ('Train', self.train_file),
            ('Test', self.test_file),
            ('Validation', self.val_file),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f'{stage} file {path} does not exist.')

        # Normalization checks
        lo, hi = self.normalize_range
        if lo >= hi:
            raise ValueError(f'normalize_range must satisfy lo < hi, got {self.normalize_range}.')

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

BNSDataModuleClassificationConfig: TypeAlias = BNSDataModuleRegressionConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class BNSDataModuleCurriculumConfig(BNSDataModuleRegressionConfig):
    """Curriculum training with per-stage mixture weights over multiple train files.

    train_files[i] is one SNR bin (e.g. high → low).
    stage_schedule[i] is the epoch at which stage i begins; must start with 0.
    stage_weights[i] is the sampling probability over train_files for stage i;
      weights need not sum to 1 (they are normalised at runtime).
      Set a weight to 0 for hard exclusion (equivalent to Option A switching).

    Example — hard switch across 3 bins:
      stage_weights = [(1,0,0), (0,1,0), (0,0,1)]

    Example — smooth mixing:
      stage_weights = [(0.8,0.15,0.05), (0.4,0.4,0.2), (0.2,0.4,0.4)]

    train_file is unused; it is set automatically to train_files[0] so the
    parent validation logic still works.
    """
    train_files: tuple[str, ...]
    stage_schedule: tuple[int, ...]
    stage_weights: tuple[tuple[float, ...], ...]
    # train_file is inherited from BNSDatasetConfig; set it to "" in YAML —
    # __post_init__ will replace it with train_files[0] before validation.

    def __post_init__(self) -> None:
        n = len(self.train_files)
        if n == 0:
            raise ValueError("train_files must not be empty.")
        if len(self.stage_schedule) != len(self.stage_weights):
            raise ValueError(
                "stage_schedule and stage_weights must have the same length."
            )
        if self.stage_schedule[0] != 0:
            raise ValueError("stage_schedule[0] must be 0.")
        for i in range(1, len(self.stage_schedule)):
            if self.stage_schedule[i] <= self.stage_schedule[i - 1]:
                raise ValueError("stage_schedule must be strictly increasing.")
        for i, w in enumerate(self.stage_weights):
            if len(w) != n:
                raise ValueError(
                    f"stage_weights[{i}] has length {len(w)}, expected {n} "
                    f"(must match len(train_files))."
                )
            if any(v < 0 for v in w):
                raise ValueError(f"stage_weights[{i}] contains negative values.")
            if sum(w) <= 0:
                raise ValueError(f"stage_weights[{i}] must have at least one positive value.")
        for path in self.train_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Train file not found: {path}")
        object.__setattr__(self, "train_file", self.train_files[0])
        super(BNSDataModuleCurriculumConfig, self).__post_init__()


@dataclass(frozen=True, slots=True)
class BNSModelConfig():
    def model_kwargs(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

@dataclass(frozen=True, slots=True)
class S4DModelConfig(BNSModelConfig):
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

    # input pre-processing
    input_norm: bool = False  # per-channel InstanceNorm1d along the time axis before the encoder

    def __post_init__(self) -> None:
        if self.dt_min >= self.dt_max:
            raise ValueError('dt_min must be < dt_max')

@dataclass(frozen=True, slots=True)
class S4DDenoisedRegressionConfig(S4DModelConfig):
    """S4DModelConfig extended with paths to a frozen denoiser checkpoint.

    denoiser_ckpt and denoiser_cfg are excluded from model_kwargs() so they
    are not forwarded to the S4Model constructor.
    """
    denoiser_ckpt: str | None = None   # path to LitModelS4DMSE .ckpt file
    denoiser_cfg:  str | None = None   # path to the denoiser's config YAML

    def model_kwargs(self) -> dict[str, object]:
        exclude = {'denoiser_ckpt', 'denoiser_cfg'}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in exclude}

@dataclass(frozen=True, slots=True)
class S4DResNetMLPModelConfig(S4DModelConfig):
    """S4DModelConfig extended with a ResNet1D + MLP decoder head."""
    resnet_layers: tuple[int, ...] = (2, 2, 2)
    resnet_latent_dim: int = 64
    mlp_width: int = 64
    mlp_depth: int = 2

@dataclass(frozen=True, slots=True)
class LinOSSModelConfig(BNSModelConfig):
    d_input: int
    d_output: int
    d_model: int
    ssm_size: int
    n_layers: int
    dropout: float
    discretization: str = 'IM'

    def __post_init__(self) -> None:
        if self.discretization not in ('IM', 'IMEX'):
            raise ValueError(f"discretization must be 'IM' or 'IMEX', got '{self.discretization}'")

@dataclass(frozen=True, slots=True)
class ConvAEModelConfig(BNSModelConfig):
    n_layers: int
    latent_channels: int
    kernel_size: int
    pool_stride: int

@dataclass(frozen=True, slots=True)
class ConvAEAPModelConfig(BNSModelConfig):
    seq_length: int
    n_layers: int
    base_dim: int

@dataclass(frozen=True, slots=True)
class ConvAttentionAEAPModelConfig(ConvAEAPModelConfig):
    num_heads: int
    dropout: float