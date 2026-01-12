from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class BNSDatasetConfig():
    hdf5_path: str

    target_variables: tuple[str, ...]
    observed_variables: tuple[str, ...]

    shuffle: bool = True
    random_seed: int = 1234

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
    
    def return_dict(self):
        pass