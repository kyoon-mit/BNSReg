import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy
import lightning as L
import h5py

class BNSDataset(Dataset):
    def __init__(self,
        hdf5_path,
        variables, # set of variables
        downsample_factor=1,
        start_time=64,
        end_time=128,
        sample_rate=2048,
        scale_factor=1.,
        normalize=False,
        include_snr=False,
        data_key='data',
        shift_start_time=0,
        shift_end_time=64,
    ):
        super().__init__()
        self.valid_keys = {
            'chi1', 'chi2', 'chirp_mass', 'dec', 'distance', 'inclination',
            'mass_1', 'mass_2', 'mass_ratio', 'phi', 'phic', 'psi', 's1z', 's2z', 'snr'
        }
        self.valid_keys.update({k + '_normalized' for k in self.valid_keys})
        self.derived_keys = {'total_mass'}
        self.h5file = h5py.File(hdf5_path, 'r')
        # if True:
        # with h5py.File(hdf5_path, 'r') as h5file:
            # self.coalescence_time = self.h5file.attrs['coalescence_time'] # Time of coalescence
            # self.duration = h5file.attrs['duration'] # Duration of the waveform in seconds
            # self.ifos = h5file.attrs['ifos'] # List of interferometers
            # self.length = h5file.attrs['length'] # Number of samples.
            # self.num_injections = h5file.attrs['num_injections'] # Number of waveform injections.
            # self.sample_rate = self.h5file.attrs['sample_rate'] # Sample rate in Hz
        self.sample_rate = sample_rate # Sample rate in Hz
            # self.waveforms_h1 = self.h5file['waveforms/h1']
            # self.waveforms_l1 = self.h5file['waveforms/l1']
        self.data = self.h5file[data_key]
            # self.param_data_from_file = self.h5file['parameters']
            # self.length = self.waveforms_h1.shape[0]
        self.length = self.data.shape[0]
        self.keys = set(variables) & (self.valid_keys | self.derived_keys)
        if not self.keys:
            raise ValueError(f'Valid variables are: {self.valid_keys}.')
        self.downsample_factor = int(downsample_factor)
        self.start_time, self.end_time = start_time, end_time
        self.scale_factor = scale_factor
        self.normalize = normalize
        self.include_snr = include_snr
        self.shift_start_time = shift_start_time
        self.shift_end_time = shift_end_time

    def _prepare_params_(self, idx: int):
        """
        Compute derived parameters and update the params dict.
        Derived parameters include chirp_mass, mass_ratio, and total_mass.
        """
        params = {}

        # compute derived only if requested
        # if 'chirp_mass' in self.keys:
        #     params['chirp_mass'] = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        # if 'mass_ratio' in self.keys:
        #     params['mass_ratio'] = m2 / m1
        if {'total_mass', 'chirp_mass', 'mass_ratio'} & self.keys:
            # load masses from file if available (safe access)
            m1 = torch.tensor(self.h5file['mass_1'][idx], dtype=torch.float32)
            m2 = torch.tensor(self.h5file['mass_2'][idx], dtype=torch.float32)
            # params['total_mass'] = m1 + m2 # <-- ?

        # add any other requested scalar parameters that exist
        for k in (self.keys - self.derived_keys):
            if k == 'redshift' and 'distance' in self.h5file:
                # if user asked for redshift but file has distance, map if appropriate
                params['redshift'] = torch.tensor(self.h5file['distance'][idx], dtype=torch.float32)
            else:
                params[k] = torch.tensor(self.h5file[k][idx], dtype=torch.float32)

        if self.include_snr:
            params['snr'] = torch.tensor(self.h5file['snr'][idx], dtype=torch.float32)
        
        return params

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load waveforms
        # h1 = self.scale_factor * torch.tensor(self.waveforms_h1[idx][::self.downsample_factor], dtype=torch.float32)
        # l1 = self.scale_factor * torch.tensor(self.waveforms_l1[idx][::self.downsample_factor], dtype=torch.float32)
        start_idx = int(self.start_time * self.sample_rate)
        end_idx = int(self.end_time * self.sample_rate)
        shift_factor = int(np.random.rand(self.shift_start_time - self.start_time, self.shift_end_time - self.start_time)*self.sample_rate)

        h1 = self.scale_factor * torch.tensor(self.data[idx][0][start_idx:end_idx:self.downsample_factor], dtype=torch.float32)
        l1 = self.scale_factor * torch.tensor(self.data[idx][1][start_idx + shift_factor:end_idx + shift_factor:self.downsample_factor], dtype=torch.float32)

        if self.normalize:
            h1 = (h1 - h1.mean()) / (h1.std())
            l1 = (l1 - l1.mean()) / (l1.std())
            
        # Load parameters as a dictionary
        params = self._prepare_params_(idx)
        return (h1, l1, params, idx)

class LitBNSDataModule(L.LightningDataModule):
    def __init__(self,
        variables, # set of variables to include in the dataset
        train_file='',
        test_file='',
        val_file='',
        downsample_factor=1,
        start_time=0,
        end_time=64,
        scale_factor=1.,
        sample_rate=2048,
        normalize=False,
        train_batch_size=1000,
        val_batch_size=1000,
        test_batch_size=1000,
        random_seed=42,
        include_snr=False,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
        data_key='data',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.dataset_args = dict(
            variables=variables,
            downsample_factor=downsample_factor,
            start_time=start_time,
            end_time=end_time,
            sample_rate=sample_rate,
            normalize=normalize,
            scale_factor=scale_factor,
            include_snr=include_snr,
            data_key=data_key
        )

    def prepare_data(self):
        if hasattr(self, 'dataset'):
            # already prepared
            return
        if self.train_file and self.test_file and self.val_file:
            # Check if the file exists
            import os.path
            if not os.path.exists(self.train_file):
                raise ValueError(f'Train file {self.train_file} does not exist.')
            elif not os.path.exists(self.test_file):
                raise ValueError(f'Test file {self.test_file} does not exist.')
            elif not os.path.exists(self.val_file):
                raise ValueError(f'Validation file {self.val_file} does not exist.')
        else:
            raise ValueError('Missing paths to the train, test, or val files.')

    def setup(self, stage: str):
        if not hasattr(self, 'dataset'):
            self.prepare_data()
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # instantiate train and val datasets separately
            self.train_dataset = BNSDataset(self.train_file, **self.dataset_args)
            self.val_dataset = BNSDataset(self.val_file, **self.dataset_args)

        # Assign test dataset for use in dataloader
        elif stage in ("test", "predict"):
            self.test_dataset = BNSDataset(self.test_file, **self.dataset_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          persistent_workers=self.persistent_workers)