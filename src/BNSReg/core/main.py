import warnings
import torch
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore', message='The `srun` command is available on your system but is not used.')


def cli_main():
    cli = LightningCLI(save_config_callback=None,
                       subclass_mode_model=True,)

if __name__ == '__main__':
    cli_main()