import warnings
import torch
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore', message='The `srun` command is available on your system but is not used.')


class FitTestCLI(LightningCLI):
    """LightningCLI subclass that automatically runs test with best checkpoint after fit."""

    def after_fit(self):
        ckpt_path = None if self.trainer.fast_dev_run else 'best'
        self.trainer.test(self.model, self.datamodule, ckpt_path=ckpt_path)


def cli_main():
    FitTestCLI(save_config_callback=None, subclass_mode_model=True, subclass_mode_data=True)


if __name__ == '__main__':
    cli_main()
