import warnings
import torch
from lightning.pytorch.cli import LightningCLI

from BNSReg.callbacks.save_config import SaveConfigCallback

torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore', message='The `srun` command is available on your system but is not used.')


class BNSRegCLI(LightningCLI):
    """LightningCLI subclass that injects SaveConfigCallback before trainer creation."""

    def before_instantiate_classes(self):
        # Inject SaveConfigCallback into whatever callbacks the YAML specifies,
        # so the config YAML is always copied to the run output directory.
        sub = self.subcommand  # 'fit', 'test', etc.
        if sub is None:
            return
        trainer_cfg = self.config.get(sub, {}).get('trainer', {})
        callbacks = trainer_cfg.get('callbacks', []) or []
        entry = {'class_path': 'BNSReg.callbacks.save_config.SaveConfigCallback',
                 'init_args': {}}
        # Only add if not already present
        already = any(
            isinstance(cb, dict) and cb.get('class_path', '').endswith('SaveConfigCallback')
            for cb in callbacks
        )
        if not already:
            callbacks.append(entry)
        trainer_cfg['callbacks'] = callbacks
        self.config[sub]['trainer'] = trainer_cfg


def cli_main():
    BNSRegCLI(save_config_callback=None, subclass_mode_model=True)


if __name__ == '__main__':
    cli_main()
