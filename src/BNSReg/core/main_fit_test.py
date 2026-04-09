from lightning.pytorch.cli import LightningCLI


class FitTestCLI(LightningCLI):
    """LightningCLI subclass that automatically runs test with best checkpoint after fit."""

    def after_fit(self):
        ckpt_path = None if self.trainer.fast_dev_run else 'best'
        self.trainer.test(self.model, self.datamodule, ckpt_path=ckpt_path)


def cli_main():
    FitTestCLI(save_config_callback=None, subclass_mode_model=True)


if __name__ == '__main__':
    cli_main()
