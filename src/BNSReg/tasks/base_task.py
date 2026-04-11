import importlib
from typing import get_type_hints

import yaml
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


class LitBaseTask(LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # Callback: write loss, y_target, z_observed to file
        return loss


class LitDenoiserBaseTask(LitBaseTask):
    """Base task that optionally prepends a frozen denoiser to any input sequence.

    Subclasses call `self._setup_denoiser(ckpt_path, cfg_path)` in their
    `__init__`, then use `self._prepare_input(X_sequence)` to transparently
    apply the denoiser (or skip it when no checkpoint is provided).
    """

    @staticmethod
    def _load_denoiser(ckpt_path: str, cfg_path: str) -> nn.Module:
        """Load any frozen Lightning denoiser task from a checkpoint + config YAML.

        The class is resolved dynamically from the YAML's `model.class_path`, and
        its config arguments are reconstructed using the class's type hints —
        no hardcoded model or config class.

        Args:
            ckpt_path: path to the Lightning .ckpt file.
            cfg_path:  path to the corresponding LightningCLI config YAML.

        Returns:
            The instantiated task in eval mode with requires_grad=False.
        """
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        mc = cfg['model']
        mod, cls_name = mc['class_path'].rsplit('.', 1)
        cls = getattr(importlib.import_module(mod), cls_name)

        # Reconstruct typed init args from the task's own type hints.
        # e.g. cfg: S4DModelConfig is instantiated from the dict in the YAML.
        hints = get_type_hints(cls.__init__)
        kwargs = {}
        for param, val in mc.get('init_args', {}).items():
            if isinstance(val, dict) and param in hints:
                kwargs[param] = hints[param](**val)
            else:
                kwargs[param] = val

        task = cls(**kwargs)

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        # The checkpoint was saved from a torch.compile'd task (keys have '._orig_mod').
        # The freshly instantiated task also calls configure_model() → torch.compile,
        # so its keys are identical — load directly with no key mangling.
        task.load_state_dict(ckpt['state_dict'])
        task.eval()
        for p in task.parameters():
            p.requires_grad_(False)
        return task

    def _setup_denoiser(self, ckpt_path: str | None, cfg_path: str | None) -> None:
        """Call from subclass __init__ to attach the denoiser (or set it to None)."""
        if ckpt_path and cfg_path:
            self.denoiser = self._load_denoiser(ckpt_path, cfg_path)
        else:
            self.denoiser = None

    def on_train_start(self) -> None:
        """Keep the denoiser in eval mode after Lightning switches the model to train."""
        if self.denoiser is not None:
            self.denoiser.eval()

    def _prepare_input(self, X_sequence: torch.Tensor) -> torch.Tensor:
        """Transpose and optionally denoise the raw input sequence.

        Args:
            X_sequence: (B, n_ifos, L)
        Returns:
            x: (B, L, n_ifos) ready for the downstream model
        """
        x = X_sequence.transpose(1, 2)       # (B, L, d_input)
        if self.denoiser is not None:
            with torch.no_grad():
                x = self.denoiser(x)          # (B, L, d_input) — denoised
        return x
