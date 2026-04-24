import time
import torch
from lightning.pytorch.callbacks import Callback


class EfficiencyCallback(Callback):
    """Logs step time, epoch wall-clock time, throughput, and peak GPU memory.

    Metrics logged (all to W&B via pl_module.log):
      eff/step_time_ms          — forward+backward+optimizer, per step
      eff/throughput_samples_s  — batch_size / step_time, per step
      eff/epoch_wall_time_s     — full epoch wall clock (includes callbacks/IO), per epoch
      eff/peak_gpu_mem_mb       — peak GPU memory allocated during the epoch, per epoch
    """

    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()
        self._epoch_t0 = time.perf_counter()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Start step timer just before the optimizer touches weights so we
        # capture the full forward+backward+optimizer time.
        self._step_t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        step_s = time.perf_counter() - self._step_t0
        batch_size = batch[0].shape[0]

        pl_module.log('eff/step_time_ms', step_s * 1e3, on_step=True, on_epoch=False, prog_bar=False)
        pl_module.log('eff/throughput_samples_s', batch_size / step_s, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_s = time.perf_counter() - self._epoch_t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

        pl_module.log('eff/epoch_wall_time_s', epoch_s, on_epoch=True, prog_bar=False)
        pl_module.log('eff/peak_gpu_mem_mb', peak_mb, on_epoch=True, prog_bar=False)
