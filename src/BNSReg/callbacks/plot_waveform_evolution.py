"""WaveformEvolutionCallback
=========================
Tracks how a denoiser's reconstruction approaches the target waveform over training.

A fixed batch of validation examples is captured once at the start of the fit and
reused for every epoch, so successive frames are directly comparable. At the end of
each validation epoch the callback runs the model on that batch and draws one figure:
target, prediction, and (optionally) the noisy input, per example and per IFO.

The figure is logged to Weights & Biases under a single stable key, so the run's
media panel exposes a step slider that scrubs through training — the animation.
Frames are also written to disk as PNGs and, at the end of the fit, assembled into
an animated GIF that is logged as a wandb Video.

Example (LightningCLI config):

    trainer:
      callbacks:
        - class_path: BNSReg.callbacks.plot_waveform_evolution.WaveformEvolutionCallback
          init_args:
            n_examples: 3
            every_n_epochs: 1
            sample_rate: 256
            zoom_seconds: 2.0
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger


class WaveformEvolutionCallback(Callback):
    """Log a per-epoch figure showing prediction vs target on a fixed val batch.

    Args:
        n_examples: number of validation examples to track.
        every_n_epochs: draw a frame every N validation epochs.
        sample_rate: strain sample rate in Hz, used for the time axis. None plots
            against sample index.
        plot_window: `[begin, end]` in absolute seconds selecting the segment to
            plot, e.g. `[26.0, 30.0]` for the 4 s leading into a merger at 30 s.
            None plots the whole strain window. Requires `sample_rate`. Every
            column uses this same segment.
        show_input: overlay the noisy input on the time-domain and FFT columns
            as light grey. It is never drawn on the spectrogram columns, where
            it would saturate the colour scale and hide the target and
            prediction.
        show_spectrogram: append two more columns, target and prediction
            spectrograms, to the right of the time and FFT columns. Off by
            default since it is the most expensive part of the figure.
        spec_nperseg: STFT segment length in samples for the spectrogram
            columns. Smaller keeps the chirp's sweep sharp in time, larger
            sharpens it in frequency.
        spec_overlap: fraction of `spec_nperseg` that consecutive STFT
            segments overlap.
        spec_dynamic_range: dB below the target's peak to clip the shared
            colour scale at. Both spectrograms use the target's scale so they
            are directly comparable.
        spec_flim: `[low, high]` frequency limits in Hz for the spectrogram
            y-axis, which is log scaled. None autoscales.
        out_dir: where PNG frames and the GIF are written. Defaults to
            `<trainer.log_dir>/waveform_evolution`.
        gif_fps: frames per second of the assembled GIF.
        max_gif_frames: cap on frames in the GIF; frames are subsampled evenly
            above this count so long runs stay small.
    """

    def __init__(
        self,
        n_examples: int = 4,
        every_n_epochs: int = 1,
        sample_rate: float | None = None,
        window_begin: float = 0.0,
        plot_window: tuple[float, float] | None = None,
        show_input: bool = True,
        show_spectrogram: bool = False,
        spec_nperseg: int = 256,
        spec_overlap: float = 0.75,
        spec_dynamic_range: float = 60.0,
        spec_flim: tuple[float, float] | None = (20.0, 512.0),
        out_dir: str | None = None,
        gif_fps: int = 4,
        max_gif_frames: int = 200,
    ):
        super().__init__()
        self.n_examples = n_examples
        self.every_n_epochs = every_n_epochs
        self.sample_rate = sample_rate
        self.window_begin = window_begin
        self.plot_window = plot_window
        self.show_input = show_input
        self.show_spectrogram = show_spectrogram
        self.spec_nperseg = spec_nperseg
        self.spec_overlap = spec_overlap
        self.spec_dynamic_range = spec_dynamic_range
        self.spec_flim = tuple(spec_flim) if spec_flim else None
        self.out_dir = Path(out_dir) if out_dir else None
        self.gif_fps = gif_fps
        self.max_gif_frames = max_gif_frames

        self._fixed_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._frame_paths: list[Path] = []

    # ── setup ────────────────────────────────────────────────────────────────
    def on_fit_start(self, trainer, pl_module) -> None:
        if self.out_dir is None:
            root = Path(trainer.log_dir or '.')
            self.out_dir = root / 'waveform_evolution'
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _capture_batch(self, trainer, pl_module) -> None:
        """Pull the first `n_examples` from the val dataloader, once."""
        loader = trainer.datamodule.val_dataloader() if trainer.datamodule else None
        if loader is None:
            return
        noisy, target = next(iter(loader))
        n = min(self.n_examples, noisy.shape[0])
        self._fixed_batch = (
            noisy[:n].to(pl_module.device),
            target[:n].to(pl_module.device),
        )

    # ── per-epoch frame ──────────────────────────────────────────────────────
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if self._fixed_batch is None:
            self._capture_batch(trainer, pl_module)
            if self._fixed_batch is None:
                return

        noisy, target = self._fixed_batch          # (N, n_ifos, L)

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            # S4ModelSeq2Seq takes and returns (B, channels, L).
            pred = pl_module(noisy)                    # (N, d_output, L)
        if was_training:
            pl_module.train()

        fig = self._draw(
            noisy.float().cpu().numpy(),
            target.float().cpu().numpy(),
            pred.float().cpu().numpy(),
            epoch=trainer.current_epoch,
        )

        # Recreate the directory each time: a frame must never take the run down
        # if the output tree is removed mid-run.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f'epoch_{trainer.current_epoch:04d}.png'
        fig.savefig(path, dpi=110, bbox_inches='tight')
        self._frame_paths.append(path)

        # One stable key: the wandb media panel turns the step axis into a slider.
        if isinstance(trainer.logger, WandbLogger):
            import wandb
            trainer.logger.experiment.log(
                {'waveforms/evolution': wandb.Image(fig)},
                step=trainer.global_step,
            )
        plt.close(fig)

    # ── GIF at the end ───────────────────────────────────────────────────────
    def _assemble_gif(self, frame_paths: list[Path], gif_name: str) -> Path | None:
        from PIL import Image

        paths = [p for p in frame_paths if p.exists()]
        if len(paths) < 2:
            return None
        if len(paths) > self.max_gif_frames:
            idx = np.linspace(0, len(paths) - 1, self.max_gif_frames).astype(int)
            paths = [paths[i] for i in idx]

        frames = [Image.open(p).convert('P', palette=Image.ADAPTIVE) for p in paths]
        gif_path = self.out_dir / gif_name
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / self.gif_fps),
            loop=0,
        )
        return gif_path

    def on_fit_end(self, trainer, pl_module) -> None:
        if not self._frame_paths:
            return

        if self.out_dir is None:
            root = Path(trainer.log_dir or '.')
            self.out_dir = root / 'waveform_evolution'
        self.out_dir.mkdir(parents=True, exist_ok=True)

        gif_path = self._assemble_gif(self._frame_paths, 'evolution.gif')
        if gif_path and isinstance(trainer.logger, WandbLogger):
            import wandb
            trainer.logger.experiment.log(
                {'waveforms/evolution_gif': wandb.Video(str(gif_path), fps=self.gif_fps)}
            )

    # ── plotting ─────────────────────────────────────────────────────────────
    def _time_axis(self, length: int) -> tuple[np.ndarray, str]:
        # Absolute seconds, so the axis matches the strain window in the config
        # (e.g. 26-30 s) rather than restarting at zero.
        if self.sample_rate:
            return self.window_begin + np.arange(length) / self.sample_rate, 'time [s]'
        return np.arange(length), 'sample'

    def _spectrogram(
        self, x: np.ndarray, t_offset: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """STFT spectrogram of a 1-D segment, in dB.

        An STFT is used rather than a Q transform: the Q transform costs far
        more per frame and this runs every epoch. Short segments keep the
        chirp's frequency sweep visible in time, which a single spectrum
        integrates away.

        Returns (times, freqs, power_db) with the DC bin dropped so a log
        frequency axis has no zero-frequency row. `t_offset` shifts the time
        axis into the same absolute seconds the waveform panel uses.
        """
        from scipy.signal import spectrogram as _scipy_spectrogram

        fs = self.sample_rate or 1.0
        nperseg = min(self.spec_nperseg, x.shape[0])
        noverlap = int(nperseg * self.spec_overlap)
        freqs, times, power = _scipy_spectrogram(
            x,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='spectrum',
            mode='psd',
        )
        power_db = 10.0 * np.log10(np.maximum(power[1:], 1e-30))
        return times + t_offset, freqs[1:], power_db

    def _fft(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw one-sided rfft magnitude of a 1-D segment: |rfft(x)|.

        No window, no power normalization; matches the quantity compared by
        MixtureMSESpectralLoss / DynamicMixtureLoss (torch.fft.rfft(x).abs()),
        unlike `_psd` which windows and normalizes into a power density.
        Drops the DC bin so the log-log axes have no zero-frequency point.
        """
        fs = self.sample_rate or 1.0
        n = x.shape[0]
        mag = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        return freqs[1:], np.maximum(mag[1:], 1e-30)

    def _draw(
        self,
        noisy: np.ndarray,
        target: np.ndarray,
        pred: np.ndarray,
        epoch: int,
    ):
        """noisy/target/pred: (N, n_ifos, L). Returns a matplotlib Figure."""
        n_ex, n_ifos, length = target.shape
        t, xlabel = self._time_axis(length)

        # Segment to plot, from the requested [begin, end] in absolute seconds.
        lo, hi = 0, length
        if self.plot_window and self.sample_rate:
            begin, end = self.plot_window
            lo = max(0, int((begin - self.window_begin) * self.sample_rate))
            hi = min(length, int((end - self.window_begin) * self.sample_rate))
        sl = slice(lo, hi)

        t_offset = t[lo] if self.sample_rate else 0.0

        # Time domain and FFT always; spectrograms only when asked for.
        n_cols = 4 if self.show_spectrogram else 2
        n_rows = n_ex * n_ifos
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(7 * n_cols, 8.8 * n_rows),
            squeeze=False,
        )

        for e in range(n_ex):
            for k in range(n_ifos):
                row = e * n_ifos + k
                mse = float(np.mean((pred[e, k, sl] - target[e, k, sl]) ** 2))

                # ── column 0: time domain over the requested window ──────────
                ax = axes[row][0]
                if self.show_input:
                    ax.plot(t[sl], noisy[e, k, sl], lw=0.5, color='0.6',
                            alpha=0.45, label='noisy input')
                ax.plot(t[sl], target[e, k, sl], lw=0.9, color='k', label='target')
                ax.plot(t[sl], pred[e, k, sl], lw=0.9, color='tab:red', label='prediction')
                ax.set_ylabel(f'ex {e} / ifo {k}')
                ax.set_title(f'MSE = {mse:.3e}', fontsize=8, loc='right')
                if row == 0:
                    ax.legend(fontsize=7, ncol=3, loc='upper left')
                if row == n_rows - 1:
                    ax.set_xlabel(xlabel)

                # ── column 1: |rfft|, the quantity the spectral loss compares ─
                axf = axes[row][1]
                if self.show_input:
                    f, m = self._fft(noisy[e, k, sl])
                    axf.loglog(f, m, lw=0.6, color='0.6', alpha=0.45, label='noisy input')
                f, m = self._fft(target[e, k, sl])
                axf.loglog(f, m, lw=0.9, color='k', label='target')
                f, m = self._fft(pred[e, k, sl])
                axf.loglog(f, m, lw=0.9, color='tab:red', label='prediction')
                axf.set_ylabel('|rfft|')
                if row == 0:
                    axf.set_title('abs(rfft) magnitude spectrum', fontsize=9)
                    axf.legend(fontsize=7, loc='upper right')
                if row == n_rows - 1:
                    axf.set_xlabel('frequency [Hz]')

                if not self.show_spectrogram:
                    continue

                # ── columns 2/3: target vs prediction spectrogram ────────────
                # The noisy input is deliberately not drawn here: at these SNRs
                # it saturates the colour scale and hides both other panels.
                spec_t = self._spectrogram(target[e, k, sl], t_offset)
                spec_p = self._spectrogram(pred[e, k, sl], t_offset)

                # One colour scale for both panels, set by the target, so the
                # two are directly comparable rather than each auto-scaled.
                vmax = float(spec_t[2].max())
                vmin = vmax - self.spec_dynamic_range

                for col, (label, (st, sf, sdb)) in enumerate(
                    (('target', spec_t), ('prediction', spec_p)), start=2
                ):
                    axs = axes[row][col]
                    mesh = axs.pcolormesh(
                        st, sf, sdb,
                        shading='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                    )
                    axs.set_yscale('log')
                    if self.spec_flim:
                        axs.set_ylim(*self.spec_flim)
                    if row == 0:
                        axs.set_title(f'{label} spectrogram', fontsize=9)
                    if row == n_rows - 1:
                        axs.set_xlabel(xlabel)
                    if col == 2:
                        axs.set_ylabel('frequency [Hz]')
                    if col == 3:
                        fig.colorbar(mesh, ax=axs, pad=0.01, label='power [dB]')

        fig.suptitle(f'epoch {epoch}', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        return fig
