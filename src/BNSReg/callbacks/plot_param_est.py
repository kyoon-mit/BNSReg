"""PlotParamEstCallback
====================
PyTorch Lightning Callback that, at the end of a test epoch:
  1. Concatenates all per-batch {y_true, y_pred, y_sigma} tensors returned by test_step.
  2. Optionally un-normalizes values when normalize_variables=True.
  3. Saves a CSV with true, predicted, and sigma columns for every target variable.
  4. Generates four diagnostic figures:
       param_hist.png    – overlaid true/predicted histograms for each variable
       pred_vs_true.png  – binned mean/median/68%/95% CI per variable
       sigma_hist.png    – predicted-sigma histograms
       z_score_hist.png  – z-score histograms
     When both 'chirp_mass' and 'mass_ratio' are target variables,
     m1 and m2 (derived via Jacobian propagation) are appended to Figs 2–4.

Bin edges are taken from the BINS dict below; widths are always nice numbers
(1 / 2 / 2.5 / 5 × 10^n) inferred from the BNS prior distributions.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback


# ── Bin edges (inferred from BNS_SNR5-50 prior YAML) ─────────────────────────
# Each linspace is chosen so that bin width = (hi-lo)/(n-1) is a nice number.

BINS: dict[str, np.ndarray] = {
    'chirp_mass':  np.linspace(0.5,   3.0,  51),   # w = 0.05
    'mass_ratio':  np.linspace(0.0,   1.0,  51),   # w = 0.02
    'mass_1':      np.linspace(1.0,   2.5,  31),   # w = 0.05
    'mass_2':      np.linspace(1.0,   2.5,  31),   # w = 0.05
    'm1':          np.linspace(1.0,   2.5,  31),   # w = 0.05  (derived)
    'm2':          np.linspace(1.0,   2.5,  31),   # w = 0.05  (derived)
    'distance':    np.linspace(100.,  1000., 37),  # w = 25 Mpc
    'a_1':         np.linspace(0.0,   1.0,  21),   # w = 0.05
    'a_2':         np.linspace(0.0,   1.0,  21),   # w = 0.05
    'tilt_1':      np.linspace(0.0,   3.2,  33),   # w = 0.1 rad
    'tilt_2':      np.linspace(0.0,   3.2,  33),   # w = 0.1 rad
    'psi':         np.linspace(0.0,   3.2,  33),   # w = 0.1 rad
    'inclination': np.linspace(0.0,   3.2,  33),   # w = 0.1 rad
    'phi_12':      np.linspace(0.0,   6.4,  33),   # w = 0.2 rad
    'phi_jl':      np.linspace(0.0,   6.4,  33),   # w = 0.2 rad
    'phic':        np.linspace(0.0,   6.4,  33),   # w = 0.2 rad
    'phi':         np.linspace(-3.2,  3.2,  65),   # w = 0.1 rad
    'dec':         np.linspace(-1.6,  1.6,  33),   # w = 0.1 rad
    'snr':         np.linspace(5.0,  50.0,  46),   # w = 1
    's1z':         np.linspace(-1.0,  1.0,  41),   # w = 0.05
    's2z':         np.linspace(-1.0,  1.0,  41),   # w = 0.05
    'chi1':        np.linspace(-1.0,  1.0,  41),   # w = 0.05
    'chi2':        np.linspace(-1.0,  1.0,  41),   # w = 0.05
}

# ── LaTeX labels ──────────────────────────────────────────────────────────────
VAR_LABELS: dict[str, str] = {
    'chirp_mass':  r'$\mathcal{M}_c\,[M_\odot]$',
    'mass_ratio':  r'$q$',
    'mass_1':      r'$m_1\,[M_\odot]$',
    'mass_2':      r'$m_2\,[M_\odot]$',
    'm1':          r'$m_1\,[M_\odot]$',
    'm2':          r'$m_2\,[M_\odot]$',
    'distance':    r'$d_L\,[\mathrm{Mpc}]$',
    'a_1':         r'$a_1$',
    'a_2':         r'$a_2$',
    'tilt_1':      r'$\theta_1\,[\mathrm{rad}]$',
    'tilt_2':      r'$\theta_2\,[\mathrm{rad}]$',
    'phi_12':      r'$\phi_{12}\,[\mathrm{rad}]$',
    'phi_jl':      r'$\phi_{JL}\,[\mathrm{rad}]$',
    'phic':        r'$\phi_c\,[\mathrm{rad}]$',
    'psi':         r'$\psi\,[\mathrm{rad}]$',
    'phi':         r'$\phi\,[\mathrm{rad}]$',
    'dec':         r'$\delta\,[\mathrm{rad}]$',
    'inclination': r'$\iota\,[\mathrm{rad}]$',
    'snr':         r'SNR',
    's1z':         r'$s_{1z}$',
    's2z':         r'$s_{2z}$',
    'chi1':        r'$\chi_1$',
    'chi2':        r'$\chi_2$',
}


def _label(key: str) -> str:
    return VAR_LABELS.get(key, key)


def _sig_label(key: str) -> str:
    inner = _label(key).strip('$')
    return rf'$\sigma_{{{inner}}}$'


def _bins_for(key: str, data: np.ndarray) -> np.ndarray:
    """Return bin edges: use BINS if defined, else auto-linspace over data range."""
    if key in BINS:
        return BINS[key]
    lo, hi = float(data.min()), float(data.max())
    return np.linspace(lo, hi, 51)


def _hist_outline(ax, data: np.ndarray, bins: np.ndarray, **kwargs) -> None:
    counts, edges = np.histogram(data, bins=bins)
    ax.stairs(counts, edges, **kwargs)


# ── Derived-quantity helpers (chirp_mass + mass_ratio → m1, m2) ──────────────

def _mc_q_to_m1_m2(mc: np.ndarray, q: np.ndarray):
    m1 = mc * (1.0 + q) ** 0.2 / q ** 0.6
    return m1, q * m1


def _propagate_m1_m2_uncertainty(mc, q, sig_mc, sig_q):
    m1, m2 = _mc_q_to_m1_m2(mc, q)
    dm1_dmc = m1 / mc
    dm1_dq  = m1 * (1.0 / (5.0 * (1.0 + q)) - 3.0 / (5.0 * q))
    dm2_dmc = m2 / mc
    dm2_dq  = m1 + q * dm1_dq
    sig_m1 = np.sqrt((dm1_dmc * sig_mc) ** 2 + (dm1_dq * sig_q) ** 2)
    sig_m2 = np.sqrt((dm2_dmc * sig_mc) ** 2 + (dm2_dq * sig_q) ** 2)
    return sig_m1, sig_m2


# ── Callback ──────────────────────────────────────────────────────────────────

class PlotParamEstCallback(Callback):
    """
    Accumulates test-batch outputs and generates eval plots at test epoch end.

    ``target_variables``, ``normalize_variables``, ``normalize_range``, and
    ``var_scales`` are all auto-read from ``trainer.datamodule`` at test start,
    so only ``save_dir`` is required in the config.

    Usage (YAML config)::

        callbacks:
          - class_path: BNSReg.callbacks.plot_param_est.PlotParamEstCallback
            init_args:
              save_dir: /path/to/outputs/plots

    Args:
        save_dir: Directory where CSV and PNG figures are saved.
    """

    def __init__(self, save_dir: str | Path = ''):
        super().__init__()
        self._save_dir_override = Path(save_dir) if save_dir else None
        self.save_dir = self._save_dir_override or Path('.')

        # Populated in on_test_start from trainer.datamodule
        self.target_variables:   list[str] = []
        self.observed_variables: list[str] = []
        self.normalize_variables: bool = False
        self.var_scales: dict[str, tuple[float, float]] = {}
        self.normalize_range: tuple[float, float] = (-1.0, 1.0)

        self._y_true:     list[torch.Tensor] = []
        self._y_pred:     list[torch.Tensor] = []
        self._y_sigma:    list[torch.Tensor] = []
        self._z_observed: list[torch.Tensor] = []

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def on_test_start(self, trainer, pl_module):
        self._y_true.clear()
        self._y_pred.clear()
        self._y_sigma.clear()
        self._z_observed.clear()

        # Resolve save_dir: explicit override > {logger.save_dir}/{project}/{run_id}/plots > fallback
        if self._save_dir_override:
            self.save_dir = self._save_dir_override
        elif hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir is not None:
            # Access experiment first to force wandb init, then try multiple id sources.
            exp = getattr(trainer.logger, 'experiment', None)
            run_id = (
                getattr(exp, 'id', None)
                or getattr(trainer.logger, 'version', None)
                or getattr(trainer.logger, 'id', None)
            )
            if run_id is not None:
                base = Path(trainer.logger.save_dir)
                project = getattr(exp, 'project', None)
                if project:
                    base = base / project
                self.save_dir = base / run_id / 'plots'

        # Read variable config from the datamodule
        cfg = trainer.datamodule.cfg
        self.target_variables    = list(cfg.target_variables)
        self.observed_variables  = list(cfg.observed_variables) if hasattr(cfg, 'observed_variables') else []
        self.normalize_variables = cfg.normalize_variables
        self.normalize_range     = tuple(cfg.normalize_range)
        if self.normalize_variables:
            # var_scales are computed from the training file and stored on each dataset
            self.var_scales = dict(trainer.datamodule.test_dataset.var_scales)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        self._y_true.append(outputs['y_true'])
        self._y_pred.append(outputs['y_pred'])
        self._y_sigma.append(outputs['y_sigma'])
        if 'z_observed' in outputs:
            self._z_observed.append(outputs['z_observed'])

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._y_true:
            return

        y_true  = torch.cat(self._y_true).numpy()   # (N, n_vars)
        y_pred  = torch.cat(self._y_pred).numpy()
        y_sigma = torch.cat(self._y_sigma).numpy()
        z_observed = torch.cat(self._z_observed).numpy() if self._z_observed else None

        if self.normalize_variables and self.var_scales:
            y_true  = self._unnormalize_values(y_true)
            y_pred  = self._unnormalize_values(y_pred)
            y_sigma = self._unnormalize_sigma(y_sigma)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_csv(y_true, y_pred, y_sigma, z_observed)
        self._plot(y_true, y_pred, y_sigma)

    # ── Un-normalization ──────────────────────────────────────────────────────

    def _unnormalize_values(self, arr: np.ndarray) -> np.ndarray:
        lo, hi = self.normalize_range
        out = arr.copy()
        for i, var in enumerate(self.target_variables):
            if var in self.var_scales:
                vmin, vmax = self.var_scales[var]
                out[:, i] = vmin + (vmax - vmin) * (arr[:, i] - lo) / (hi - lo)
        return out

    def _unnormalize_sigma(self, arr: np.ndarray) -> np.ndarray:
        lo, hi = self.normalize_range
        out = arr.copy()
        for i, var in enumerate(self.target_variables):
            if var in self.var_scales:
                vmin, vmax = self.var_scales[var]
                out[:, i] = arr[:, i] * (vmax - vmin) / (hi - lo)
        return out

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _save_csv(self, y_true: np.ndarray, y_pred: np.ndarray, y_sigma: np.ndarray,
                  z_observed: np.ndarray | None = None):
        cols = (
            [f'{v}_true'       for v in self.target_variables] +
            [f'{v}_pred'       for v in self.target_variables] +
            [f'sigma_{v}_pred' for v in self.target_variables]
        )
        arrays = [y_true, y_pred, y_sigma]
        if z_observed is not None and self.observed_variables:
            cols += list(self.observed_variables)
            arrays.append(z_observed)
        pd.DataFrame(
            np.concatenate(arrays, axis=1),
            columns=cols,
        ).to_csv(self.save_dir / 'param_est_results.csv', index=False)

    # ── Figures ───────────────────────────────────────────────────────────────

    def _plot(self, y_true: np.ndarray, y_pred: np.ndarray, y_sigma: np.ndarray):
        vars_ = self.target_variables
        true_d  = {v: y_true[:,  i] for i, v in enumerate(vars_)}
        pred_d  = {v: y_pred[:,  i] for i, v in enumerate(vars_)}
        sigma_d = {v: y_sigma[:, i] for i, v in enumerate(vars_)}

        # ── Optionally derive m1, m2 when both chirp_mass and mass_ratio present
        vars_234 = list(vars_)
        if 'chirp_mass' in vars_ and 'mass_ratio' in vars_:
            mc_t, q_t = true_d['chirp_mass'],  true_d['mass_ratio']
            mc_p, q_p = pred_d['chirp_mass'],  pred_d['mass_ratio']
            s_mc, s_q = sigma_d['chirp_mass'], sigma_d['mass_ratio']
            m1t, m2t = _mc_q_to_m1_m2(mc_t, q_t)
            m1p, m2p = _mc_q_to_m1_m2(mc_p, q_p)
            s_m1, s_m2 = _propagate_m1_m2_uncertainty(mc_p, q_p, s_mc, s_q)
            for key, tr, pr, sg in [('m1', m1t, m1p, s_m1), ('m2', m2t, m2p, s_m2)]:
                true_d[key]  = tr
                pred_d[key]  = pr
                sigma_d[key] = sg
            vars_234 = vars_ + ['m1', 'm2']

        z_d = {v: (pred_d[v] - true_d[v]) / sigma_d[v] for v in vars_234}

        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                             'axes.spines.right': False})

        # ── Figure 1: true vs predicted histograms (target vars only) ────────
        n1 = len(vars_)
        ncols1 = min(n1, 4)
        nrows1 = math.ceil(n1 / ncols1)
        fig1, axes1 = plt.subplots(nrows1, ncols1, figsize=(5*ncols1, 4*nrows1), squeeze=False)
        for ax, v in zip(axes1.flat, vars_):
            bins = _bins_for(v, true_d[v])
            _hist_outline(ax, true_d[v], bins=bins, color='red',   linewidth=1.2, label='True')
            _hist_outline(ax, pred_d[v], bins=bins, color='green', linewidth=1.2, label='Predicted')
            ax.set_xlabel(_label(v))
            ax.set_ylabel('Counts')
            ax.legend(frameon=False)
        for ax in axes1.flat[n1:]:
            ax.set_visible(False)
        fig1.tight_layout()

        # ── Figures 2-4: all vars including derived m1, m2 ───────────────────
        n234 = len(vars_234)
        ncols2 = min(n234, 4)
        nrows2 = math.ceil(n234 / ncols2)

        # Figure 2: binned pred vs true
        fig2, axes2 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4.5*nrows2), squeeze=False)
        for ax, v in zip(axes2.flat, vars_234):
            edges = _bins_for(v, true_d[v])
            centers = 0.5 * (edges[:-1] + edges[1:])
            tr, pr = true_d[v], pred_d[v]

            mean_p   = np.full(len(centers), np.nan)
            median_p = np.full(len(centers), np.nan)
            q16      = np.full(len(centers), np.nan)
            q84      = np.full(len(centers), np.nan)
            q2p5     = np.full(len(centers), np.nan)
            q97p5    = np.full(len(centers), np.nan)

            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                mask = (tr >= lo) & (tr < hi)
                if mask.sum() < 2:
                    continue
                p_bin        = pr[mask]
                mean_p[i]    = np.mean(p_bin)
                median_p[i]  = np.median(p_bin)
                q16[i]       = np.percentile(p_bin, 16)
                q84[i]       = np.percentile(p_bin, 84)
                q2p5[i]      = np.percentile(p_bin,  2.5)
                q97p5[i]     = np.percentile(p_bin, 97.5)

            data_lo = float(np.nanmin(tr))
            data_hi = float(np.nanmax(tr))
            mask_e  = (edges >= data_lo) & (edges <= data_hi)
            valid   = np.where(mask_e[:-1] & mask_e[1:])[0]
            cents_c = centers[valid]
            lims    = [data_lo, data_hi]

            ax.plot(lims, lims, color='gray', alpha=0.5, linewidth=1.2, linestyle=':')
            ax.step(cents_c, q2p5[valid],   where='mid', color='steelblue', lw=0.8, ls='dotted', alpha=0.7, label='95%')
            ax.step(cents_c, q97p5[valid],  where='mid', color='steelblue', lw=0.8, ls='dotted', alpha=0.7)
            ax.step(cents_c, q16[valid],    where='mid', color='steelblue', lw=0.8, ls='dashed', alpha=0.9, label='68%')
            ax.step(cents_c, q84[valid],    where='mid', color='steelblue', lw=0.8, ls='dashed', alpha=0.9)
            ax.step(cents_c, median_p[valid], where='mid', color='steelblue', lw=1.4, label='Median')
            ax.step(cents_c, mean_p[valid], where='mid', color='purple', lw=1.2, alpha=0.8, label='Mean')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')
            ax.set_xlabel('True ' + _label(v))
            ax.set_ylabel('Pred ' + _label(v))
            ax.legend(frameon=False, fontsize=9, loc='upper left')
        for ax in axes2.flat[n234:]:
            ax.set_visible(False)
        fig2.tight_layout()

        # Figure 3: sigma histograms
        fig3, axes3 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4*nrows2), squeeze=False)
        for ax, v in zip(axes3.flat, vars_234):
            _hist_outline(ax, sigma_d[v], bins=20, color='black', linewidth=1.2)
            ax.set_xlabel(_sig_label(v))
            ax.set_ylabel('Counts')
        for ax in axes3.flat[n234:]:
            ax.set_visible(False)
        fig3.tight_layout()

        # Figure 4: z-score histograms
        fig4, axes4 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4*nrows2), squeeze=False)
        for ax, v in zip(axes4.flat, vars_234):
            _hist_outline(ax, z_d[v], bins=20, color='steelblue', linewidth=1.2)
            ax.axvline( 0, color='gray', linestyle='--',     linewidth=1.0)
            ax.axvline( 1, color='gray', linestyle='dotted', linewidth=1.0)
            ax.axvline(-1, color='gray', linestyle='dotted', linewidth=1.0)
            inner = _label(v).strip('$')
            ax.set_xlabel(rf'$(x_{{\mathrm{{pred}}}} - x_{{\mathrm{{true}}}})\,/\,\sigma_{{{inner}}}$')
            ax.set_ylabel('Counts')
        for ax in axes4.flat[n234:]:
            ax.set_visible(False)
        fig4.tight_layout()

        # Save
        names = ['param_hist', 'pred_vs_true', 'sigma_hist', 'z_score_hist']
        for fig, name in zip([fig1, fig2, fig3, fig4], names):
            out = self.save_dir / f'{name}.png'
            fig.savefig(out, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out}')
