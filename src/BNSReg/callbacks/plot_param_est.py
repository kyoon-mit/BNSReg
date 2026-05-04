"""PlotParamEstCallback
====================
Unified PyTorch Lightning Callback for parameter estimation and sky localization.

At the end of a test epoch:
  1. Concatenates all per-batch outputs from test_step.
  2. Optionally un-normalizes values when normalize_variables=True.
  3. Saves a CSV with results for every target variable.
  4. Generates diagnostic figures.

**Standard parameter estimation mode** (e.g. chirp_mass, mass_ratio):
  - param_hist.png       – true/predicted histograms
  - pred_vs_true.png     – binned mean/median/CI per variable
  - sigma_hist.png       – predicted-sigma histograms
  - z_score_hist.png     – z-score histograms
  - snr_binned_residuals.png, snr_fraction_within_cutoff.png, snr_violin.png (if SNR in observed_variables)

**Sky localization mode** (when 'dec' and 'phi' are both in target_variables):
  - sky_hist.png            – true/predicted histograms for dec and phi
  - pred_vs_true.png        – binned pred vs true for dec, phi, cos(theta)
  - angular_error_hist.png  – distribution of angular error in degrees
  - snr_binned.png, snr_vs_angular_cutoff.png, snr_vs_ring_cutoff.png, snr_violin.png (if SNR provided)

Sky mode is detected automatically from target_variables. y_sigma is optional (absent in sky mode).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback


# ── Bin edges ─────────────────────────────────────────────────────────────────
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
    'cos_theta':   np.linspace(-1.0,  1.0,  41),   # w = 0.05
    'snr':         np.linspace(5.0,  50.0,  19),   # w = 2.5
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
    'cos_theta':   r'$\cos\theta$',
    'inclination': r'$\iota\,[\mathrm{rad}]$',
    'snr':         r'SNR',
    's1z':         r'$s_{1z}$',
    's2z':         r'$s_{2z}$',
    'chi1':        r'$\chi_1$',
    'chi2':        r'$\chi_2$',
}

# H1-L1 baseline unit vector (ml4gw Earth-fixed frame)
_H1L1_BASELINE = np.array([0.6953014731407166, -0.5535351634025574, -0.4584263563156128])


def _label(key: str) -> str:
    return VAR_LABELS.get(key, key)


def _sig_label(key: str) -> str:
    inner = _label(key).strip('$')
    return rf'$\sigma_{{{inner}}}$'


def _bins_for(key: str, data: np.ndarray) -> np.ndarray:
    if key in BINS:
        bins = BINS[key]
        # If data lies outside predefined range (e.g. normalized SNR in [0,1]
        # instead of raw 10-50), fall back to auto-range over the actual data.
        if float(data.max()) < bins[0] or float(data.min()) > bins[-1]:
            lo, hi = float(data.min()), float(data.max())
            return np.linspace(lo, hi, len(bins))
        return bins
    lo, hi = float(data.min()), float(data.max())
    return np.linspace(lo, hi, 51)


def _snr_bins(snr_arr: np.ndarray) -> np.ndarray:
    """Unit-width integer SNR bins spanning the full dataset range [a, b]."""
    a = int(np.floor(float(snr_arr.min())))
    b = int(np.ceil(float(snr_arr.max())))
    return np.arange(a, b + 1, dtype=float)


def _snr_xticks(ax, a: int, b: int) -> None:
    """Set x-ticks to always include a and b, spaced ~5 units apart."""
    step = max(2, (b - a) // 5)
    ticks = sorted(set(range(a, b + 1, step)) | {a, b})
    ax.set_xticks(ticks)
    ax.set_xlim(a - 0.5, b + 0.5)


def _hist_outline(ax, data: np.ndarray, bins, **kwargs) -> None:
    counts, edges = np.histogram(data, bins=bins)
    ax.stairs(counts, edges, **kwargs)


def _binned_stats(x, y, bins):
    """Bin y by x; return (centers, median, q16, q84, mae)."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(len(centers), np.nan)
    q16 = np.full(len(centers), np.nan)
    q84 = np.full(len(centers), np.nan)
    mae = np.full(len(centers), np.nan)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 2:
            continue
        yb = y[mask]
        med[i] = np.median(yb)
        q16[i] = np.percentile(yb, 16)
        q84[i] = np.percentile(yb, 84)
        mae[i] = np.mean(np.abs(yb))
    return centers, med, q16, q84, mae


# ── Derived-quantity helpers ───────────────────────────────────────────────────

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


def _angles_to_vec(dec: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """(N,) dec, phi radians → (N, 3) unit vectors."""
    x = np.cos(dec) * np.cos(phi)
    y = np.cos(dec) * np.sin(phi)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


def _vec_to_angles(vec: np.ndarray):
    """(N, 3) unit vectors → dec (N,), phi (N,) in radians."""
    dec = np.arcsin(np.clip(vec[:, 2], -1.0, 1.0))
    phi = np.arctan2(vec[:, 1], vec[:, 0])
    return dec, phi


# ── Callback ──────────────────────────────────────────────────────────────────

class PlotParamEstCallback(Callback):
    """
    Unified eval callback for parameter estimation and sky localization.

    Accumulates test-batch outputs and generates diagnostic plots at test epoch end.
    Sky localization mode is detected automatically when 'dec' and 'phi' are both
    in target_variables.

    ``target_variables``, ``normalize_variables``, ``normalize_range``, and
    ``var_scales`` are auto-read from ``trainer.datamodule`` at test start.

    Args:
        save_dir: Directory where CSV and PNG figures are saved.
    """

    def __init__(self, save_dir: str | Path = ''):
        super().__init__()
        self._save_dir_override = Path(save_dir) if save_dir else None
        self.save_dir = self._save_dir_override or Path('.')

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

        if self._save_dir_override:
            self.save_dir = self._save_dir_override
        elif hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir is not None:
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

        cfg = trainer.datamodule.cfg
        self.target_variables    = list(cfg.target_variables)
        self.observed_variables  = list(cfg.observed_variables) if hasattr(cfg, 'observed_variables') else []
        self.normalize_variables = cfg.normalize_variables
        self.normalize_range     = tuple(cfg.normalize_range)
        if self.normalize_variables:
            self.var_scales = dict(trainer.datamodule.test_dataset.var_scales)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        self._y_true.append(outputs['y_true'])
        self._y_pred.append(outputs['y_pred'])
        if 'y_sigma' in outputs:
            self._y_sigma.append(outputs['y_sigma'])
        if 'z_observed' in outputs:
            self._z_observed.append(outputs['z_observed'])

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._y_true:
            return

        y_true = torch.cat(self._y_true).numpy()
        y_pred = torch.cat(self._y_pred).numpy()
        y_sigma = torch.cat(self._y_sigma).numpy() if self._y_sigma else None
        z_observed = torch.cat(self._z_observed).numpy() if self._z_observed else None

        if self.normalize_variables and self.var_scales:
            y_true = self._unnormalize_values(y_true, self.target_variables)
            y_pred_for_unnorm = y_pred if y_pred.shape[1] == len(self.target_variables) else y_pred
            # Only unnormalize if shapes match (not 3D unit vec)
            if y_pred.shape[1] == len(self.target_variables):
                y_pred = self._unnormalize_values(y_pred, self.target_variables)
            if y_sigma is not None:
                y_sigma = self._unnormalize_sigma(y_sigma, self.target_variables)
            if z_observed is not None and self.observed_variables:
                z_observed = self._unnormalize_values(z_observed, self.observed_variables)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        is_sky = 'dec' in self.target_variables and 'phi' in self.target_variables
        if is_sky:
            self._run_sky(y_true, y_pred, z_observed)
        else:
            self._run_param_est(y_true, y_pred, y_sigma, z_observed)

    # ── Un-normalization ──────────────────────────────────────────────────────

    def _unnormalize_values(self, arr: np.ndarray, variables: list[str]) -> np.ndarray:
        lo, hi = self.normalize_range
        out = arr.copy()
        for i, var in enumerate(variables):
            if var in self.var_scales:
                vmin, vmax = self.var_scales[var]
                out[:, i] = vmin + (vmax - vmin) * (arr[:, i] - lo) / (hi - lo)
        return out

    def _unnormalize_sigma(self, arr: np.ndarray, variables: list[str]) -> np.ndarray:
        lo, hi = self.normalize_range
        out = arr.copy()
        for i, var in enumerate(variables):
            if var in self.var_scales:
                vmin, vmax = self.var_scales[var]
                out[:, i] = arr[:, i] * (vmax - vmin) / (hi - lo)
        return out

    # ── SNR helper ────────────────────────────────────────────────────────────

    def _get_snr(self, z_observed):
        if z_observed is not None and 'snr' in self.observed_variables:
            return z_observed[:, list(self.observed_variables).index('snr')]
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # STANDARD PARAMETER ESTIMATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_param_est(self, y_true, y_pred, y_sigma, z_observed):
        vars_ = self.target_variables
        true_d  = {v: y_true[:, i] for i, v in enumerate(vars_)}
        pred_d  = {v: y_pred[:, i] for i, v in enumerate(vars_)}
        sigma_d = {v: y_sigma[:, i] for i, v in enumerate(vars_)} if y_sigma is not None else {}

        vars_ext = list(vars_)
        if 'chirp_mass' in vars_ and 'mass_ratio' in vars_:
            mc_t, q_t = true_d['chirp_mass'], true_d['mass_ratio']
            mc_p, q_p = pred_d['chirp_mass'], pred_d['mass_ratio']
            m1t, m2t = _mc_q_to_m1_m2(mc_t, q_t)
            m1p, m2p = _mc_q_to_m1_m2(mc_p, q_p)
            for key, tr, pr in [('m1', m1t, m1p), ('m2', m2t, m2p)]:
                true_d[key] = tr
                pred_d[key] = pr
            if sigma_d:
                s_mc, s_q = sigma_d['chirp_mass'], sigma_d['mass_ratio']
                s_m1, s_m2 = _propagate_m1_m2_uncertainty(mc_p, q_p, s_mc, s_q)
                sigma_d['m1'] = s_m1
                sigma_d['m2'] = s_m2
            vars_ext = vars_ + ['m1', 'm2']

        z_d = {}
        if sigma_d:
            z_d = {v: (pred_d[v] - true_d[v]) / (sigma_d[v] + 1e-12) for v in vars_ext}

        self._save_csv_param_est(y_true, y_pred, y_sigma, z_observed)

        snr_arr = self._get_snr(z_observed)
        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

        figs, names = [], []

        # Figure 1: true vs predicted histograms
        n1 = len(vars_)
        ncols1 = min(n1, 4)
        nrows1 = math.ceil(n1 / ncols1)
        fig1, axes1 = plt.subplots(nrows1, ncols1, figsize=(5*ncols1, 4*nrows1), squeeze=False)
        for ax, v in zip(axes1.flat, vars_):
            bins = _bins_for(v, true_d[v])
            _hist_outline(ax, true_d[v], bins=bins, color='red',   linewidth=1.2, label='True')
            _hist_outline(ax, pred_d[v], bins=bins, color='green', linewidth=1.2, label='Predicted')
            ax.set_xlabel(_label(v)); ax.set_ylabel('Counts'); ax.legend(frameon=False)
        for ax in axes1.flat[n1:]:
            ax.set_visible(False)
        fig1.tight_layout()
        figs.append(fig1); names.append('param_hist')

        # Figure 2: binned pred vs true
        n_ext = len(vars_ext)
        ncols2 = min(n_ext, 4)
        nrows2 = math.ceil(n_ext / ncols2)
        fig2, axes2 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4.5*nrows2), squeeze=False)
        for ax, v in zip(axes2.flat, vars_ext):
            edges = _bins_for(v, true_d[v])
            centers = 0.5 * (edges[:-1] + edges[1:])
            tr, pr = true_d[v], pred_d[v]
            mean_p   = np.full(len(centers), np.nan)
            median_p = np.full(len(centers), np.nan)
            q16 = np.full(len(centers), np.nan)
            q84 = np.full(len(centers), np.nan)
            q2p5 = np.full(len(centers), np.nan)
            q97p5 = np.full(len(centers), np.nan)
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                mask = (tr >= lo) & (tr < hi)
                if mask.sum() < 2:
                    continue
                p_bin = pr[mask]
                mean_p[i]   = np.mean(p_bin)
                median_p[i] = np.median(p_bin)
                q16[i]  = np.percentile(p_bin, 16)
                q84[i]  = np.percentile(p_bin, 84)
                q2p5[i] = np.percentile(p_bin, 2.5)
                q97p5[i] = np.percentile(p_bin, 97.5)
            data_lo = float(np.nanmin(tr)); data_hi = float(np.nanmax(tr))
            mask_e = (edges >= data_lo) & (edges <= data_hi)
            valid = np.where(mask_e[:-1] & mask_e[1:])[0]
            cents_c = centers[valid]
            lims = [data_lo, data_hi]
            ax.plot(lims, lims, color='gray', alpha=0.5, linewidth=1.2, linestyle=':')
            ax.step(cents_c, q2p5[valid],    where='mid', color='steelblue', lw=0.8, ls='dotted', alpha=0.7, label='95%')
            ax.step(cents_c, q97p5[valid],   where='mid', color='steelblue', lw=0.8, ls='dotted', alpha=0.7)
            ax.step(cents_c, q16[valid],     where='mid', color='steelblue', lw=0.8, ls='dashed', alpha=0.9, label='68%')
            ax.step(cents_c, q84[valid],     where='mid', color='steelblue', lw=0.8, ls='dashed', alpha=0.9)
            ax.step(cents_c, median_p[valid], where='mid', color='steelblue', lw=1.4, label='Median')
            ax.step(cents_c, mean_p[valid],  where='mid', color='purple',    lw=1.2, alpha=0.8, label='Mean')
            ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
            ax.set_xlabel('True ' + _label(v)); ax.set_ylabel('Pred ' + _label(v))
            ax.legend(frameon=False, fontsize=9, loc='upper left')
        for ax in axes2.flat[n_ext:]:
            ax.set_visible(False)
        fig2.tight_layout()
        figs.append(fig2); names.append('pred_vs_true')

        if sigma_d:
            # Figure 3: sigma histograms
            fig3, axes3 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4*nrows2), squeeze=False)
            for ax, v in zip(axes3.flat, vars_ext):
                _hist_outline(ax, sigma_d[v], bins=20, color='black', linewidth=1.2)
                ax.set_xlabel(_sig_label(v)); ax.set_ylabel('Counts')
            for ax in axes3.flat[n_ext:]:
                ax.set_visible(False)
            fig3.tight_layout()
            figs.append(fig3); names.append('sigma_hist')

            # Figure 4: z-score histograms
            fig4, axes4 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4*nrows2), squeeze=False)
            for ax, v in zip(axes4.flat, vars_ext):
                _hist_outline(ax, z_d[v], bins=20, color='steelblue', linewidth=1.2)
                ax.axvline( 0, color='gray', linestyle='--',     linewidth=1.0)
                ax.axvline( 1, color='gray', linestyle='dotted', linewidth=1.0)
                ax.axvline(-1, color='gray', linestyle='dotted', linewidth=1.0)
                inner = _label(v).strip('$')
                ax.set_xlabel(rf'$(x_{{\mathrm{{pred}}}} - x_{{\mathrm{{true}}}})\,/\,\sigma_{{{inner}}}$')
                ax.set_ylabel('Counts')
            for ax in axes4.flat[n_ext:]:
                ax.set_visible(False)
            fig4.tight_layout()
            figs.append(fig4); names.append('z_score_hist')

        if snr_arr is not None:
            snr_bins = _snr_bins(snr_arr)
            snr_a, snr_b = int(snr_bins[0]), int(snr_bins[-1])
            snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
            ncols5 = min(n_ext, 4)
            nrows5 = math.ceil(n_ext / ncols5)

            # Figure 5: residuals binned by SNR
            fig5, axes5 = plt.subplots(nrows5, ncols5 * 2,
                                       figsize=(5 * ncols5 * 2, 4 * nrows5), squeeze=False)
            for col_idx, v in enumerate(vars_ext):
                res = pred_d[v] - true_d[v]
                row, col_pair = divmod(col_idx, ncols5)
                ax_res = axes5[row, col_pair * 2]
                ax_z   = axes5[row, col_pair * 2 + 1]
                mae     = np.full(len(snr_centers), np.nan)
                med_res = np.full(len(snr_centers), np.nan)
                q16_res = np.full(len(snr_centers), np.nan)
                q84_res = np.full(len(snr_centers), np.nan)
                med_z   = np.full(len(snr_centers), np.nan) if z_d else None
                q16_z   = np.full(len(snr_centers), np.nan) if z_d else None
                q84_z   = np.full(len(snr_centers), np.nan) if z_d else None
                for i, (lo, hi) in enumerate(zip(snr_bins[:-1], snr_bins[1:])):
                    mask = (snr_arr >= lo) & (snr_arr < hi)
                    if mask.sum() < 2:
                        continue
                    mae[i]     = np.mean(np.abs(res[mask]))
                    med_res[i] = np.median(res[mask])
                    q16_res[i] = np.percentile(res[mask], 16)
                    q84_res[i] = np.percentile(res[mask], 84)
                    if z_d:
                        med_z[i] = np.median(z_d[v][mask])
                        q16_z[i] = np.percentile(z_d[v][mask], 16)
                        q84_z[i] = np.percentile(z_d[v][mask], 84)
                valid = ~np.isnan(mae)
                sc = snr_centers[valid]
                ax_res.fill_between(sc, q16_res[valid], q84_res[valid], alpha=0.25, color='steelblue', label='68%')
                ax_res.plot(sc, med_res[valid], color='steelblue', lw=1.4, label='Median')
                ax_res.plot(sc, mae[valid],     color='tomato',    lw=1.2, ls='--', label='MAE')
                ax_res.axhline(0, color='gray', lw=0.8, ls=':')
                ax_res.set_xlabel(_label('snr')); ax_res.set_ylabel(rf'$\Delta$' + _label(v))
                ax_res.legend(frameon=False, fontsize=8, loc='best')
                _snr_xticks(ax_res, snr_a, snr_b)
                if z_d and med_z is not None:
                    ax_z.fill_between(sc, q16_z[valid], q84_z[valid], alpha=0.25, color='purple', label='68%')
                    ax_z.plot(sc, med_z[valid], color='purple', lw=1.4, label='Median z')
                    ax_z.axhline(0, color='gray', lw=0.8, ls=':')
                    ax_z.axhline( 1, color='gray', lw=0.6, ls='dotted')
                    ax_z.axhline(-1, color='gray', lw=0.6, ls='dotted')
                    ax_z.set_xlabel(_label('snr')); ax_z.set_ylabel(rf'z-score ' + _label(v))
                    ax_z.legend(frameon=False, fontsize=8, loc='best')
                    _snr_xticks(ax_z, snr_a, snr_b)
                else:
                    ax_z.set_visible(False)
            for ax in axes5.flat[n_ext * 2:]:
                ax.set_visible(False)
            fig5.tight_layout()
            figs.append(fig5); names.append('snr_binned_residuals')

            cmap = plt.cm.viridis
            cuts = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]

            # Figure 6: fraction within relative-error cutoff vs SNR
            fig6, axes6 = plt.subplots(nrows5, ncols5, figsize=(5*ncols5, 4*nrows5), squeeze=False)
            for ax, v in zip(axes6.flat, vars_ext):
                res = pred_d[v] - true_d[v]
                denom = np.where(np.abs(true_d[v]) > 1e-8, np.abs(true_d[v]), 1.0)
                rel_err = np.abs(res / denom)
                for j, cut in enumerate(cuts):
                    frac = np.full(len(snr_centers), np.nan)
                    for i, (lo, hi) in enumerate(zip(snr_bins[:-1], snr_bins[1:])):
                        mask = (snr_arr >= lo) & (snr_arr < hi)
                        if mask.sum() > 0:
                            frac[i] = 100.0 * (rel_err[mask] < cut).mean()
                    valid = ~np.isnan(frac)
                    ax.plot(snr_centers[valid], frac[valid], marker='o', ms=3,
                            color=cmap(j / (len(cuts) - 1)), label=str(cut))
                ax.set_xlabel(_label('snr')); ax.set_ylabel('% within cutoff')
                ax.set_title(_label(v))
                ax.legend(frameon=False, fontsize=7, title='|Δ/true|', loc='lower right')
                _snr_xticks(ax, snr_a, snr_b)
            for ax in axes6.flat[n_ext:]:
                ax.set_visible(False)
            fig6.tight_layout()
            figs.append(fig6); names.append('snr_fraction_within_cutoff')

            # Figure 7: violin of relative error vs SNR bins
            fig7, axes7 = plt.subplots(nrows5, ncols5, figsize=(5*ncols5, 4*nrows5), squeeze=False)
            for ax, v in zip(axes7.flat, vars_ext):
                res = pred_d[v] - true_d[v]
                denom = np.where(np.abs(true_d[v]) > 1e-8, np.abs(true_d[v]), 1.0)
                rel_err = res / denom
                groups = [rel_err[(snr_arr >= snr_bins[i]) & (snr_arr < snr_bins[i+1])]
                          for i in range(len(snr_bins)-1)]
                nonempty = [i for i, g in enumerate(groups) if len(g) > 1]
                if nonempty:
                    vp = ax.violinplot([groups[i] for i in nonempty],
                                       positions=snr_centers[nonempty], widths=0.8,
                                       showmedians=True, showextrema=False)
                    for body in vp['bodies']:
                        body.set_alpha(0.6)
                ax.axhline(0, color='gray', lw=0.8, ls=':')
                ax.set_xlabel(_label('snr')); ax.set_ylabel(r'$(\hat{x} - x)/|x|$')
                ax.set_title(_label(v))
                _snr_xticks(ax, snr_a, snr_b)
            for ax in axes7.flat[n_ext:]:
                ax.set_visible(False)
            fig7.tight_layout()
            figs.append(fig7); names.append('snr_violin')

        self._save_figs(figs, names)

    def _save_csv_param_est(self, y_true, y_pred, y_sigma, z_observed):
        vars_ = self.target_variables
        residual = y_pred - y_true
        cols = (
            [f'{v}_true'       for v in vars_] +
            [f'{v}_pred'       for v in vars_] +
            ([f'sigma_{v}_pred' for v in vars_] if y_sigma is not None else []) +
            [f'{v}_residual'   for v in vars_]
        )
        ordered = [y_true, y_pred]
        if y_sigma is not None:
            zscore = residual / (y_sigma + 1e-12)
            ordered.append(y_sigma)
            cols += [f'{v}_zscore' for v in vars_]
        ordered.append(residual)
        if y_sigma is not None:
            ordered.append(zscore)
        if z_observed is not None and self.observed_variables:
            cols += list(self.observed_variables)
            ordered.append(z_observed)
        pd.DataFrame(
            np.concatenate(ordered, axis=1),
            columns=cols,
        ).to_csv(self.save_dir / 'param_est_results.csv', index=False)

    # ═══════════════════════════════════════════════════════════════════════════
    # SKY LOCALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_sky(self, y_true, y_pred, z_observed):
        """y_pred can be (N,2) [dec,phi] or (N,3) unit vector — handles both."""
        dec_true = y_true[:, self.target_variables.index('dec')]
        phi_true = y_true[:, self.target_variables.index('phi')]

        # Convert y_pred to dec/phi regardless of representation
        if y_pred.shape[1] == 3:
            # Unit vector output from cosine-loss model
            norm = np.linalg.norm(y_pred, axis=1, keepdims=True)
            y_pred_norm = y_pred / (norm + 1e-8)
            dec_pred, phi_pred = _vec_to_angles(y_pred_norm)
            cos_theta_pred = y_pred_norm @ _H1L1_BASELINE
        else:
            # Direct (dec, phi) output
            dec_pred = y_pred[:, self.target_variables.index('dec')]
            phi_pred = y_pred[:, self.target_variables.index('phi')]
            v_pred = _angles_to_vec(dec_pred, phi_pred)
            cos_theta_pred = v_pred @ _H1L1_BASELINE

        v_true = _angles_to_vec(dec_true, phi_true)
        cos_theta_true = v_true @ _H1L1_BASELINE
        ring_dist = np.abs(cos_theta_pred - cos_theta_true)

        v_pred_for_angle = _angles_to_vec(dec_pred, phi_pred)
        cos_sim = np.clip((v_pred_for_angle * v_true).sum(axis=1), -1 + 1e-6, 1 - 1e-6)
        angular_error_deg = np.degrees(np.arccos(cos_sim))

        snr = self._get_snr(z_observed)

        self._save_csv_sky(dec_true, phi_true, dec_pred, phi_pred,
                           cos_theta_true, cos_theta_pred, ring_dist, angular_error_deg, snr)

        plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
        figs, names = [], []

        # Figure 1: true/pred histograms for dec and phi
        fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
        for ax, true, pred, key in [
            (axes1[0], dec_true, dec_pred, 'dec'),
            (axes1[1], phi_true, phi_pred, 'phi'),
        ]:
            bins = _bins_for(key, true)
            _hist_outline(ax, true, bins, color='red',   lw=1.2, label='True')
            _hist_outline(ax, pred, bins, color='green', lw=1.2, label='Predicted')
            ax.set_xlabel(_label(key)); ax.set_ylabel('Counts'); ax.legend(frameon=False)
        fig1.tight_layout()
        figs.append(fig1); names.append('sky_hist')

        # Figure 2: binned pred vs true for dec, phi, cos(theta)
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        for ax, true, pred, key in [
            (axes2[0], dec_true,       dec_pred,       'dec'),
            (axes2[1], phi_true,       phi_pred,       'phi'),
            (axes2[2], cos_theta_true, cos_theta_pred, 'cos_theta'),
        ]:
            bins = _bins_for(key, true)
            centers, med, q16, q84, _ = _binned_stats(true, pred, bins)
            valid = ~np.isnan(med)
            lims = [float(bins[0]), float(bins[-1])]
            ax.plot(lims, lims, color='gray', alpha=0.5, lw=1.2, ls=':')
            ax.fill_between(centers[valid], q16[valid], q84[valid],
                            alpha=0.25, color='steelblue', label='68%')
            ax.plot(centers[valid], med[valid], color='steelblue', lw=1.4, label='Median')
            ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
            ax.set_xlabel('True ' + _label(key)); ax.set_ylabel('Pred ' + _label(key))
            ax.legend(frameon=False, fontsize=9)
        fig2.tight_layout()
        figs.append(fig2); names.append('pred_vs_true')

        # Figure 3: angular error histogram
        bins_ang = np.linspace(0.0, 180.0, 37)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        _hist_outline(ax3, angular_error_deg, bins_ang, color='steelblue', lw=1.2)
        ax3.axvline(np.median(angular_error_deg), color='red', ls='--', lw=1.2,
                    label=f'Median = {np.median(angular_error_deg):.1f}°')
        ax3.set_xlabel('Angular error [deg]'); ax3.set_ylabel('Counts')
        ax3.legend(frameon=False)
        fig3.tight_layout()
        figs.append(fig3); names.append('angular_error_hist')

        if snr is not None:
            snr_bins = _snr_bins(snr)
            snr_a, snr_b = int(snr_bins[0]), int(snr_bins[-1])
            snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])

            # Figure 4: SNR-binned ring_distance, |Δdec|, |Δphi|
            fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
            for ax, vals, lbl in [
                (axes4[0], ring_dist,                    r'Ring dist $|\Delta\cos\theta|$'),
                (axes4[1], np.abs(dec_pred - dec_true),  r'$|\Delta\delta|\,[\mathrm{rad}]$'),
                (axes4[2], np.abs(phi_pred - phi_true),  r'$|\Delta\phi|\,[\mathrm{rad}]$'),
            ]:
                _, med, q16, q84, mae = _binned_stats(snr, vals, snr_bins)
                valid = ~np.isnan(med)
                ax.fill_between(snr_centers[valid], q16[valid], q84[valid],
                                alpha=0.25, color='steelblue', label='68%')
                ax.plot(snr_centers[valid], med[valid], color='steelblue', lw=1.4, label='Median')
                ax.plot(snr_centers[valid], mae[valid], color='tomato',    lw=1.2, ls='--', label='MAE')
                ax.axhline(0, color='gray', lw=0.8, ls=':')
                ax.set_xlabel('SNR'); ax.set_ylabel(lbl)
                ax.legend(frameon=False, fontsize=8, loc='best')
                _snr_xticks(ax, snr_a, snr_b)
            fig4.tight_layout()
            figs.append(fig4); names.append('snr_binned')

            cmap = plt.cm.viridis
            snr_cut_obj = pd.cut(snr, bins=snr_bins)
            bin_totals = pd.Series(np.ones(len(snr))).groupby(snr_cut_obj, observed=False).sum().values

            def _frac_within(values, cuts):
                out = np.zeros((len(snr_centers), len(cuts)))
                for j, cut in enumerate(cuts):
                    mask = np.abs(values) < cut
                    counts = pd.Series(mask.astype(int)).groupby(snr_cut_obj, observed=False).sum().values
                    with np.errstate(invalid='ignore'):
                        out[:, j] = np.where(bin_totals > 0, counts / bin_totals, np.nan)
                return out

            # Figure 5: fraction within angular error cutoff
            ang_cuts = [5, 10, 20, 30, 45, 60, 90]
            frac_ang = _frac_within(angular_error_deg, ang_cuts)
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            for j, cut in enumerate(ang_cuts):
                valid = ~np.isnan(frac_ang[:, j])
                ax5.plot(snr_centers[valid], frac_ang[valid, j] * 100, marker='o', ms=4,
                         color=cmap(j / (len(ang_cuts) - 1)), label=f'< {cut}°')
            ax5.set_xlabel('SNR'); ax5.set_ylabel('% within cutoff')
            ax5.set_title('Fraction within angular error cutoff vs SNR')
            ax5.legend(frameon=False, fontsize=9, title='Angular cutoff', loc='lower right')
            _snr_xticks(ax5, snr_a, snr_b)
            fig5.tight_layout()
            figs.append(fig5); names.append('snr_vs_angular_cutoff')

            # Figure 6: fraction within ring distance cutoff
            ring_cuts = [0.05, 0.1, 0.2, 0.3, 0.5]
            frac_ring = _frac_within(ring_dist, ring_cuts)
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            for j, cut in enumerate(ring_cuts):
                valid = ~np.isnan(frac_ring[:, j])
                ax6.plot(snr_centers[valid], frac_ring[valid, j] * 100, marker='o', ms=4,
                         color=cmap(j / (len(ring_cuts) - 1)), label=f'< {cut}')
            ax6.set_xlabel('SNR'); ax6.set_ylabel('% within cutoff')
            ax6.set_title('Fraction within ring distance cutoff vs SNR')
            ax6.legend(frameon=False, fontsize=9, title='Ring dist cutoff', loc='lower right')
            _snr_xticks(ax6, snr_a, snr_b)
            fig6.tight_layout()
            figs.append(fig6); names.append('snr_vs_ring_cutoff')

            # Figure 7: violin of angular error and ring distance vs SNR bins
            bin_groups_ang  = [angular_error_deg[(snr >= snr_bins[i]) & (snr < snr_bins[i+1])]
                               for i in range(len(snr_bins)-1)]
            bin_groups_ring = [ring_dist[(snr >= snr_bins[i]) & (snr < snr_bins[i+1])]
                               for i in range(len(snr_bins)-1)]
            nonempty = [i for i, g in enumerate(bin_groups_ang) if len(g) > 1]
            fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))
            for ax, groups, ylabel in [
                (ax7a, bin_groups_ang,  'Angular error [°]'),
                (ax7b, bin_groups_ring, 'Ring distance'),
            ]:
                if nonempty:
                    vp = ax.violinplot([groups[i] for i in nonempty],
                                       positions=snr_centers[nonempty], widths=0.8,
                                       showmedians=True, showextrema=False)
                    for body in vp['bodies']:
                        body.set_alpha(0.6)
                ax.set_xlabel('SNR'); ax.set_ylabel(ylabel)
                _snr_xticks(ax, snr_a, snr_b)
            fig7.tight_layout()
            figs.append(fig7); names.append('snr_violin')

        self._save_figs(figs, names)

    def _save_csv_sky(self, dec_true, phi_true, dec_pred, phi_pred,
                      cos_theta_true, cos_theta_pred, ring_dist, angular_error_deg, snr):
        data = {
            'dec_true':           dec_true,
            'phi_true':           phi_true,
            'dec_pred':           dec_pred,
            'phi_pred':           phi_pred,
            'dec_residual':       dec_pred - dec_true,
            'phi_residual':       phi_pred - phi_true,
            'cos_theta_true':     cos_theta_true,
            'cos_theta_pred':     cos_theta_pred,
            'ring_distance':      ring_dist,
            'angular_error_deg':  angular_error_deg,
        }
        if snr is not None:
            data['snr'] = snr
        pd.DataFrame(data).to_csv(self.save_dir / 'sky_loc_results.csv', index=False)
        print(f'Saved {len(dec_true)} rows → {self.save_dir / "sky_loc_results.csv"}')

    # ── Save helper ───────────────────────────────────────────────────────────

    def _save_figs(self, figs, names):
        for fig, name in zip(figs, names):
            out = self.save_dir / f'{name}.png'
            fig.savefig(out, dpi=400, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out}')
