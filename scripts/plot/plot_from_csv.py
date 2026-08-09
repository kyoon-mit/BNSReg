"""Generate eval plots from an existing param_est_results.csv or sky_loc_results.csv.

Usage:
    python scripts/plot/plot_from_csv.py /path/to/plots/param_est_results.csv
    python scripts/plot/plot_from_csv.py /path/to/plots/     # scans dir for CSV

Auto-detects mode:
  - param_est:   columns ending _true / _pred / sigma_*_pred / *_zscore / snr
  - sky_loc:     columns dec_true, phi_true, dec_pred, phi_pred, ..., snr (optional)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Constants and helpers from plot_param_est.py ──────────────────────────────
BINS = {
    'chirp_mass':  np.linspace(0.5,   3.0,  51),
    'mass_ratio':  np.linspace(0.0,   1.0,  51),
    'mass_1':      np.linspace(1.0,   2.5,  31),
    'mass_2':      np.linspace(1.0,   2.5,  31),
    'm1':          np.linspace(1.0,   2.5,  31),
    'm2':          np.linspace(1.0,   2.5,  31),
    'distance':    np.linspace(100.,  1000., 37),
    'a_1':         np.linspace(0.0,   1.0,  21),
    'a_2':         np.linspace(0.0,   1.0,  21),
    'tilt_1':      np.linspace(0.0,   3.2,  33),
    'tilt_2':      np.linspace(0.0,   3.2,  33),
    'psi':         np.linspace(0.0,   3.2,  33),
    'inclination': np.linspace(0.0,   3.2,  33),
    'phi_12':      np.linspace(0.0,   6.4,  33),
    'phi_jl':      np.linspace(0.0,   6.4,  33),
    'phic':        np.linspace(0.0,   6.4,  33),
    'phi':         np.linspace(-3.2,  3.2,  65),
    'dec':         np.linspace(-1.6,  1.6,  33),
    'cos_theta':   np.linspace(-1.0,  1.0,  41),
    'snr':         np.linspace(5.0,  50.0,  19),
    's1z':         np.linspace(-1.0,  1.0,  41),
    's2z':         np.linspace(-1.0,  1.0,  41),
    'chi1':        np.linspace(-1.0,  1.0,  41),
    'chi2':        np.linspace(-1.0,  1.0,  41),
}

VAR_LABELS = {
    'chirp_mass':  r'$\mathcal{M}_c\,[M_\odot]$',
    'mass_ratio':  r'$q$',
    'm1':          r'$m_1\,[M_\odot]$',
    'm2':          r'$m_2\,[M_\odot]$',
    'distance':    r'$d_L\,[\mathrm{Mpc}]$',
    'phi':         r'$\phi\,[\mathrm{rad}]$',
    'dec':         r'$\delta\,[\mathrm{rad}]$',
    'cos_theta':   r'$\cos\theta$',
    'snr':         r'SNR',
}

_H1L1_BASELINE = np.array([0.6953014731407166, -0.5535351634025574, -0.4584263563156128])


def _label(k):
    return VAR_LABELS.get(k, k)


def _bins_for(k, data):
    if k in BINS:
        bins = BINS[k]
        # Fall back to auto-range when data lies outside predefined bins
        # (e.g. normalized SNR in [0,1] instead of raw 10-50).
        if float(data.max()) < bins[0] or float(data.min()) > bins[-1]:
            lo, hi = float(data.min()), float(data.max())
            return np.linspace(lo, hi, len(bins))
        return bins
    lo, hi = float(data.min()), float(data.max())
    return np.linspace(lo, hi, 51)


def _snr_bins(snr_arr):
    a = int(np.floor(float(snr_arr.min())))
    b = int(np.ceil(float(snr_arr.max())))
    return np.arange(a, b + 1, dtype=float)


def _snr_xticks(ax, a, b):
    step = max(2, (b - a) // 5)
    ticks = sorted(set(range(a, b + 1, step)) | {a, b})
    ax.set_xticks(ticks)
    ax.set_xlim(a - 0.5, b + 0.5)


def _hist_outline(ax, data, bins, **kw):
    counts, edges = np.histogram(data, bins=bins)
    ax.stairs(counts, edges, **kw)


def _binned_stats(x, y, bins):
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


def _mc_q_to_m1_m2(mc, q):
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


def _angles_to_vec(dec, phi):
    return np.stack([np.cos(dec)*np.cos(phi), np.cos(dec)*np.sin(phi), np.sin(dec)], axis=-1)


def _save_figs(figs, names, save_dir):
    for fig, name in zip(figs, names):
        out = save_dir / f'{name}.png'
        fig.savefig(out, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {out}')


# ── Param-est plotting ─────────────────────────────────────────────────────────

def plot_param_est(df, save_dir):
    vars_ = [c.removesuffix('_true') for c in df.columns if c.endswith('_true')]
    true_d  = {v: df[f'{v}_true'].values  for v in vars_}
    pred_d  = {v: df[f'{v}_pred'].values  for v in vars_}
    has_sigma = f'sigma_{vars_[0]}_pred' in df.columns
    sigma_d = {v: df[f'sigma_{v}_pred'].values for v in vars_} if has_sigma else {}

    has_snr = 'snr' in df.columns
    snr_arr = df['snr'].values if has_snr else None

    vars_ext = list(vars_)
    if 'chirp_mass' in vars_ and 'mass_ratio' in vars_:
        mc_t, q_t = true_d['chirp_mass'], true_d['mass_ratio']
        mc_p, q_p = pred_d['chirp_mass'], pred_d['mass_ratio']
        m1t, m2t = _mc_q_to_m1_m2(mc_t, q_t)
        m1p, m2p = _mc_q_to_m1_m2(mc_p, q_p)
        for key, tr, pr in [('m1', m1t, m1p), ('m2', m2t, m2p)]:
            true_d[key] = tr; pred_d[key] = pr
        if sigma_d:
            s_mc, s_q = sigma_d['chirp_mass'], sigma_d['mass_ratio']
            s_m1, s_m2 = _propagate_m1_m2_uncertainty(mc_p, q_p, s_mc, s_q)
            sigma_d['m1'] = s_m1; sigma_d['m2'] = s_m2
        vars_ext = vars_ + ['m1', 'm2']

    residual_d = {v: pred_d[v] - true_d[v] for v in vars_ext}
    z_d = {v: residual_d[v] / (sigma_d[v] + 1e-12) for v in vars_ext} if sigma_d else {}

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    figs, names = [], []

    n1 = len(vars_); ncols1 = min(n1, 4); nrows1 = math.ceil(n1 / ncols1)
    fig1, axes1 = plt.subplots(nrows1, ncols1, figsize=(5*ncols1, 4*nrows1), squeeze=False)
    for ax, v in zip(axes1.flat, vars_):
        bins = _bins_for(v, true_d[v])
        _hist_outline(ax, true_d[v], bins, color='red',   lw=1.2, label='True')
        _hist_outline(ax, pred_d[v], bins, color='green', lw=1.2, label='Predicted')
        ax.set_xlabel(_label(v)); ax.set_ylabel('Counts'); ax.legend(frameon=False)
    for ax in axes1.flat[n1:]: ax.set_visible(False)
    fig1.tight_layout(); figs.append(fig1); names.append('param_hist')

    n234 = len(vars_ext); ncols2 = min(n234, 4); nrows2 = math.ceil(n234 / ncols2)
    fig2, axes2 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4.5*nrows2), squeeze=False)
    for ax, v in zip(axes2.flat, vars_ext):
        edges = _bins_for(v, true_d[v])
        centers = 0.5 * (edges[:-1] + edges[1:])
        tr, pr = true_d[v], pred_d[v]
        mean_p = np.full(len(centers), np.nan); median_p = mean_p.copy()
        q16 = mean_p.copy(); q84 = mean_p.copy(); q2p5 = mean_p.copy(); q97p5 = mean_p.copy()
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (tr >= lo) & (tr < hi)
            if mask.sum() < 2: continue
            p_bin = pr[mask]
            mean_p[i]=np.mean(p_bin); median_p[i]=np.median(p_bin)
            q16[i]=np.percentile(p_bin,16); q84[i]=np.percentile(p_bin,84)
            q2p5[i]=np.percentile(p_bin,2.5); q97p5[i]=np.percentile(p_bin,97.5)
        data_lo=float(np.nanmin(tr)); data_hi=float(np.nanmax(tr))
        mask_e = (edges>=data_lo)&(edges<=data_hi)
        valid = np.where(mask_e[:-1]&mask_e[1:])[0]; sc = centers[valid]; lims=[data_lo,data_hi]
        ax.plot(lims,lims,color='gray',alpha=0.5,lw=1.2,ls=':')
        ax.step(sc,q2p5[valid], where='mid',color='steelblue',lw=0.8,ls='dotted',alpha=0.7,label='95%')
        ax.step(sc,q97p5[valid],where='mid',color='steelblue',lw=0.8,ls='dotted',alpha=0.7)
        ax.step(sc,q16[valid],  where='mid',color='steelblue',lw=0.8,ls='dashed',alpha=0.9,label='68%')
        ax.step(sc,q84[valid],  where='mid',color='steelblue',lw=0.8,ls='dashed',alpha=0.9)
        ax.step(sc,median_p[valid],where='mid',color='steelblue',lw=1.4,label='Median')
        ax.step(sc,mean_p[valid],  where='mid',color='purple',lw=1.2,alpha=0.8,label='Mean')
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
        ax.set_xlabel('True '+_label(v)); ax.set_ylabel('Pred '+_label(v))
        ax.legend(frameon=False, fontsize=9, loc='upper left')
    for ax in axes2.flat[n234:]: ax.set_visible(False)
    fig2.tight_layout(); figs.append(fig2); names.append('pred_vs_true')

    if sigma_d:
        fig3, axes3 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4*nrows2), squeeze=False)
        for ax, v in zip(axes3.flat, vars_ext):
            _hist_outline(ax, sigma_d[v], bins=20, color='black', lw=1.2)
            ax.set_xlabel(rf'$\sigma_{{{_label(v).strip("$")}}}$'); ax.set_ylabel('Counts')
        for ax in axes3.flat[n234:]: ax.set_visible(False)
        fig3.tight_layout(); figs.append(fig3); names.append('sigma_hist')

        fig4, axes4 = plt.subplots(nrows2, ncols2, figsize=(5*ncols2, 4*nrows2), squeeze=False)
        for ax, v in zip(axes4.flat, vars_ext):
            _hist_outline(ax, z_d[v], bins=20, color='steelblue', lw=1.2)
            ax.axvline(0, color='gray', ls='--', lw=1.0)
            ax.axvline(1, color='gray', ls='dotted', lw=1.0)
            ax.axvline(-1, color='gray', ls='dotted', lw=1.0)
            inner = _label(v).strip('$')
            ax.set_xlabel(rf'$(x_{{\mathrm{{pred}}}}-x_{{\mathrm{{true}}}})/\sigma_{{{inner}}}$')
            ax.set_ylabel('Counts')
        for ax in axes4.flat[n234:]: ax.set_visible(False)
        fig4.tight_layout(); figs.append(fig4); names.append('z_score_hist')

    if snr_arr is not None:
        snr_bins = _snr_bins(snr_arr)
        snr_a, snr_b = int(snr_bins[0]), int(snr_bins[-1])
        snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
        ncols5 = min(n234, 4); nrows5 = math.ceil(n234 / ncols5)

        fig5, axes5 = plt.subplots(nrows5, ncols5*2, figsize=(5*ncols5*2, 4*nrows5), squeeze=False)
        for col_idx, v in enumerate(vars_ext):
            res = residual_d[v]; row, cpair = divmod(col_idx, ncols5)
            ax_res = axes5[row, cpair*2]; ax_z = axes5[row, cpair*2+1]
            mae=np.full(len(snr_centers),np.nan); med_res=mae.copy()
            q16_res=mae.copy(); q84_res=mae.copy()
            for i,(lo,hi) in enumerate(zip(snr_bins[:-1],snr_bins[1:])):
                mask=(snr_arr>=lo)&(snr_arr<hi)
                if mask.sum()<2: continue
                mae[i]=np.mean(np.abs(res[mask])); med_res[i]=np.median(res[mask])
                q16_res[i]=np.percentile(res[mask],16); q84_res[i]=np.percentile(res[mask],84)
            valid = ~np.isnan(mae); sc=snr_centers[valid]
            ax_res.fill_between(sc,q16_res[valid],q84_res[valid],alpha=0.25,color='steelblue',label='68%')
            ax_res.plot(sc,med_res[valid],color='steelblue',lw=1.4,label='Median')
            ax_res.plot(sc,mae[valid],color='tomato',lw=1.2,ls='--',label='MAE')
            ax_res.axhline(0,color='gray',lw=0.8,ls=':')
            ax_res.set_xlabel(_label('snr')); ax_res.set_ylabel(rf'$\Delta$'+_label(v))
            ax_res.legend(frameon=False,fontsize=8,loc='best')
            _snr_xticks(ax_res, snr_a, snr_b)
            if z_d:
                med_z=np.full(len(snr_centers),np.nan); q16_z=med_z.copy(); q84_z=med_z.copy()
                for i,(lo,hi) in enumerate(zip(snr_bins[:-1],snr_bins[1:])):
                    mask=(snr_arr>=lo)&(snr_arr<hi)
                    if mask.sum()<2: continue
                    med_z[i]=np.median(z_d[v][mask])
                    q16_z[i]=np.percentile(z_d[v][mask],16); q84_z[i]=np.percentile(z_d[v][mask],84)
                ax_z.fill_between(sc,q16_z[valid],q84_z[valid],alpha=0.25,color='purple',label='68%')
                ax_z.plot(sc,med_z[valid],color='purple',lw=1.4,label='Median z')
                ax_z.axhline(0,color='gray',lw=0.8,ls=':')
                ax_z.axhline(1,color='gray',lw=0.6,ls='dotted'); ax_z.axhline(-1,color='gray',lw=0.6,ls='dotted')
                ax_z.set_xlabel(_label('snr')); ax_z.set_ylabel(rf'z-score '+_label(v))
                ax_z.legend(frameon=False,fontsize=8,loc='best')
                _snr_xticks(ax_z, snr_a, snr_b)
            else:
                ax_z.set_visible(False)
        for ax in axes5.flat[n234*2:]: ax.set_visible(False)
        fig5.tight_layout(); figs.append(fig5); names.append('snr_binned_residuals')

        cmap = plt.cm.viridis
        cuts = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
        fig6, axes6 = plt.subplots(nrows5, ncols5, figsize=(5*ncols5, 4*nrows5), squeeze=False)
        for ax, v in zip(axes6.flat, vars_ext):
            denom = np.where(np.abs(true_d[v])>1e-8, np.abs(true_d[v]), 1.0)
            rel_err = np.abs(residual_d[v] / denom)
            for j, cut in enumerate(cuts):
                frac = np.full(len(snr_centers), np.nan)
                for i,(lo,hi) in enumerate(zip(snr_bins[:-1],snr_bins[1:])):
                    mask=(snr_arr>=lo)&(snr_arr<hi)
                    if mask.sum()>0: frac[i]=100.0*(rel_err[mask]<cut).mean()
                valid = ~np.isnan(frac)
                ax.plot(snr_centers[valid],frac[valid],marker='o',ms=3,
                        color=cmap(j/(len(cuts)-1)),label=str(cut))
            ax.set_xlabel(_label('snr')); ax.set_ylabel('% within cutoff')
            ax.set_title(_label(v)); ax.legend(frameon=False,fontsize=7,title='|Δ/true|',loc='lower right')
            _snr_xticks(ax, snr_a, snr_b)
        for ax in axes6.flat[n234:]: ax.set_visible(False)
        fig6.tight_layout(); figs.append(fig6); names.append('snr_fraction_within_cutoff')

        fig7, axes7 = plt.subplots(nrows5, ncols5, figsize=(5*ncols5, 4*nrows5), squeeze=False)
        for ax, v in zip(axes7.flat, vars_ext):
            denom = np.where(np.abs(true_d[v])>1e-8, np.abs(true_d[v]), 1.0)
            rel_err = residual_d[v] / denom
            groups = [rel_err[(snr_arr>=snr_bins[i])&(snr_arr<snr_bins[i+1])] for i in range(len(snr_bins)-1)]
            nonempty = [i for i,g in enumerate(groups) if len(g)>1]
            if nonempty:
                vp = ax.violinplot([groups[i] for i in nonempty], positions=snr_centers[nonempty],
                                   widths=0.8, showmedians=True, showextrema=False)
                for body in vp['bodies']: body.set_alpha(0.6)
            ax.axhline(0,color='gray',lw=0.8,ls=':')
            ax.set_xlabel(_label('snr')); ax.set_ylabel(r'$(\hat{x}-x)/|x|$'); ax.set_title(_label(v))
            _snr_xticks(ax, snr_a, snr_b)
        for ax in axes7.flat[n234:]: ax.set_visible(False)
        fig7.tight_layout(); figs.append(fig7); names.append('snr_violin')

    _save_figs(figs, names, save_dir)


# ── Sky-loc plotting ───────────────────────────────────────────────────────────

def plot_sky_loc(df, save_dir):
    dec_true = df['dec_true'].values
    phi_true = df['phi_true'].values
    dec_pred = df['dec_pred'].values
    phi_pred = df['phi_pred'].values

    has_snr = 'snr' in df.columns
    snr_arr = df['snr'].values if has_snr else None

    if 'ring_distance' in df.columns:
        ring_dist = df['ring_distance'].values
        angular_error_deg = df['angular_error_deg'].values
        cos_theta_true = df['cos_theta_true'].values
        cos_theta_pred = df['cos_theta_pred'].values
    else:
        v_true = _angles_to_vec(dec_true, phi_true)
        v_pred = _angles_to_vec(dec_pred, phi_pred)
        cos_theta_true = v_true @ _H1L1_BASELINE
        cos_theta_pred = v_pred @ _H1L1_BASELINE
        ring_dist = np.abs(cos_theta_pred - cos_theta_true)
        cos_sim = np.clip((v_pred * v_true).sum(axis=1), -1+1e-6, 1-1e-6)
        angular_error_deg = np.degrees(np.arccos(cos_sim))

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    figs, names = [], []

    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
    for ax, true, pred, key in [(axes1[0],dec_true,dec_pred,'dec'),(axes1[1],phi_true,phi_pred,'phi')]:
        bins = _bins_for(key, true)
        _hist_outline(ax, true, bins, color='red',   lw=1.2, label='True')
        _hist_outline(ax, pred, bins, color='green', lw=1.2, label='Predicted')
        ax.set_xlabel(_label(key)); ax.set_ylabel('Counts'); ax.legend(frameon=False)
    fig1.tight_layout(); figs.append(fig1); names.append('sky_hist')

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, true, pred, key in [
        (axes2[0], dec_true, dec_pred, 'dec'),
        (axes2[1], phi_true, phi_pred, 'phi'),
        (axes2[2], cos_theta_true, cos_theta_pred, 'cos_theta'),
    ]:
        bins = _bins_for(key, true)
        centers, med, q16, q84, _ = _binned_stats(true, pred, bins)
        valid = ~np.isnan(med); lims=[float(bins[0]),float(bins[-1])]
        ax.plot(lims,lims,color='gray',alpha=0.5,lw=1.2,ls=':')
        ax.fill_between(centers[valid],q16[valid],q84[valid],alpha=0.25,color='steelblue',label='68%')
        ax.plot(centers[valid],med[valid],color='steelblue',lw=1.4,label='Median')
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
        ax.set_xlabel('True '+_label(key)); ax.set_ylabel('Pred '+_label(key))
        ax.legend(frameon=False,fontsize=9)
    fig2.tight_layout(); figs.append(fig2); names.append('pred_vs_true')

    bins_ang = np.linspace(0.0, 180.0, 37)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    _hist_outline(ax3, angular_error_deg, bins_ang, color='steelblue', lw=1.2)
    ax3.axvline(np.median(angular_error_deg), color='red', ls='--', lw=1.2,
                label=f'Median = {np.median(angular_error_deg):.1f}°')
    ax3.set_xlabel('Angular error [deg]'); ax3.set_ylabel('Counts'); ax3.legend(frameon=False)
    fig3.tight_layout(); figs.append(fig3); names.append('angular_error_hist')

    if snr_arr is not None:
        snr_bins = _snr_bins(snr_arr)
        snr_a, snr_b = int(snr_bins[0]), int(snr_bins[-1])
        snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])

        fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
        for ax, vals, lbl in [
            (axes4[0], ring_dist,                   r'Ring dist $|\Delta\cos\theta|$'),
            (axes4[1], np.abs(dec_pred-dec_true),   r'$|\Delta\delta|\,[\mathrm{rad}]$'),
            (axes4[2], np.abs(phi_pred-phi_true),   r'$|\Delta\phi|\,[\mathrm{rad}]$'),
        ]:
            _, med, q16, q84, mae = _binned_stats(snr_arr, vals, snr_bins)
            valid = ~np.isnan(med)
            ax.fill_between(snr_centers[valid],q16[valid],q84[valid],alpha=0.25,color='steelblue',label='68%')
            ax.plot(snr_centers[valid],med[valid],color='steelblue',lw=1.4,label='Median')
            ax.plot(snr_centers[valid],mae[valid],color='tomato',lw=1.2,ls='--',label='MAE')
            ax.axhline(0,color='gray',lw=0.8,ls=':'); ax.set_xlabel('SNR'); ax.set_ylabel(lbl)
            ax.legend(frameon=False,fontsize=8,loc='best')
            _snr_xticks(ax, snr_a, snr_b)
        fig4.tight_layout(); figs.append(fig4); names.append('snr_binned')

        cmap = plt.cm.viridis
        snr_cut_obj = pd.cut(snr_arr, bins=snr_bins)
        bin_totals = pd.Series(np.ones(len(snr_arr))).groupby(snr_cut_obj, observed=True).sum().values

        def _frac_within(values, cuts):
            out = np.zeros((len(snr_centers), len(cuts)))
            for j, cut in enumerate(cuts):
                mask = np.abs(values) < cut
                counts = pd.Series(mask.astype(int)).groupby(snr_cut_obj, observed=True).sum().values
                with np.errstate(invalid='ignore'):
                    out[:,j] = np.where(bin_totals>0, counts/bin_totals, np.nan)
            return out

        ang_cuts = [5, 10, 20, 30, 45, 60, 90]
        frac_ang = _frac_within(angular_error_deg, ang_cuts)
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        for j, cut in enumerate(ang_cuts):
            valid = ~np.isnan(frac_ang[:,j])
            ax5.plot(snr_centers[valid],frac_ang[valid,j]*100,marker='o',ms=4,
                     color=cmap(j/(len(ang_cuts)-1)),label=f'< {cut}°')
        ax5.set_xlabel('SNR'); ax5.set_ylabel('% within cutoff')
        ax5.set_title('Fraction within angular error cutoff vs SNR')
        ax5.legend(frameon=False,fontsize=9,title='Angular cutoff',loc='lower right')
        _snr_xticks(ax5, snr_a, snr_b)
        fig5.tight_layout(); figs.append(fig5); names.append('snr_vs_angular_cutoff')

        ring_cuts = [0.05, 0.1, 0.2, 0.3, 0.5]
        frac_ring = _frac_within(ring_dist, ring_cuts)
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        for j, cut in enumerate(ring_cuts):
            valid = ~np.isnan(frac_ring[:,j])
            ax6.plot(snr_centers[valid],frac_ring[valid,j]*100,marker='o',ms=4,
                     color=cmap(j/(len(ring_cuts)-1)),label=f'< {cut}')
        ax6.set_xlabel('SNR'); ax6.set_ylabel('% within cutoff')
        ax6.set_title('Fraction within ring distance cutoff vs SNR')
        ax6.legend(frameon=False,fontsize=9,title='Ring dist cutoff',loc='lower right')
        _snr_xticks(ax6, snr_a, snr_b)
        fig6.tight_layout(); figs.append(fig6); names.append('snr_vs_ring_cutoff')

        bin_groups_ang  = [angular_error_deg[(snr_arr>=snr_bins[i])&(snr_arr<snr_bins[i+1])] for i in range(len(snr_bins)-1)]
        bin_groups_ring = [ring_dist[(snr_arr>=snr_bins[i])&(snr_arr<snr_bins[i+1])]         for i in range(len(snr_bins)-1)]
        nonempty = [i for i,g in enumerate(bin_groups_ang) if len(g)>1]
        fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))
        for ax, groups, ylabel in [(ax7a,bin_groups_ang,'Angular error [°]'),(ax7b,bin_groups_ring,'Ring distance')]:
            if nonempty:
                vp = ax.violinplot([groups[i] for i in nonempty], positions=snr_centers[nonempty],
                                   widths=0.8, showmedians=True, showextrema=False)
                for body in vp['bodies']: body.set_alpha(0.6)
            ax.set_xlabel('SNR'); ax.set_ylabel(ylabel)
            _snr_xticks(ax, snr_a, snr_b)
        fig7.tight_layout(); figs.append(fig7); names.append('snr_violin')

    _save_figs(figs, names, save_dir)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Re-generate eval plots from saved CSV.')
    parser.add_argument('csv_path', help='Path to param_est_results.csv or sky_loc_results.csv, '
                        'or a directory containing one of them.')
    parser.add_argument('--out', default='', help='Output directory (default: same dir as CSV)')
    args = parser.parse_args()

    p = Path(args.csv_path)
    if p.is_dir():
        for name in ('param_est_results.csv', 'sky_loc_results.csv'):
            if (p / name).exists():
                p = p / name
                break
        else:
            print(f'ERROR: no param_est_results.csv or sky_loc_results.csv in {p}', file=sys.stderr)
            sys.exit(1)

    print(f'Reading {p}')
    df = pd.read_csv(p)
    print(f'  {len(df)} rows, columns: {list(df.columns)}')

    save_dir = Path(args.out) if args.out else p.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    is_sky = 'dec_true' in df.columns and 'phi_true' in df.columns
    if is_sky:
        print('Detected sky localization CSV')
        plot_sky_loc(df, save_dir)
    else:
        print('Detected param-est CSV')
        plot_param_est(df, save_dir)

    print('Done.')


if __name__ == '__main__':
    main()
