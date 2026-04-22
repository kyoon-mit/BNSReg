from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── predefined bins ───────────────────────────────────────────────────────────

BINS = {
    'chirp_mass': np.linspace(0.5, 3.0, 51),
    'mass_ratio':  np.linspace(0.0, 1.0, 51),
    'mass1':       np.linspace(0.5, 3.0, 51),
    'mass2':       np.linspace(0.5, 3.0, 51),
}

MIN_VAL = {
    'chirp_mass': 0.85-0.05,
    'mass_ratio': 0.4,
    'mass1':      1.0,
    'mass2':      1.0,
}

MAX_VAL = {
    'chirp_mass': 2.2+0.05,
    'mass_ratio': 1.0,
    'mass1':      2.5,
    'mass2':      2.5,
}

# ── LaTeX labels ──────────────────────────────────────────────────────────────

LABEL = {
    'chirp_mass': r'$\mathcal{M}_{c}$',
    'mass_ratio': r'$q$',
    'mass1':      r'$m_1$',
    'mass2':      r'$m_2$',
}

SIG_LABEL = {
    'chirp_mass': r'$\sigma_{\mathcal{M}_{c}}$',
    'mass_ratio': r'$\sigma_{q}$',
    'mass1':      r'$\sigma_{m_1}$',
    'mass2':      r'$\sigma_{m_2}$',
}


# ── helpers ───────────────────────────────────────────────────────────────────

def mc_q_to_m1_m2(mc, q):
    """
    Convert chirp mass Mc and mass ratio q = m2/m1 (q <= 1) to m1, m2.
        Mc = (m1*m2)^(3/5) / (m1+m2)^(1/5)
        q  = m2 / m1   =>   m2 = q*m1
    Solving: m1 = Mc * (1+q)^(1/5) / q^(3/5)
    """
    m1 = mc * (1.0 + q) ** 0.2 / q ** 0.6
    m2 = q * m1
    return m1, m2


def propagate_m1_m2_uncertainty(mc, q, sigma_mc, sigma_q):
    """
    First-order error propagation for m1(Mc,q) and m2(Mc,q).

    dm1/dMc = (1+q)^(1/5) / q^(3/5)                    = m1/Mc
    dm1/dq  = Mc * [ (1/5)(1+q)^(-4/5)/q^(3/5)
                    - (3/5)(1+q)^(1/5)/q^(8/5) ]
            = m1 * [ 1/(5*(1+q)) - 3/(5*q) ]

    dm2/dMc = q * dm1/dMc = m2/Mc
    dm2/dq  = m1 + q*dm1/dq
    """
    m1, m2 = mc_q_to_m1_m2(mc, q)

    dm1_dmc = m1 / mc
    dm1_dq  = m1 * (1.0 / (5.0 * (1.0 + q)) - 3.0 / (5.0 * q))

    dm2_dmc = m2 / mc
    dm2_dq  = m1 + q * dm1_dq

    sigma_m1 = np.sqrt((dm1_dmc * sigma_mc) ** 2 + (dm1_dq * sigma_q) ** 2)
    sigma_m2 = np.sqrt((dm2_dmc * sigma_mc) ** 2 + (dm2_dq * sigma_q) ** 2)

    return sigma_m1, sigma_m2


def hist_outline(ax, data, bins, **kwargs):
    """Draw a histogram as a step outline (no fill)."""
    counts, edges = np.histogram(data, bins=bins)
    ax.stairs(counts, edges, **kwargs)



# ── main function ─────────────────────────────────────────────────────────────

def plot_gw_results(csv_path: Path, save_path: Path | None = None):
    """
    Read a CSV with columns:
        M_chirp_true, M_ratio_true,
        M_chirp_pred, M_ratio_pred,
        sigma_M_chirp_pred, sigma_M_ratio_pred

    and produce four figures:
        1. Overlaid true/pred histograms for Mc and q
        2. Pred vs True scatter (colored by sigma) for Mc, q, m1, m2
        3. Uncertainty histograms for Mc, q, m1, m2
        4. Z-score histograms for Mc, q, m1, m2
    """
    df = pd.read_csv(csv_path)

    # ── unpack columns ────────────────────────────────────────────────────────
    mc_true = df['M_chirp_true'].values
    q_true  = df['M_ratio_true'].values
    mc_pred = df['M_chirp_pred'].values
    q_pred  = df['M_ratio_pred'].values
    sig_mc  = df['sigma_M_chirp_pred'].values
    sig_q   = df['sigma_M_ratio_pred'].values

    # ── derived quantities ────────────────────────────────────────────────────
    m1_true, m2_true         = mc_q_to_m1_m2(mc_true, q_true)
    m1_pred, m2_pred         = mc_q_to_m1_m2(mc_pred, q_pred)
    sig_m1_pred, sig_m2_pred = propagate_m1_m2_uncertainty(mc_pred, q_pred, sig_mc, sig_q)

    z_mc = (mc_pred - mc_true) / sig_mc
    z_q  = (q_pred  - q_true)  / sig_q
    z_m1 = (m1_pred - m1_true) / sig_m1_pred
    z_m2 = (m2_pred - m2_true) / sig_m2_pred

    # ── shared style ──────────────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 1 – True vs Predicted histograms for Mc and q
    # ═════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, true, pred, key in zip(
        axes1,
        [mc_true, q_true],
        [mc_pred, q_pred],
        ['chirp_mass', 'mass_ratio'],
    ):
        hist_outline(ax, true, bins=BINS[key], color='red',   linewidth=1.2, label='True')
        hist_outline(ax, pred, bins=BINS[key], color='green', linewidth=1.2, label='Predicted')
        ax.set_xlabel(LABEL[key])
        ax.set_ylabel('Counts')
        ax.legend(frameon=False)

    fig1.tight_layout()

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 2 – Binned pred vs true: mean, median, 68%, 95% quantiles
    # ═════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

    entries = [
        (mc_true, mc_pred, 'chirp_mass'),
        (q_true,  q_pred,  'mass_ratio'),
        (m1_true, m1_pred, 'mass1'),
        (m2_true, m2_pred, 'mass2'),
    ]

    for ax, (true, pred, key) in zip(axes2.flat, entries):
        edges = BINS[key]
        centers = 0.5 * (edges[:-1] + edges[1:])

        # per-bin statistics
        mean_p   = np.full(len(centers), np.nan)
        median_p = np.full(len(centers), np.nan)
        q16      = np.full(len(centers), np.nan)  # 68%: [16, 84]
        q84      = np.full(len(centers), np.nan)
        q2p5     = np.full(len(centers), np.nan)  # 95%: [2.5, 97.5]
        q97p5    = np.full(len(centers), np.nan)

        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (true >= lo) & (true < hi)
            if mask.sum() < 2:
                continue
            p_bin        = pred[mask]
            mean_p[i]    = np.mean(p_bin)
            median_p[i]  = np.median(p_bin)
            q16[i]       = np.percentile(p_bin, 16)
            q84[i]       = np.percentile(p_bin, 84)
            q2p5[i]      = np.percentile(p_bin,  2.5)
            q97p5[i]     = np.percentile(p_bin, 97.5)

        # stair-step plotting helper
        def stairs_pair(ax, x, y, **kw):
            ax.step(x, y, where='mid', **kw)
        # clip edges to MIN_VAL / MAX_VAL
        lo_cut, hi_cut = MIN_VAL[key], MAX_VAL[key]
        mask_e      = (edges >= lo_cut) & (edges <= hi_cut)
        edges_cut   = edges[mask_e]
        centers_cut = 0.5 * (edges_cut[:-1] + edges_cut[1:])
        valid       = np.where(mask_e[:-1] & mask_e[1:])[0]
        lims        = [lo_cut, hi_cut]

        # perfect prediction diagonal
        ax.plot(lims, lims, color='gray', alpha=0.5, linewidth=1.2,
                linestyle=':')
        # 95% quantile lines (dotted)
        stairs_pair(ax, centers_cut, q2p5[valid],
                    color='steelblue', linewidth=0.8, linestyle='dotted', alpha=0.7, label='95%')
        stairs_pair(ax, centers_cut, q97p5[valid],
                    color='steelblue', linewidth=0.8, linestyle='dotted', alpha=0.7)
        # 68% quantile lines (dashed)
        stairs_pair(ax, centers_cut, q16[valid],
                    color='steelblue', linewidth=0.8, linestyle='dashed', alpha=0.9, label='68%')
        stairs_pair(ax, centers_cut, q84[valid],
                    color='steelblue', linewidth=0.8, linestyle='dashed', alpha=0.9)
        # median
        stairs_pair(ax, centers_cut, median_p[valid],
                    color='steelblue', linewidth=1.4, label='Median')
        # mean
        stairs_pair(ax, centers_cut, mean_p[valid],
                    color='purple', linewidth=1.2, alpha=0.8, label='Mean')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.set_xlabel('True ' + LABEL[key])
        ax.set_ylabel('Pred ' + LABEL[key])
        ax.legend(frameon=False, fontsize=9, loc='upper left')

    fig2.tight_layout()

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 3 – Uncertainty histograms
    # ═════════════════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(2, 2, figsize=(11, 8))

    data3 = [sig_mc, sig_q, sig_m1_pred, sig_m2_pred]
    keys3 = ['chirp_mass', 'mass_ratio', 'mass1', 'mass2']

    for ax, d, key in zip(axes3.flat, data3, keys3):
        hist_outline(ax, d, bins=20, color='black', linewidth=1.2)
        ax.set_xlabel(SIG_LABEL[key])
        ax.set_ylabel('Counts')

    fig3.tight_layout()

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 4 – Z-score histograms
    # ═════════════════════════════════════════════════════════════════════════
    fig4, axes4 = plt.subplots(2, 2, figsize=(11, 8))

    data4 = [z_mc, z_q, z_m1, z_m2]
    keys4 = ['chirp_mass', 'mass_ratio', 'mass1', 'mass2']

    for ax, d, key in zip(axes4.flat, data4, keys4):
        hist_outline(ax, d, bins=20, color='steelblue', linewidth=1.2)
        ax.axvline( 0, color='gray', linestyle='--',    linewidth=1.0)
        ax.axvline( 1, color='gray', linestyle='dotted', linewidth=1.0)
        ax.axvline(-1, color='gray', linestyle='dotted', linewidth=1.0)
        ax.set_xlabel(r'$(x_{\mathrm{pred}} - x_{\mathrm{true}})\,/\,' +
                      SIG_LABEL[key].replace('$', '') + '$')
        ax.set_ylabel('Counts')

    fig4.tight_layout()

    # ── save / show ───────────────────────────────────────────────────────────
    if save_path:
        save_path = Path(save_path)
        names = ['mchirp_q_hist', 'pred_vs_true', 'uncertainty_hist', 'z_score_hist']
        for fig, name in zip([fig1, fig2, fig3, fig4], names):
            out = save_path / f'{name}.png'
            fig.savefig(out, dpi=400, bbox_inches='tight')
            print(f'Saved {out}')
    else:
        plt.show()

    return fig1, fig2, fig3, fig4


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot GW parameter estimation results.')
    parser.add_argument('--csv_path', type=Path, help='Path to the input CSV file.')
    parser.add_argument('--save_path', type=Path, default=None,
                        help='Directory to save figures (e.g. output/).')
    args = parser.parse_args()
    plot_gw_results(csv_path=args.csv_path, save_path=args.save_path)