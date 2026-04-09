from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── predefined bins ───────────────────────────────────────────────────────────

BINS = np.linspace(0.5, 3.0, 51)
MIN_VAL = 0.85 - 0.05
MAX_VAL = 2.2 + 0.05

# ── LaTeX labels ──────────────────────────────────────────────────────────────

LABEL     = r'$\mathcal{M}_{c}$'
SIG_LABEL = r'$\sigma_{\mathcal{M}_{c}}$'


# ── helpers ───────────────────────────────────────────────────────────────────

def hist_outline(ax, data, bins, **kwargs):
    """Draw a histogram as a step outline (no fill)."""
    counts, edges = np.histogram(data, bins=bins)
    ax.stairs(counts, edges, **kwargs)


# ── main function ─────────────────────────────────────────────────────────────

def plot_gw_results(csv_path: Path, save_path: Path | None = None):
    """
    Read a CSV with columns:
        M_chirp_true, M_chirp_pred, sigma_M_chirp_pred

    and produce four figures:
        1. Overlaid true/pred histograms for Mc
        2. Binned pred vs true: mean, median, 68%, 95% quantiles
        3. Uncertainty histogram for sigma_Mc
        4. Z-score histogram for Mc
    """
    df = pd.read_csv(csv_path)

    mc_true = df['M_chirp_true'].values
    mc_pred = df['M_chirp_pred'].values
    sig_mc  = df['sigma_M_chirp_pred'].values

    z_mc = (mc_pred - mc_true) / sig_mc

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 1 – True vs Predicted histograms for Mc
    # ═════════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))

    hist_outline(ax1, mc_true, bins=BINS, color='red',   linewidth=1.2, label='True')
    hist_outline(ax1, mc_pred, bins=BINS, color='green', linewidth=1.2, label='Predicted')
    ax1.set_xlabel(LABEL)
    ax1.set_ylabel('Counts')
    ax1.legend(frameon=False)

    fig1.tight_layout()

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 2 – Binned pred vs true: mean, median, 68%, 95% quantiles
    # ═════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(6, 6))

    edges   = BINS
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean_p   = np.full(len(centers), np.nan)
    median_p = np.full(len(centers), np.nan)
    q16      = np.full(len(centers), np.nan)
    q84      = np.full(len(centers), np.nan)
    q2p5     = np.full(len(centers), np.nan)
    q97p5    = np.full(len(centers), np.nan)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (mc_true >= lo) & (mc_true < hi)
        if mask.sum() < 2:
            continue
        p_bin        = mc_pred[mask]
        mean_p[i]    = np.mean(p_bin)
        median_p[i]  = np.median(p_bin)
        q16[i]       = np.percentile(p_bin, 16)
        q84[i]       = np.percentile(p_bin, 84)
        q2p5[i]      = np.percentile(p_bin,  2.5)
        q97p5[i]     = np.percentile(p_bin, 97.5)

    def stairs_pair(ax, x, y, **kw):
        ax.step(x, y, where='mid', **kw)

    mask_e      = (edges >= MIN_VAL) & (edges <= MAX_VAL)
    edges_cut   = edges[mask_e]
    centers_cut = 0.5 * (edges_cut[:-1] + edges_cut[1:])
    valid       = np.where(mask_e[:-1] & mask_e[1:])[0]
    lims        = [MIN_VAL, MAX_VAL]

    ax2.plot(lims, lims, color='gray', alpha=0.5, linewidth=1.2, linestyle=':')
    stairs_pair(ax2, centers_cut, q2p5[valid],
                color='steelblue', linewidth=0.8, linestyle='dotted', alpha=0.7, label='95%')
    stairs_pair(ax2, centers_cut, q97p5[valid],
                color='steelblue', linewidth=0.8, linestyle='dotted', alpha=0.7)
    stairs_pair(ax2, centers_cut, q16[valid],
                color='steelblue', linewidth=0.8, linestyle='dashed', alpha=0.9, label='68%')
    stairs_pair(ax2, centers_cut, q84[valid],
                color='steelblue', linewidth=0.8, linestyle='dashed', alpha=0.9)
    stairs_pair(ax2, centers_cut, median_p[valid],
                color='steelblue', linewidth=1.4, label='Median')
    stairs_pair(ax2, centers_cut, mean_p[valid],
                color='purple', linewidth=1.2, alpha=0.8, label='Mean')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_aspect('equal')
    ax2.set_xlabel('True ' + LABEL)
    ax2.set_ylabel('Pred ' + LABEL)
    ax2.legend(frameon=False, fontsize=9, loc='upper left')

    fig2.tight_layout()

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 3 – Uncertainty histogram
    # ═════════════════════════════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))

    hist_outline(ax3, sig_mc, bins=20, color='black', linewidth=1.2)
    ax3.set_xlabel(SIG_LABEL)
    ax3.set_ylabel('Counts')

    fig3.tight_layout()

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 4 – Z-score histogram
    # ═════════════════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=(6, 4.5))

    hist_outline(ax4, z_mc, bins=20, color='steelblue', linewidth=1.2)
    ax4.axvline( 0, color='gray', linestyle='--',     linewidth=1.0)
    ax4.axvline( 1, color='gray', linestyle='dotted', linewidth=1.0)
    ax4.axvline(-1, color='gray', linestyle='dotted', linewidth=1.0)
    ax4.set_xlabel(r'$(\mathcal{M}_{c,\mathrm{pred}} - \mathcal{M}_{c,\mathrm{true}})\,/\,\sigma_{\mathcal{M}_{c}}$')
    ax4.set_ylabel('Counts')

    fig4.tight_layout()

    # ── save / show ───────────────────────────────────────────────────────────
    if save_path:
        save_path = Path(save_path)
        names = ['mchirp_hist', 'pred_vs_true', 'uncertainty_hist', 'z_score_hist']
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
    parser = argparse.ArgumentParser(description='Plot GW chirp mass estimation results.')
    parser.add_argument('--csv_path', type=Path, help='Path to the input CSV file.')
    parser.add_argument('--save_path', type=Path, default=None,
                        help='Directory to save figures (e.g. output/).')
    args = parser.parse_args()
    plot_gw_results(csv_path=args.csv_path, save_path=args.save_path)
