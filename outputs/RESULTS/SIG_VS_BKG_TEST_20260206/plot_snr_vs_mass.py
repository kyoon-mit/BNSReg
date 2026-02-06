import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot(
    csv_path: str | Path,
    save_dir: str | Path,
    snr_low: int,
    snr_high: int,
):
    df = pd.read_csv(csv_path)
    df['diff_chirp_mass'] = df['pred_chirp_mass'] - df['truth_chirp_mass']
    df['rel_error'] = (df['diff_chirp_mass'])/df['truth_chirp_mass']

    # Define cuts
    log_cuts = (
        np.arange(1, 10)[:, None]
        * 10.0**np.arange(-4, 0)
    ).T.flatten()
    log_cuts = np.append(log_cuts, 1.0)

    # Define SNR bins
    snr_bins = np.linspace(snr_low, snr_high+1, snr_high+1-snr_low, endpoint=False)
    snr_binned = pd.cut(df['snr'], bins=snr_bins)

    cuts = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

    # total events per SNR bin (independent of cut)
    bin_totals = df.groupby(snr_binned, observed=True).size()

    # Plot data for each cutoff
    rows = []

    for cut in cuts:
        mask = abs(df['rel_error']) < cut

        counts = (
            df.loc[mask]
            .groupby(snr_binned, observed=True)
            .size()
            .reindex(bin_totals.index, fill_value=0)
        )

        tmp = (counts / bin_totals).reset_index(name='fraction')
        tmp['snr_center'] = tmp['snr'].apply(lambda x: x.mid)
        tmp['cut'] = cut

        rows.append(tmp)

    df_plot = pd.concat(rows, ignore_index=True)
    fig_snr_vs_cutoff = px.line(
        df_plot,
        x='snr_center',
        y='fraction',
        color='cut',
        markers=True,
        labels={
            'snr_center': 'SNR (binned)',
            'fraction': 'Fraction of events below cutoff (per bin)',
            'cut': 'Cutoff'
        },
        title='SNR vs. Fraction of events below cutoff (per bin)'
    )

    fig_snr_vs_cutoff.show()
    fig_snr_vs_cutoff.write_html(Path(save_dir) / 'snr_vs_cutoff_frac_events.html')

    # Violin plot
    df['snr_bin'] = pd.cut(df['snr'], bins=snr_bins)
    snr_cat = df['snr_bin'].cat.categories.astype(str).tolist()
    df['snr_bin_str'] = pd.Categorical(df['snr_bin'].astype(str), categories=snr_cat, ordered=True)
    df = df.sort_values('snr_bin_str')

    fig_violin = px.violin(df,
        x='snr_bin_str',
        y='rel_error',
        box=True,
        points=False,
        title='Violin plot of snr vs. (pred_chirp - true_chirp)/true_chirp'
    )
    fig_violin.write_html(Path(save_dir) / 'violin_snr_vs_rel_error.html')

    # Scatter plot
    fig_scatter = px.scatter(df,
        x='snr',
        y='rel_error',
        hover_data=['rel_error'],
        title='Scatter plot of snr vs. (pred_chirp - true_chirp)/true_chirp'
    )
    fig_scatter.write_html(Path(save_dir) / 'scatter_snr_vs_rel_error.html')

    # Histogram
    # Step-style histograms (no filled bars), bin width = 0.01
    bin_width = 0.05
    xmin = 0.75
    xmax = 2.25
    bins = np.arange(xmin, xmax + bin_width, bin_width)
    truth_hist, edges = np.histogram(my_csv_df['truth_chirp_mass'], bins=bins, density=True)
    pred_hist, edges_pred = np.histogram(my_csv_df['pred_chirp_mass'], bins=bins, density=True)
    # Build step coordinates for plotting (horizontal segments with vertical jumps)
    def make_step_xy(hist, edges):
        x = np.empty(2 * len(hist))
        y = np.empty(2 * len(hist))
        for i in range(len(hist)):
            x[2*i] = edges[i]
            x[2*i+1] = edges[i+1]
            y[2*i] = hist[i]
            y[2*i+1] = hist[i]
        return x, y

    truth_x, truth_y = make_step_xy(truth_hist, edges)
    pred_x, pred_y = make_step_xy(pred_hist, edges)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=truth_x, y=truth_y, mode='lines', name='truth_chirp_mass', line=dict(color='red')))
    fig_hist.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='lines', name='pred_chirp_mass', line=dict(color='green')))
    fig_hist.update_layout(xaxis_title='Chirp mass', yaxis_title='Density', title='Predicted and Truth Chirp Masses')
    fig_hist.show()
    fig_hist.write_html(Path(save_dir) / 'hist_pred_chirp_truth_chirp.html')

    return

if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot SNR vs chirp mass metrics')
    parser.add_argument('--csv_path', type=str, help='Path to CSV file with predictions and truth values')
    parser.add_argument('--save_dir', type=str, help='Directory to save output plots')
    parser.add_argument('--snr-low', type=int, help='Minimum SNR bin value')
    parser.add_argument('--snr-high', type=int, help='Maximum SNR bin value')
    
    args = parser.parse_args()
    
    plot(
        csv_path=args.csv_path,
        save_dir=args.save_dir,
        snr_low=args.snr_low,
        snr_high=args.snr_high,
    )
