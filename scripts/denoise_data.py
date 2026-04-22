"""
denoise_data.py

Runs the trained LitModelConvAttentionAEAP denoising model (loaded from a YAML
config + checkpoint) over the train/val/test HDF5 files and writes new HDF5 files
with the same directory structure and the same HDF5 keys.

Output 'injected_data' has shape (N, n_ifos, seq_len) where seq_len is determined
by the window / sampling settings in the config, e.g. for window=[0,55]s at 512Hz
downsampled by 2: seq_len = 14080.  sig_only_data is also saved at the same window.
All parameter datasets (chirp_mass, mass_ratio, snr, …) are copied verbatim.

Output files are written to the same parent directory as the source files with
'_denoised' appended to the stem, e.g.:
    .../train/sig_combined_train.h5  →  .../train/sig_combined_train_denoised.h5

Because the output is already downsampled, use these settings in the regression
config to read back:
    strain_frequency  = strain_frequency / downsample_factor   (e.g. 256)
    downsample_factor = 1
    strain_duration   = window_end - window_begin              (e.g. 55)
    window_begin      = 0
    window_end        = window_end                             (e.g. 55)

Usage:
    python denoise_data.py --config configs/seq_denoise/user/conv_attn_ae_ap/h1_0_55s-n4-b4-h16.yaml
    python denoise_data.py --config <yaml> --ckpt <path_to_ckpt>  # override checkpoint
    python denoise_data.py --config <yaml> --limit_n 200           # smoke-test
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from BNSReg.core.config import ConvAttentionAEAPModelConfig
from BNSReg.tasks.seq_denoise.conv_ae import LitModelConvAttentionAEAP

# Keys holding strain data that we re-derive from the denoiser (do not blindly copy)
STRAIN_KEYS = {'injected_data', 'sig_only_data', 'bkg_only_data'}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_model_cfg(cfg: dict) -> ConvAttentionAEAPModelConfig:
    return ConvAttentionAEAPModelConfig(**cfg['model']['init_args']['model_cfg'])


def parse_data_params(cfg: dict) -> dict:
    """Return the raw data_cfg dict from the YAML."""
    return cfg['data']['init_args']['data_cfg']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    ckpt_path: str,
    model_cfg: ConvAttentionAEAPModelConfig,
    device: torch.device,
) -> LitModelConvAttentionAEAP:
    # Instantiate directly from the YAML config, then load the state dict.
    # load_from_checkpoint goes through jsonargparse and fails on hparam
    # validation when the base-class type annotation doesn't include subclass fields.
    model = LitModelConvAttentionAEAP(model_cfg=model_cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def denoise_split(
    src_path: str,
    out_path: Path,
    model: LitModelConvAttentionAEAP,
    start_idx: int,
    end_idx: int,
    downsample: int,
    injected_key: str,
    sig_only_key: str,
    batch_size: int,
    device: torch.device,
    limit_n: int | None = None,
) -> None:
    with h5py.File(src_path, 'r') as in_f:
        N_total, n_ifos, full_len = in_f[injected_key].shape

    N = N_total if limit_n is None else min(limit_n, N_total)
    seq_len = (end_idx - start_idx + downsample - 1) // downsample  # ceil-safe

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'  src : {src_path}')
    print(f'  out : {out_path}')
    print(f'  N={N}, n_ifos={n_ifos}, seq_len={seq_len}')

    with h5py.File(src_path, 'r') as in_f, h5py.File(out_path, 'w') as out_f:
        # Pre-allocate strain datasets
        out_f.create_dataset('injected_data', shape=(N, n_ifos, seq_len), dtype='float32')
        if sig_only_key and sig_only_key in in_f:
            out_f.create_dataset('sig_only_data', shape=(N, n_ifos, seq_len), dtype='float32')

        # Copy all parameter / metadata keys verbatim (chirp_mass, mass_ratio, snr, …)
        for key in in_f.keys():
            if key in STRAIN_KEYS:
                continue
            src_ds = in_f[key]
            out_f.create_dataset(key, data=src_ds[:N])

        # Batch inference loop
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            B = batch_end - batch_start

            # Read windowed + downsampled strain for all IFOs: (B, n_ifos, seq_len)
            inj = in_f[injected_key][batch_start:batch_end, :, start_idx:end_idx:downsample]
            inj = inj.astype(np.float32)

            denoised = np.zeros_like(inj)  # (B, n_ifos, seq_len)
            with torch.no_grad():
                for ifo in range(n_ifos):
                    x = torch.from_numpy(inj[:, ifo, :]).to(device)  # (B, seq_len)
                    y = model(x)                                       # (B, 1, seq_len)
                    denoised[:, ifo, :] = y[:, 0, :].cpu().numpy()

            out_f['injected_data'][batch_start:batch_end] = denoised

            if sig_only_key and sig_only_key in in_f and 'sig_only_data' in out_f:
                sig = in_f[sig_only_key][batch_start:batch_end, :, start_idx:end_idx:downsample]
                out_f['sig_only_data'][batch_start:batch_end] = sig.astype(np.float32)

            if (batch_start // batch_size + 1) % 50 == 0:
                print(f'    batch {batch_start // batch_size + 1}  ({batch_end}/{N})')

    print(f'  done.\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, ckpt_override: str | None, limit_n: int | None) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    cfg = load_yaml(config_path)
    model_cfg = parse_model_cfg(cfg)
    data_params = parse_data_params(cfg)

    ckpt_path = ckpt_override or cfg.get('ckpt_path')
    if not ckpt_path:
        raise ValueError('No checkpoint path in YAML and none provided via --ckpt.')
    print(f'Checkpoint : {ckpt_path}')

    model = load_model(ckpt_path, model_cfg, device)
    print('Model loaded.\n')

    # Window / sampling settings
    strain_freq   = int(data_params['strain_frequency'])
    downsample    = int(data_params['downsample_factor'])
    window_begin  = float(data_params['window_begin'])
    window_end    = float(data_params['window_end'])
    start_idx     = int(window_begin * strain_freq)
    end_idx       = int(window_end   * strain_freq)
    batch_size    = int(data_params.get('test_batch_size', 64))

    injected_key  = data_params.get('injected_data_key', 'injected_data')
    sig_only_key  = data_params.get('sig_only_data_key', '')

    splits = {
        'train': data_params['train_file'],
        'val':   data_params['val_file'],
        'test':  data_params['test_file'],
    }

    for split, src_path in splits.items():
        src = Path(src_path)
        out = src.parent / (src.stem + '_denoised' + src.suffix)
        print(f'=== {split.upper()} ===')
        denoise_split(
            src_path=str(src),
            out_path=out,
            model=model,
            start_idx=start_idx,
            end_idx=end_idx,
            downsample=downsample,
            injected_key=injected_key,
            sig_only_key=sig_only_key,
            batch_size=batch_size,
            device=device,
            limit_n=limit_n,
        )

    eff_freq = strain_freq // downsample
    duration = window_end - window_begin
    print('All splits done.')
    print(f'\nUse these settings in the regression config to read the denoised files:')
    print(f'  strain_frequency  = {eff_freq}')
    print(f'  downsample_factor = 1')
    print(f'  strain_duration   = {int(duration)}')
    print(f'  window_begin      = 0')
    print(f'  window_end        = {int(duration)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise H1+L1 strain data and save to HDF5.')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to the seq_denoise YAML config '
             '(e.g. configs/seq_denoise/user/conv_attn_ae_ap/h1_0_55s-n4-b4-h16.yaml).'
    )
    parser.add_argument(
        '--ckpt', type=str, default=None,
        help='Override the checkpoint path specified in the YAML.'
    )
    parser.add_argument(
        '--limit_n', type=int, default=None,
        help='Process only the first N samples per split (useful for smoke-testing).'
    )
    args = parser.parse_args()
    main(config_path=args.config, ckpt_override=args.ckpt, limit_n=args.limit_n)
    sys.exit(0)
