#!/usr/bin/env python
"""
Apply a trained seq-denoise checkpoint to the test split of a dataset and
write a denoised HDF5 file in the same format as the original.

Usage:
    python scripts/eval/apply_denoiser.py \
        --ckpt /path/to/s4d_mse_ckpt_epoch=123.ckpt \
        --yaml /path/to/seq_denoise_0-4s_d32_s32_l4.yaml \
        [--batch-size 256] [--num-workers 4]

Output layout (data_dir is derived from the test_file path in the YAML):
    <data_dir>/denoised/<yaml_stem>/
        denoised_sig_combined_test.h5   -- same keys as original, windowed to
                                           the model's window (e.g. 0–4 s);
                                           injected_data_key contains the
                                           denoised output
        <yaml_stem>.yaml                -- config copy
        <ckpt_filename>.ckpt            -- checkpoint copy
"""

import argparse
import importlib
import shutil
from pathlib import Path
from typing import get_type_hints

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from BNSReg.core.config import BNSDataModuleConfig
from BNSReg.dataset.seq_encoder_dataset import BNSDatasetSeqEncoder


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(yaml_cfg: dict, ckpt_path: str, device: torch.device):
    """Instantiate a LitModel from its YAML config and load checkpoint weights.

    Mirrors LitDenoiserBaseTask._load_denoiser so the same checkpoint format
    is supported regardless of which seq-denoise architecture was used.
    """
    mc = yaml_cfg['model']
    module_str, cls_name = mc['class_path'].rsplit('.', 1)
    cls = getattr(importlib.import_module(module_str), cls_name)

    # Reconstruct typed init args (e.g. cfg: S4DModelConfig) from the YAML dict.
    hints = get_type_hints(cls.__init__)
    kwargs = {}
    for param, val in mc.get('init_args', {}).items():
        if isinstance(val, dict) and param in hints:
            kwargs[param] = hints[param](**val)
        else:
            kwargs[param] = val

    model = cls(**kwargs)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Apply a seq-denoise checkpoint to the test dataset.'
    )
    parser.add_argument('--ckpt',        required=True, help='Path to .ckpt file')
    parser.add_argument('--yaml',        required=True, dest='yaml_path',
                        help='Path to the LightningCLI config YAML')
    parser.add_argument('--batch-size',  type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Load YAML ---------------------------------------------------------
    with open(args.yaml_path) as f:
        yaml_cfg = yaml.safe_load(f)

    data_cfg_dict: dict = yaml_cfg['data']['init_args']['data_cfg']
    test_file     = Path(data_cfg_dict['test_file'])
    injected_key  = data_cfg_dict.get('injected_data_key', 'data')
    sig_only_key  = data_cfg_dict.get('sig_only_data_key', '')

    # --- Derive output directory -------------------------------------------
    # test_file is expected at <data_dir>/test/sig_combined_test.h5
    data_dir  = test_file.parent.parent
    yaml_stem = Path(args.yaml_path).stem           # e.g. seq_denoise_0-4s_d32_s32_l4
    out_dir   = data_dir / 'denoised' / yaml_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5    = out_dir / 'denoised_sig_combined_test.h5'
    print(f'Output dir : {out_dir}')

    # --- Copy config artefacts ---------------------------------------------
    shutil.copy2(args.yaml_path, out_dir / Path(args.yaml_path).name)
    shutil.copy2(args.ckpt,      out_dir / Path(args.ckpt).name)
    print('Copied YAML and checkpoint to output dir.')

    # --- Dataset & DataLoader ----------------------------------------------
    cfg = BNSDataModuleConfig(
        **{
            **data_cfg_dict,
            'train_batch_size': args.batch_size,
            'val_batch_size':   args.batch_size,
            'test_batch_size':  args.batch_size,
            'shuffle':          False,
            'num_workers':      args.num_workers,
            'prefetch_factor':  args.num_workers if args.num_workers > 0 else None,
            'persistent_workers': args.num_workers > 0,
        }
    )
    dataset = BNSDatasetSeqEncoder('test', cfg)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.num_workers if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    n_samples = len(dataset)
    n_ifos    = dataset.n_ifos
    ds        = cfg.downsample_factor
    win_len   = len(range(dataset.start_idx, dataset.end_idx, ds))
    print(f'Samples: {n_samples} | IFOs: {n_ifos} | Window length: {win_len} samples')

    # --- Load model --------------------------------------------------------
    model = load_model(yaml_cfg, args.ckpt, device)
    print('Model loaded.')

    # --- Build output HDF5 -------------------------------------------------
    strain_keys = {injected_key}
    if sig_only_key:
        strain_keys.add(sig_only_key)

    with h5py.File(test_file, 'r') as f_in, \
         h5py.File(out_h5, 'w') as f_out:

        # Copy all non-strain datasets/groups verbatim (parameters, metadata…)
        for key in f_in.keys():
            if key not in strain_keys:
                f_in.copy(key, f_out)

        # Allocate windowed strain datasets
        f_out.create_dataset(
            injected_key,
            shape=(n_samples, n_ifos, win_len),
            dtype=np.float32,
        )
        if sig_only_key:
            f_out.create_dataset(
                sig_only_key,
                shape=(n_samples, n_ifos, win_len),
                dtype=np.float32,
            )

        # --- Inference loop ------------------------------------------------
        offset = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                B = inputs.size(0)

                # inputs:  (B, n_ifos, L)  float32
                # model expects (B, L, d_input), returns (B, L, d_output)
                x        = inputs.transpose(1, 2).to(device)
                denoised = model(x)                     # (B, L, d_output)
                denoised = denoised.transpose(1, 2)     # (B, n_ifos, L)

                f_out[injected_key][offset:offset + B] = (
                    denoised.cpu().float().numpy()
                )
                if sig_only_key:
                    f_out[sig_only_key][offset:offset + B] = (
                        targets.numpy()
                    )

                offset += B
                if batch_idx % 10 == 0:
                    print(f'  [{offset:>7}/{n_samples}]')

    print(f'\nDone. Written to:\n  {out_h5}')


if __name__ == '__main__':
    main()
