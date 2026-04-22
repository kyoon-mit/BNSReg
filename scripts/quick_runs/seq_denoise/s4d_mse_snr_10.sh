#!/usr/bin/env bash
# Quick sanity check (fast_dev_run=true) for every MSE denoising config.
# Runs fit then test for each config — mirrors the full training flow.
# Usage: bash scripts/quick_runs/seq_denoise/s4d_mse_snr_10.sh

set -e

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg
CONFIG_DIR=${TOP_DIR}/configs/seq_denoise/ai4gw@cern/s4d_mse_snr_10

cd ${TOP_DIR}

for config in \
    "${CONFIG_DIR}/seq_denoise_0-4s_d16_s16_l2.yaml" \
    "${CONFIG_DIR}/seq_denoise_0-4s_d32_s16_l2.yaml" \
    "${CONFIG_DIR}/seq_denoise_0-4s_d32_s32_l2.yaml" \
    "${CONFIG_DIR}/seq_denoise_0-4s_d32_s32_l4.yaml" \
    "${CONFIG_DIR}/seq_denoise_0-4s_d64_s32_l4.yaml"
do
    echo "========================================"
    echo "Config: $(basename ${config})"
    echo "========================================"

    python -m BNSReg.core.main fit \
        --config "${config}" \
        --trainer.fast_dev_run true \
        --trainer.logger false

    python -m BNSReg.core.main test \
        --config "${config}" \
        --trainer.fast_dev_run true \
        --trainer.logger false
done

echo "All configs passed fast_dev_run."
