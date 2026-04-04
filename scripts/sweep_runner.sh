#!/bin/bash
# Wrapper for wandb sweep agent.
# Handles coupled time_window param and sets early stopping patience.
# Usage: called by wandb agent via sweep yaml command section.

TIME_WINDOW=""
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --time_window=*) TIME_WINDOW="${1#--time_window=}"; shift;;
    --time_window)   TIME_WINDOW="$2"; shift 2;;
    *) OTHER_ARGS+=("$1"); shift;;
  esac
done

IFS=',' read -r begin end <<< "$TIME_WINDOW"

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

python -m BNSReg.core.main fit \
  "${OTHER_ARGS[@]}" \
  --data.init_args.data_cfg.window_begin "$begin" \
  --data.init_args.data_cfg.window_end "$end"
