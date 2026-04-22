#!/bin/bash
# Wrapper for wandb sweep agent.
# Handles coupled time_window param and sets early stopping patience.
# Usage: called by wandb agent via sweep yaml command section.

TIME_WINDOW=""
VAR=""
D_MODEL=""
D_STATE=""
N_LAYERS=""
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --time_window=*) TIME_WINDOW="${1#--time_window=}"; shift;;
    --time_window)   TIME_WINDOW="$2"; shift 2;;
    --var=*)         VAR="${1#--var=}"; shift;;
    --var)           VAR="$2"; shift 2;;
    --model.init_args.model_cfg.d_model=*) D_MODEL="${1#--model.init_args.model_cfg.d_model=}"; OTHER_ARGS+=("$1"); shift;;
    --model.init_args.model_cfg.d_model)   D_MODEL="$2"; OTHER_ARGS+=("$1" "$2"); shift 2;;
    --model.init_args.model_cfg.d_state=*) D_STATE="${1#--model.init_args.model_cfg.d_state=}"; OTHER_ARGS+=("$1"); shift;;
    --model.init_args.model_cfg.d_state)   D_STATE="$2"; OTHER_ARGS+=("$1" "$2"); shift 2;;
    --model.init_args.model_cfg.n_layers=*) N_LAYERS="${1#--model.init_args.model_cfg.n_layers=}"; OTHER_ARGS+=("$1"); shift;;
    --model.init_args.model_cfg.n_layers)   N_LAYERS="$2"; OTHER_ARGS+=("$1" "$2"); shift 2;;
    *) OTHER_ARGS+=("$1"); shift;;
  esac
done

IFS=',' read -r begin end <<< "$TIME_WINDOW"

D_MODEL="${D_MODEL:-32}"
D_STATE="${D_STATE:-32}"
N_LAYERS="${N_LAYERS:-4}"

# Adaptive batch size: scale down with larger d_model, more layers, and longer sequences.
# Memory scales ~ d_model * n_layers * seq_len, so batch size scales inversely.
# Base: 512 @ d_model=32, n_layers=4, seq_len=4s (1024 samples @ 256 Hz).
SEQ_LEN=$(( end - begin ))   # in seconds; relative to base of 4s
BATCH_SIZE=$(( 512 * 32 * 4 * 4 / D_MODEL / N_LAYERS / SEQ_LEN ))
if (( BATCH_SIZE < 16 )); then BATCH_SIZE=16; fi

RUN_NAME="${VAR}_${begin}_${end}s_d${D_MODEL}_s${D_STATE}_l${N_LAYERS}"
SAVE_DIR="/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs/ai4gw@cern__s4d_gaussnll_snr_15/${RUN_NAME}"

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

python -m BNSReg.core.main_fit_test fit \
  "${OTHER_ARGS[@]}" \
  --data.init_args.data_cfg.window_begin "$begin" \
  --data.init_args.data_cfg.window_end "$end" \
  --trainer.logger.init_args.name "$RUN_NAME" \
  --data.init_args.data_cfg.train_batch_size "$BATCH_SIZE" \
  --data.init_args.data_cfg.val_batch_size "$BATCH_SIZE" \
  --data.init_args.data_cfg.test_batch_size "$BATCH_SIZE"
