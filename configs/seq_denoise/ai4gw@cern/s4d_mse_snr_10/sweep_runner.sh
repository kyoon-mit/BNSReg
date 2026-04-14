#!/bin/bash
# Wrapper called by wandb sweep agent for the seq_denoise MSE sweep.
# Parses model architecture args from wandb and runs fit+test.

CONFIG=""
D_MODEL=""
D_STATE=""
N_LAYERS=""
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --config=*)                              CONFIG="${1#--config=}"; shift;;
    --config)                               CONFIG="$2"; shift 2;;
    --model.init_args.cfg.d_model=*)        D_MODEL="${1#--model.init_args.cfg.d_model=}"; OTHER_ARGS+=("$1"); shift;;
    --model.init_args.cfg.d_model)          D_MODEL="$2"; OTHER_ARGS+=("$1" "$2"); shift 2;;
    --model.init_args.cfg.d_state=*)        D_STATE="${1#--model.init_args.cfg.d_state=}"; OTHER_ARGS+=("$1"); shift;;
    --model.init_args.cfg.d_state)          D_STATE="$2"; OTHER_ARGS+=("$1" "$2"); shift 2;;
    --model.init_args.cfg.n_layers=*)       N_LAYERS="${1#--model.init_args.cfg.n_layers=}"; OTHER_ARGS+=("$1"); shift;;
    --model.init_args.cfg.n_layers)         N_LAYERS="$2"; OTHER_ARGS+=("$1" "$2"); shift 2;;
    *) OTHER_ARGS+=("$1"); shift;;
  esac
done

D_MODEL="${D_MODEL:-32}"
D_STATE="${D_STATE:-32}"
N_LAYERS="${N_LAYERS:-4}"

# Adaptive batch size: scale down with larger d_model and more layers.
# Base: 256 @ d_model=32, n_layers=4.
BATCH_SIZE=$(( 256 * 32 * 4 / D_MODEL / N_LAYERS ))
if (( BATCH_SIZE < 16 )); then BATCH_SIZE=16; fi

RUN_NAME="seq_denoise_0-4s_d${D_MODEL}_s${D_STATE}_l${N_LAYERS}"
SAVE_DIR="/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs/ai4gw@cern/s4d_mse_snr_10/${RUN_NAME}"

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

python -m BNSReg.core.main_fit_test fit \
  --config "$CONFIG" \
  "${OTHER_ARGS[@]}" \
  --trainer.logger.init_args.name     "$RUN_NAME" \
  --trainer.logger.init_args.save_dir "$SAVE_DIR" \
  --data.init_args.data_cfg.train_batch_size "$BATCH_SIZE" \
  --data.init_args.data_cfg.val_batch_size   "$BATCH_SIZE" \
  --data.init_args.data_cfg.test_batch_size  "$BATCH_SIZE"
