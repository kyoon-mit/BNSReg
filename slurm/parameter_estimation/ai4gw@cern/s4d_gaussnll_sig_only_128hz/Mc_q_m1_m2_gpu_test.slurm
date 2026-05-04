#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --time=3-00:00:00
#SBATCH --job-name=mass4_sig128hz
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_sig_only_128hz/Mc_q_m1_m2/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_sig_only_128hz/Mc_q_m1_m2/error-%j.err

set -e

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg
CONFIG=${TOP_DIR}/configs/parameter_estimation/ai4gw@cern/s4d_gaussnll_sig_only_128hz/Mc_q_m1_m2_0_4s_d512_s32_l4.yaml
MARKER_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_sig_only_128hz/Mc_q_m1_m2/markers

mkdir -p "$MARKER_DIR"

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

cd ${TOP_DIR}
echo "Host: $(hostname)"

# ------------------------------------------------------------------
MARKER_FDR="$MARKER_DIR/fast_dev_run.done"
if [ ! -f "$MARKER_FDR" ]; then
  echo "=== Fast dev run ==="
  python -m BNSReg.core.main_fit_test fit \
    --config "$CONFIG" \
    --trainer.fast_dev_run true
  touch "$MARKER_FDR"
  echo "=== Fast dev run passed ==="
else
  echo "=== Fast dev run already done, skipping ==="
fi

# ------------------------------------------------------------------
# Sweep d_model: 512 -> 256 -> 128 -> 64
# Fixed: 0-4s window, d_state=32, n_layers=4
# Batch scales inversely with d_model (H200 has headroom):
#   d512 -> 256 | d256 -> 512 | d128 -> 1024 | d64 -> 2048
# ------------------------------------------------------------------
for ARGS in "512 256" "256 512" "128 1024" "64 2048"; do
  read -r D_MODEL BATCH <<< "$ARGS"
  RUN_NAME="Mc_q_m1_m2_0_4s_d${D_MODEL}_s32_l4"
  MARKER="$MARKER_DIR/d${D_MODEL}.done"

  if [ -f "$MARKER" ]; then
    echo "=== Skipping ${RUN_NAME} (already done) ==="
    continue
  fi

  echo "=== Training: ${RUN_NAME} (batch=${BATCH}) ==="
  python -m BNSReg.core.main_fit_test fit \
    --config "$CONFIG" \
    --model.init_args.model_cfg.d_model "$D_MODEL" \
    --data.init_args.data_cfg.train_batch_size "$BATCH" \
    --data.init_args.data_cfg.val_batch_size "$BATCH" \
    --data.init_args.data_cfg.test_batch_size "$BATCH" \
    --trainer.logger.init_args.name "$RUN_NAME"
  touch "$MARKER"
  echo "=== Done: ${RUN_NAME} ==="
done

echo "=== All Mc_q_m1_m2 d_model sweep completed ==="
