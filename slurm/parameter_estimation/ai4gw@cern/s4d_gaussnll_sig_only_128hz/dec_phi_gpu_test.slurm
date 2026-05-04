#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=12:00:00
#SBATCH --job-name=sky_sig128hz
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_sig_only_128hz/dec_phi/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_sig_only_128hz/dec_phi/error-%j.err

set -e

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg
CONFIG=${TOP_DIR}/configs/parameter_estimation/ai4gw@cern/s4d_gaussnll_sig_only_128hz/dec_phi_0_4s_d512_s32_l4.yaml
MARKER_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_sig_only_128hz/dec_phi/markers

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
#   0-4s  -> 512 samples  -> batch 128
#   0-8s  -> 1024 samples -> batch 64
#   0-16s -> 2048 samples -> batch 32
#   0-32s -> 4096 samples -> batch 16
# ------------------------------------------------------------------
for ARGS in "0 4 128" "0 8 64" "0 16 32" "0 32 16"; do
  read -r begin end batch <<< "$ARGS"
  RUN_NAME="dec_phi_${begin}_${end}s_d512_s32_l4"
  MARKER="$MARKER_DIR/${begin}_${end}s.done"

  if [ -f "$MARKER" ]; then
    echo "=== Skipping ${RUN_NAME} (already done) ==="
    continue
  fi

  echo "=== Training: ${RUN_NAME} ==="
  python -m BNSReg.core.main_fit_test fit \
    --config "$CONFIG" \
    --data.init_args.data_cfg.window_begin "$begin" \
    --data.init_args.data_cfg.window_end "$end" \
    --data.init_args.data_cfg.train_batch_size "$batch" \
    --data.init_args.data_cfg.val_batch_size "$batch" \
    --data.init_args.data_cfg.test_batch_size "$batch" \
    --trainer.logger.init_args.name "$RUN_NAME"
  touch "$MARKER"
  echo "=== Done: ${RUN_NAME} ==="
done

echo "=== All dec_phi windows completed ==="
