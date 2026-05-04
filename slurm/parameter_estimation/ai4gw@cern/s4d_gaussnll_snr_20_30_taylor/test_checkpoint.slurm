#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=00:30:00
#SBATCH --job-name=snr30_50_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_snr_20_30_taylor/chirp_mass_sweep/test-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_snr_20_30_taylor/chirp_mass_sweep/test-%j.err

# Usage:
#   sbatch --export=CKPT=<path/to/checkpoint.ckpt>,D_MODEL=512,D_STATE=64,WIN_BEGIN=55,WIN_END=63 \
#     slurm/parameter_estimation/ai4gw@cern/s4d_gaussnll_snr_30_50/test_checkpoint.slurm

CKPT="${CKPT:?ERROR: CKPT not set}"
D_MODEL="${D_MODEL:-64}"
D_STATE="${D_STATE:-32}"
WIN_BEGIN="${WIN_BEGIN:-59}"
WIN_END="${WIN_END:-63}"
RUN_ID=$(basename "$(dirname "$(dirname "$CKPT")")")
PLOTS_DIR="$(dirname "$(dirname "$CKPT")")/plots"

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

exec 2>&1   # merge stderr into stdout so errors appear in the .out file

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312
cd ${TOP_DIR}

echo "Running on host: $(hostname)"
echo "Checkpoint: ${CKPT}"
echo "Run ID: ${RUN_ID}, d_model=${D_MODEL}, d_state=${D_STATE}, window=${WIN_BEGIN}-${WIN_END}s"
echo "Plots dir: ${PLOTS_DIR}"

python -m BNSReg.core.main_fit_test test \
  --config configs/parameter_estimation/ai4gw@cern/s4d_gaussnll_snr_20_30_taylor/base_chirp_mass.yaml \
  --model.init_args.model_cfg.d_model "${D_MODEL}" \
  --model.init_args.model_cfg.d_state "${D_STATE}" \
  --data.init_args.data_cfg.window_begin "${WIN_BEGIN}" \
  --data.init_args.data_cfg.window_end "${WIN_END}" \
  --data.init_args.data_cfg.test_batch_size 128 \
  --trainer.logger.init_args.id "${RUN_ID}" \
  --trainer.logger.init_args.name "Mc_${WIN_BEGIN}_${WIN_END}s_d${D_MODEL}_s${D_STATE}" \
  --ckpt_path "${CKPT}"
echo "Python exit code: $?"
