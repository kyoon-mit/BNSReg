#!/bin/bash
#SBATCH --partition=gpu_requeue
#SBATCH --time=1-00:00:00
#SBATCH --job-name=check_snr20_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_snr_20_30_taylor/chirp_mass_sweep/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_snr_20_30_taylor/chirp_mass_sweep/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg
SWEEP_FILE=${TOP_DIR}/configs/parameter_estimation/checks/s4d_gaussnll_snr_20/Mc_arch_sweep.yaml

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

cd ${TOP_DIR}

echo "Running on host: $(hostname)"

WANDB_OUTPUT=$(wandb sweep "$SWEEP_FILE" 2>&1)
SWEEP_ID=$(echo "$WANDB_OUTPUT" | grep -oP '(?<=wandb agent )\S+' | head -1)

echo "Sweep ID: ${SWEEP_ID}"

wandb agent ${SWEEP_ID}
