#!/bin/bash
#SBATCH --partition=gpu_requeue
#SBATCH --time=2-00:00:00
#SBATCH --job-name=sweep_worker
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs/slurm_logs/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs/slurm_logs/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

echo "Running on host: $(hostname)"
echo "Variable: ${VAR}"
echo "Sweep ID: ${SWEEP_ID}"

cd ${TOP_DIR}
wandb agent ${SWEEP_ID}
