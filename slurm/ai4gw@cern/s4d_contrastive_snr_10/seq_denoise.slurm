#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=12:00:00
#SBATCH --job-name=s4d_contrastive_snr10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_contrastive_snr_10/seq_denoise/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_contrastive_snr_10/seq_denoise/error-%j.err

# mkdir -p /n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_contrastive_snr_10/seq_denoise

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg
CONFIG=${TOP_DIR}/configs/seq_denoise/ai4gw@cern/s4d_contrastive_snr_10/seq_denoise_0-4s_d32_s32_l4.yaml

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

cd ${TOP_DIR}

echo "Running on host: $(hostname)"
echo "Using config: ${CONFIG}"

srun --cpu-bind=none python -m BNSReg.core.main fit \
    --config "${CONFIG}"
