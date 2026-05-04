#!/bin/bash
#SBATCH --partition=gpu_requeue
#SBATCH --time=0:10:00
#SBATCH --job-name=apply_denoiser
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/eval/apply_denoiser/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/eval/apply_denoiser/error-%j.err

# mkdir -p /n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/eval/apply_denoiser

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

CKPT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs/ai4gw@cern__s4d_next_token_snr_10/seq_denoise_0-4s_d32_s32_l4/checkpoints/s4d_next_token_ckpt_epoch=53.ckpt
YAML=${TOP_DIR}/configs/seq_denoise/ai4gw@cern/s4d_next_token_snr_10/seq_denoise_0-4s_d32_s32_l4.yaml

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

echo "Running on host: $(hostname)"

cd ${TOP_DIR}

python scripts/eval/apply_denoiser.py \
    --ckpt  "${CKPT}" \
    --yaml  "${YAML}" \
    --batch-size 256 \
    --num-workers 4
