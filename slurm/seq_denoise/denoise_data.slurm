#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=12:00:00
#SBATCH --job-name=denoise
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

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/seq_denoise/user/conv_attn_ae_ap/h1_0_55s-n4-b4-h16.yaml
ckpt_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs/seq_denoise_snr_15_30_seq_0_55s/h1_0_55s-n4-b4-h16/checkpoints/conv_attn_ae_ap_ckpt_epoch=15.ckpt

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"
echo "Using checkpoint:  $ckpt_path"

cd ${TOP_DIR}
srun --cpu-bind=none python scripts/denoise_data.py \
    --config "$config_path" \
    --ckpt "$ckpt_path"
