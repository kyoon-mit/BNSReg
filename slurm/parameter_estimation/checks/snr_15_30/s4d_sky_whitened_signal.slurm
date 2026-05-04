#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=12:00:00
#SBATCH --job-name=s4d_sky_whitened_signal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/checks/s4d_sky_snr_15_30/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/checks/s4d_sky_snr_15_30/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/checks/s4d_sky_snr_15_30/check_whitened_signal_d256_s32_l4.yaml

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
python -m BNSReg.core.main_fit_test fit --config "$config_path"
