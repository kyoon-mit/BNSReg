#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=12:00:00
#SBATCH --job-name=linoss_snr15_30_59-63s_d256_s32_l4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/checks/linoss_gaussnll_snr_15_30/chirp_mass/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/checks/linoss_gaussnll_snr_15_30/chirp_mass/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/checks/linoss_gaussnll_snr_15_30/chirp_mass_59-63s_d256_s32_l4_paramnorm.yaml

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
python -m BNSReg.core.main_fit_test fit --config "$config_path"