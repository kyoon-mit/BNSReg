#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --job-name=s4d_snr_30_50_premerger
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/checks/s4d_gaussnll_snr_30_50/chirp_mass/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/checks/s4d_gaussnll_snr_30_50/chirp_mass/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/checks/snr_30_50/s4d_chirp_mass_1-31s_d256_s64_l4.yaml

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
srun --cpu-bind=none python -m BNSReg.core.main_fit_test fit --config "$config_path"