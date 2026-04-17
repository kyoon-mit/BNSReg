#!/bin/bash
#SBATCH --partition=gpu_requeue
#SBATCH --time=12:00:00
#SBATCH --job-name=chirp_mass_curriculum
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_snr_5_50/chirp_mass_curriculum/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_snr_5_50/chirp_mass_curriculum/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/ai4gw@cern/s4d_gaussnll_curriculum/chirp_mass_1-5s_d64_s16_l8.yaml
echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
srun --cpu-bind=none python -m BNSReg.core.main_fit_test fit \
    --config "$config_path"