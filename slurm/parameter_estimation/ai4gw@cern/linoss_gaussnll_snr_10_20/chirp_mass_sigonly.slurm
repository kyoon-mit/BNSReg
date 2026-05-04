#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=12:00:00
#SBATCH --job-name=linoss_sigonly
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/linoss_gaussnll_snr_10_20/chirp_mass_sigonly/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/linoss_gaussnll_snr_10_20/chirp_mass_sigonly/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/ai4gw@cern/linoss_gaussnll_snr_10_20/chirp_mass_59-63s_linoss_d512_p32_l4_sigonly.yaml
echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
srun --cpu-bind=none python -m BNSReg.core.main fit \
    --config "$config_path"
srun --cpu-bind=none python -m BNSReg.core.main test \
    --config "$config_path"
