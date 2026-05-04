#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --time=2-00:00:00
#SBATCH --job-name=1-5s_snr15_1000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_snr_15_1000/chirp_mass/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/ai4gw@cern/s4d_gaussnll_snr_15_1000/chirp_mass/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/ai4gw@cern/s4d_gaussnll_snr_15_1000/chirp_mass_1-5s_d128_s64_l4.yaml
echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
srun --cpu-bind=none python -m BNSReg.core.main fit \
    --config "$config_path"
srun --cpu-bind=none python -m BNSReg.core.main test \
    --config "$config_path"