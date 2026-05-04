#!/bin/bash
#SBATCH --partition=iaifi_gpu_requeue
#SBATCH --time=6:00:00
#SBATCH --job-name=finetune_snr_10_20_uniform_to_snr_5_50_POWERLAW
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/fine_tune/chirp_mass/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/fine_tune/chirp_mass/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/fine_tune/s4d_gaussnll_snr_10_20_uniform_to_snr_5_50_powerlaw/chirp_mass_59-63s_d512_s32_l4.yaml
echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
python -m BNSReg.core.main_fit_test fit --config "$config_path"