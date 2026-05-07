#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=20:00
#SBATCH --job-name=mass_merger_bkg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/00_AI4GW_TALK/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/00_AI4GW_TALK/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/00_AI4GW_TALK/HOW_TO_GET_RESULTS/configs/mass_premerger.yaml

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

echo NULL