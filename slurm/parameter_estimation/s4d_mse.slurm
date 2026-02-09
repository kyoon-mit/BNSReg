#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=12:00:00
#SBATCH --job-name=s4d
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

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/s4d_mse_fasrc_slurm.yaml

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
srun --cpu-bind=none python -m BNSReg.core.main fit \
    --config "$config_path" \
    --trainer.logger.init_args.id ${SLURM_JOB_ID}