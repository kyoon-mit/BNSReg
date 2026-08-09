#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --time=12:00:00
#SBATCH --job-name=s4d_chirp_premerger_snr_8_50
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/checks/s4d_gaussnll_snr_8_50/chirp_mass/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/slurm_logs/checks/s4d_gaussnll_snr_8_50/chirp_mass/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm_cuda312

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/checks/snr_8_20/s4d_chirp_mass_51-55s_d512_s32_l4.yaml

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
srun --cpu_bind=none python -m BNSReg.core.main_fit_test fit --config "$config_path" \
--trainer.logger.init_args.id chirp_mass_ckpt2_51-55s_d512_s32_l4 \
--trainer.num_nodes 2 \
--ckpt_path /n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/00_AI4GW_TALK/mass_premerger/chirp_mass_ckpt_51-55s_d512_s32_l4/checkpoints/s4d_gaussnll_var0sigma_ckpt_epoch=310.ckpt