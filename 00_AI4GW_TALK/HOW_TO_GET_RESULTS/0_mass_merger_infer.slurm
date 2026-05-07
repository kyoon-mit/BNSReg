#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=20:00
#SBATCH --job-name=mass_merger_sig
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

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/00_AI4GW_TALK/HOW_TO_GET_RESULTS/configs/mass_merger.yaml

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
python -m BNSReg.core.main_fit_test test --config "$config_path" \
    --trainer.logger.init_args.project 00_AI4GW_TALK \
    --trainer.logger.init_args.id mass_merger_sig \
    --trainer.logger.init_args.save_dir /n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/00_AI4GW_TALK/ZZ_misc_outputs \
    --data.init_args.data_cfg.injected_data_key whitened_injected \
    --data.init_args.data_cfg.num_workers 1 \
    --data.init_args.data_cfg.prefetch_factor 1 \
    --ckpt_path /n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/00_AI4GW_TALK/mass_merger/chirp_mass_ft_FIX_snr_5_50_59-63s_d64_s64_l4/checkpoints/s4d_gaussnll_var0mse_ckpt_epoch=1177.ckpt