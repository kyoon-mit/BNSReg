#!/bin/bash
#SBATCH --partition=gpu_requeue
#SBATCH --time=20:00
#SBATCH --job-name=s_merger_bkg
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

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/00_AI4GW_TALK/HOW_TO_GET_RESULTS/configs/sky_merger.yaml

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"

cd ${TOP_DIR}
python -m BNSReg.core.main_fit_test test --config "$config_path" \
    --trainer.logger.init_args.project 00_AI4GW_TALK \
    --trainer.logger.init_args.id sky_merger_bkg \
    --trainer.logger.init_args.save_dir /n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/00_AI4GW_TALK/ZZ_misc_outputs \
    --data.init_args.data_cfg.injected_data_key whitened_bkg \
    --data.init_args.data_cfg.test_file /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/checks/bns_snr_8_50_uniform_4096Hz_100K_5s/test/sig_combined_test.h5 \
    --data.init_args.data_cfg.num_workers 1 \
    --data.init_args.data_cfg.prefetch_factor 1 \
    --ckpt_path /n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/00_AI4GW_TALK/sky_merger/sky_ft_ckpt2_SNR_5_50_62-63s_d64_s64_l4/checkpoints/s4d_sky_angErr_ckpt_epoch=3202.ckpt