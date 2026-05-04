#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=0-02:00:00
#SBATCH --job-name=inj_test
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

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/ai4gw@cern/s4d_gaussnll_snr_5_50/chirp_mass_60-64s_d32_s32_l4.yaml
ckpt_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs/ai4gw@cern__s4d_gaussnll_snr_5_50/chirp_mass_60-64s_d32_s32_l4/checkpoints/s4d_gaussnll_var0mseloss_ckpt_epoch=386.ckpt

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"
echo "Using checkpoint:  $ckpt_path"

cd ${TOP_DIR}

# ── Test step (runs PlotParamEstCallback → saves CSV + 4 PNGs in .../plots/) ──
srun --cpu-bind=none python -m BNSReg.core.main test \
    --config "$config_path" \
    --ckpt_path "$ckpt_path"

# ── Training step (commented out as example) ──────────────────────────────────
# srun --cpu-bind=none python -m BNSReg.core.main fit \
#     --config "$config_path"
