#!/bin/bash
# Registers all sig_only sweeps with WandB and submits one SLURM worker per sweep.
# Usage: bash scripts/launch_sig_only_sweeps.sh [wandb_entity]
#
# Logs SLURM job IDs, sweep IDs, and variable names to:
#   outputs/slurm_logs/sig_only_sweep_jobs.log

ENTITY=${1:-kyoon-mit-massachusetts-institute-of-technology}

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg
SWEEP_DIR=${TOP_DIR}/sweeps/sig_only
SLURM_SCRIPT=${TOP_DIR}/slurm/parameter_estimation/s4d_gaussnll_sweep_worker.slurm
LOG_FILE=${TOP_DIR}/outputs/slurm_logs/sig_only_sweep_jobs.log

mkdir -p "$(dirname "$LOG_FILE")"

VARS=(Mc q m1 m2 a1 a2 distance)

cd ${TOP_DIR}

echo "=== sig_only sweep launch: $(date) ===" | tee -a "$LOG_FILE"

for VAR in "${VARS[@]}"; do
  SWEEP_FILE="${SWEEP_DIR}/${VAR}_arch_sweep.yaml"

  # Register sweep
  WANDB_OUTPUT=$(wandb sweep --entity "$ENTITY" "$SWEEP_FILE" 2>&1)
  SWEEP_ID=$(echo "$WANDB_OUTPUT" | grep -oP '(?<=sweep with ID: )\S+' | head -1)

  if [[ -z "$SWEEP_ID" ]]; then
    echo "ERROR: could not register sweep for ${VAR}" | tee -a "$LOG_FILE"
    echo "wandb output: $WANDB_OUTPUT" | tee -a "$LOG_FILE"
    continue
  fi

  # Construct full sweep path required by wandb agent
  PROJECT=$(grep '^project:' "$SWEEP_FILE" | awk '{print $2}')
  FULL_SWEEP_ID="${ENTITY}/${PROJECT}/${SWEEP_ID}"

  # Submit SLURM worker
  SBATCH_OUTPUT=$(sbatch --export=SWEEP_ID="$FULL_SWEEP_ID",VAR="$VAR" "$SLURM_SCRIPT")
  JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+')

  echo "VAR=${VAR}  SWEEP_ID=${FULL_SWEEP_ID}  SLURM_JOB=${JOB_ID}" | tee -a "$LOG_FILE"
done

echo "=== done. check: squeue -u $(whoami) ===" | tee -a "$LOG_FILE"
