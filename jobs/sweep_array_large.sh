#!/bin/bash -l
#
# Large-run array script for TinyGPU A100.
# Each task gets isolated WORK paths to prevent concurrent write collisions.
#
# Submit with:
#   sbatch jobs/sweep_array_large.sh
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=sc_ae_larray
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
#SBATCH --array=0-11
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail

module load python
conda activate scvi_env 2>/dev/null || echo "WARN: conda env scvi_env not found, using user packages"

# Uncomment and set if your compute nodes require a proxy for internet access:
# export http_proxy=http://your.proxy.address:port
# export https_proxy=http://your.proxy.address:port

PROJECT_HOME="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_HOME}"

if [[ -z "${WORK:-}" ]]; then
  export WORK="${HOME}/work"
  echo "WARN: WORK not set; falling back to ${WORK}"
fi

if [[ -n "${SCAE_DIMS_CSV:-}" ]]; then
  IFS=',' read -r -a DIMS <<< "${SCAE_DIMS_CSV}"
else
  DIMS=(2 4 6 8 10 12 16 20 24 32 48 64)
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#DIMS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for DIMS length ${#DIMS[@]}"
  exit 2
fi

CURRENT_DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}

TASK_ROOT="job_${SLURM_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}"
export SCAE_WORK_ROOT="${WORK}/sc_autoencoder_project"
export SCAE_PERSIST_ROOT="${SCAE_WORK_ROOT}/array_runs/${TASK_ROOT}"

export SCAE_RESULTS_DIR="${SCAE_PERSIST_ROOT}/results"
export SCAE_LOG_DIR="${SCAE_PERSIST_ROOT}/logs"
export SCAE_CHECKPOINT_DIR="${SCAE_PERSIST_ROOT}/checkpoints"

STAGE_ROOT="${TMPDIR:-/tmp}/${USER}/scae_array_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export SCAE_DATA_DIR="${STAGE_ROOT}/data"

mkdir -p logs "${SCAE_RESULTS_DIR}" "${SCAE_LOG_DIR}" "${SCAE_CHECKPOINT_DIR}" "${SCAE_DATA_DIR}"
mkdir -p "${SCAE_WORK_ROOT}/data"

echo "=== LARGE ARRAY TASK START ==="
echo "Job:            ${SLURM_JOB_ID}"
echo "Task:           ${SLURM_ARRAY_TASK_ID}"
echo "Latent dim:     ${CURRENT_DIM}"
echo "Persistent root:${SCAE_PERSIST_ROOT}"
echo "Stage root:     ${STAGE_ROOT}"
echo "Start:          $(date)"

# Stage-in data for this task
if [[ -d "${SCAE_WORK_ROOT}/data" ]]; then
  rsync -a --delete "${SCAE_WORK_ROOT}/data/" "${SCAE_DATA_DIR}/"
fi

nvidia-smi --query-gpu=gpu_name,memory.total,compute_cap --format=csv,noheader || true

SCAE_CONFIG="${SCAE_CONFIG:-config/large_scale.yaml}"
echo "Experiment config: ${SCAE_CONFIG}"

python scripts/run_experiment.py \
  --config "${SCAE_CONFIG}" \
  --latent_dims "${CURRENT_DIM}" \
  --device auto \
  --output "${SCAE_RESULTS_DIR}/experiment_results_all.csv" \
  2>&1 | tee -a "${SCAE_LOG_DIR}/run_experiment.log"

# Stage-out processed cache updates
if [[ -d "${SCAE_DATA_DIR}/processed" ]]; then
  mkdir -p "${SCAE_WORK_ROOT}/data/processed"
  rsync -a "${SCAE_DATA_DIR}/processed/" "${SCAE_WORK_ROOT}/data/processed/"
fi

echo "Task complete: $(date)"
echo "Task results: ${SCAE_RESULTS_DIR}/experiment_results_all.csv"

echo "--- Post-array merge placeholder (run after all tasks complete) ---"
echo "python scripts/merge_array_results.py --input_root ${SCAE_WORK_ROOT}/array_runs/job_${SLURM_JOB_ID} --output ${SCAE_WORK_ROOT}/array_runs/job_${SLURM_JOB_ID}/merged_experiment_results.csv"
