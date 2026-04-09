#!/bin/bash -l
#
# Phase 4: AIFI first-pass reduced sweep (scvi:nb only).
#
# Submit from project root with:
#   sbatch jobs/phase4_aifi_scvi_nb_reduced.sh
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=phase4_aifi_reduced
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail

module load python
conda activate scvi_env 2>/dev/null || echo "WARN: conda env scvi_env not found, using user packages"

export NUMBA_DISABLE_COVERAGE=1

PROJECT_HOME="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_HOME}"
mkdir -p logs

if [[ -z "${WORK:-}" ]]; then
  echo "ERROR: WORK is not set. Expected AIFI dataset under \$WORK/sc_autoencoder_project/data/processed/."
  exit 2
fi

export SCAE_WORK_ROOT="${WORK}/sc_autoencoder_project"
export SCAE_PERSIST_ROOT="${SCAE_WORK_ROOT}/phase4/aifi_reduced/job_${SLURM_JOB_ID}"

export SCAE_RESULTS_DIR="${SCAE_PERSIST_ROOT}/results"
export SCAE_LOG_DIR="${SCAE_PERSIST_ROOT}/logs"
export SCAE_CHECKPOINT_DIR="${SCAE_PERSIST_ROOT}/checkpoints"

STAGE_ROOT="${TMPDIR:-/tmp}/${USER}/scae_phase4_aifi_${SLURM_JOB_ID}"
export SCAE_DATA_DIR="${STAGE_ROOT}/data"

mkdir -p "${SCAE_RESULTS_DIR}" "${SCAE_LOG_DIR}" "${SCAE_CHECKPOINT_DIR}" "${SCAE_DATA_DIR}/processed"

AIFI_WORK_FILE="${AIFI_WORK_FILE:-${SCAE_WORK_ROOT}/data/processed/aifi_immune_full.h5ad}"
AIFI_STAGE_FILE="${SCAE_DATA_DIR}/processed/aifi_immune_full.h5ad"

if [[ ! -f "${AIFI_WORK_FILE}" ]]; then
  echo "ERROR: Missing AIFI processed file: ${AIFI_WORK_FILE}"
  exit 2
fi

echo "=== PHASE 4 AIFI REDUCED START ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_JOB_NODELIST}"
echo "WORK file:    ${AIFI_WORK_FILE}"
echo "Stage file:   ${AIFI_STAGE_FILE}"
echo "Persist root: ${SCAE_PERSIST_ROOT}"
echo "Start:        $(date)"

# Stage AIFI artifact from WORK -> TMPDIR for local IO
rsync -a "${AIFI_WORK_FILE}" "${AIFI_STAGE_FILE}"

# Quick schema gate before training
python3 - <<PY
import os
import anndata as ad

path = os.environ["SCAE_DATA_DIR"] + "/processed/aifi_immune_full.h5ad"
a = ad.read_h5ad(path, backed="r")
missing = [k for k in ("cell_type", "batch_id") if k not in a.obs.columns]
if missing:
    raise SystemExit(f"Missing required obs keys for AIFI run: {missing}")
if "counts" not in a.layers:
    raise SystemExit("Missing required layers['counts'] for NB/scVI path")
print("AIFI schema gate passed")
PY

SCAE_CONFIG="${SCAE_CONFIG:-config/phase4_aifi_scvi_nb_reduced.yaml}"

echo "Using config: ${SCAE_CONFIG}"

python scripts/run_experiment.py \
  --config "${SCAE_CONFIG}" \
  --device auto \
  --output "${SCAE_RESULTS_DIR}/phase4_aifi_scvi_nb_reduced.csv" \
  2>&1 | tee -a "${SCAE_LOG_DIR}/run_experiment.log"

echo "End: $(date)"
echo "=== PHASE 4 AIFI REDUCED DONE ==="
