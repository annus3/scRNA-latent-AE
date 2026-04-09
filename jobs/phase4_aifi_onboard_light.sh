#!/bin/bash -l
#
# Phase 4 Stage 0: lightweight one-time onboarding for AIFI full atlas.
#
# Canonical ready-file contract:
#   - obs["cell_type"] from trusted AIFI label column (default: AIFI_L2)
#   - obs["batch_id"] as canonical batch key
#   - layers["counts"] as canonical counts layer
#
# This script intentionally avoids heavy preprocessing steps for atlas-scale input.
#
# Submit from project root with:
#   sbatch jobs/phase4_aifi_onboard_light.sh
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=aifi_onboard
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
  echo "ERROR: WORK is not set"
  exit 2
fi

WORK_ROOT="${WORK}/sc_autoencoder_project"
AIFI_SOURCE_FILE="${AIFI_SOURCE_FILE:-${WORK_ROOT}/data/raw/aifi_immune_full_source.h5ad}"
AIFI_READY_FILE="${AIFI_READY_FILE:-${WORK_ROOT}/data/processed/aifi_immune_full.h5ad}"

if [[ ! -f "${AIFI_SOURCE_FILE}" ]]; then
  echo "ERROR: Missing source AIFI file: ${AIFI_SOURCE_FILE}"
  exit 2
fi

mkdir -p "$(dirname "${AIFI_READY_FILE}")"

STAGE_ROOT="${TMPDIR:-/tmp}/${USER}/scae_aifi_onboard_${SLURM_JOB_ID}"
mkdir -p "${STAGE_ROOT}"
AIFI_STAGE_READY="${STAGE_ROOT}/aifi_immune_full.h5ad"

export AIFI_READY_FILE

echo "=== AIFI LIGHTWEIGHT ONBOARD START ==="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Source file:   ${AIFI_SOURCE_FILE}"
echo "Ready file:    ${AIFI_READY_FILE}"
echo "Staged output: ${AIFI_STAGE_READY}"
echo "Start:         $(date)"

python3 scripts/onboard_contract_h5ad.py \
  --source "${AIFI_SOURCE_FILE}" \
  --out "${AIFI_STAGE_READY}" \
  --label-primary AIFI_L2 \
  --label-fallback AIFI_L3,AIFI_L1 \
  --batch-primary batch_id \
  --batch-fallback sample_id,donor_id,batch \
  --batch-canonical batch_id \
  --counts-source-policy x_only \
  --counts-transform round \
  --skip-filter-hvg \
  --skip-normalize-log1p

rsync -a "${AIFI_STAGE_READY}" "${AIFI_READY_FILE}"

python3 - <<"PY"
import os
import anndata as ad

path = os.environ["AIFI_READY_FILE"]
a = ad.read_h5ad(path, backed="r")
missing = [k for k in ("cell_type", "batch_id") if k not in a.obs.columns]
if missing:
    raise SystemExit(f"Missing required obs keys in ready file: {missing}")
if "counts" not in a.layers:
    raise SystemExit("Missing layers['counts'] in ready file")
if a.layers["counts"].shape != a.X.shape:
    raise SystemExit(
        f"counts/X shape mismatch in ready file: {a.layers['counts'].shape} vs {a.X.shape}"
    )
print("ready_file_ok shape=", a.shape)
print("cell_type_source=", a.uns.get("cell_type_source", ""))
print("batch_source=", a.uns.get("batch_source", ""))
PY

echo "Ready file written: ${AIFI_READY_FILE}"
ls -lh "${AIFI_READY_FILE}"
echo "End: $(date)"
echo "=== AIFI LIGHTWEIGHT ONBOARD DONE ==="
