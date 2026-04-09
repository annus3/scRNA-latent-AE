#!/bin/bash -l
#
# Phase 4 Stage 0: lightweight one-time onboarding for ts2_lung dataset.
#
# Canonical ready-file contract:
#   - obs["cell_type"] from curated ontology-like labels (default: cell_ontology_class)
#   - obs["donor_id"] as canonical batch key
#   - layers["counts"] as canonical counts layer
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=ts2_lung_onboard
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
TS2_SOURCE_FILE="${TS2_SOURCE_FILE:-${WORK_ROOT}/data/raw/ts2_lung_source.h5ad}"
TS2_READY_FILE="${TS2_READY_FILE:-${WORK_ROOT}/data/processed/ts2_lung.h5ad}"

if [[ ! -f "${TS2_SOURCE_FILE}" ]]; then
  echo "ERROR: Missing source TS2-Lung file: ${TS2_SOURCE_FILE}"
  exit 2
fi

mkdir -p "$(dirname "${TS2_READY_FILE}")"

STAGE_ROOT="${TMPDIR:-/tmp}/${USER}/scae_ts2_lung_onboard_${SLURM_JOB_ID}"
mkdir -p "${STAGE_ROOT}"
TS2_STAGE_READY="${STAGE_ROOT}/ts2_lung.h5ad"

export TS2_READY_FILE

echo "=== TS2-LUNG LIGHTWEIGHT ONBOARD START ==="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Source file:   ${TS2_SOURCE_FILE}"
echo "Ready file:    ${TS2_READY_FILE}"
echo "Staged output: ${TS2_STAGE_READY}"
echo "Start:         $(date)"

python3 scripts/onboard_contract_h5ad.py \
  --source "${TS2_SOURCE_FILE}" \
  --out "${TS2_STAGE_READY}" \
  --label-primary cell_type \
  --label-fallback "" \
  --batch-primary donor_id \
  --batch-fallback donor,batch,batch_id \
  --batch-canonical donor_id \
  --counts-source-policy x_only \
  --counts-transform round \
  --skip-filter-hvg \
  --skip-normalize-log1p

rsync -a "${TS2_STAGE_READY}" "${TS2_READY_FILE}"

python3 - <<"PY"
import os
import anndata as ad

path = os.environ["TS2_READY_FILE"]
a = ad.read_h5ad(path, backed="r")
missing = [k for k in ("cell_type", "donor_id") if k not in a.obs.columns]
if missing:
    raise SystemExit(f"Missing required obs keys in ready file: {missing}")
if "counts" not in a.layers:
    raise SystemExit("Missing layers['counts'] in ready file")
if a.layers["counts"].shape != a.X.shape:
    raise SystemExit(
        f"counts/X shape mismatch in ready file: {a.layers['counts'].shape} vs {a.X.shape}"
    )
cell_type_source = str(a.uns.get("cell_type_source", ""))
onboard = a.uns.get("onboard_contract", {})
source_label_key = str(onboard.get("source_label_key", "")) if isinstance(onboard, dict) else ""
contract_label_key = str(onboard.get("label_key", "")) if isinstance(onboard, dict) else ""
source_label_ontology_key = str(onboard.get("source_label_ontology_key", "")) if isinstance(onboard, dict) else ""
source_label_ontology_nonnull_fraction = onboard.get("source_label_ontology_nonnull_fraction", float("nan")) if isinstance(onboard, dict) else float("nan")
try:
    source_label_ontology_nonnull_fraction = float(source_label_ontology_nonnull_fraction)
except Exception:
    source_label_ontology_nonnull_fraction = float("nan")

if cell_type_source != "provided:cell_type":
    raise SystemExit(
        f"Strict TS2 provenance check failed: cell_type_source={cell_type_source!r} "
        "(expected 'provided:cell_type')"
    )
if not (
    source_label_key == "cell_type"
    and contract_label_key == "cell_type"
    and source_label_ontology_key == "cell_type_ontology_term_id"
    and source_label_ontology_nonnull_fraction >= 0.99
):
    raise SystemExit(
        "Strict TS2 provenance check failed: onboard_contract must record "
        "source_label_key='cell_type', label_key='cell_type', "
        "source_label_ontology_key='cell_type_ontology_term_id' with >=0.99 non-null coverage. "
        f"Got source_label_key={source_label_key!r}, label_key={contract_label_key!r}, "
        f"source_label_ontology_key={source_label_ontology_key!r}, "
        f"source_label_ontology_nonnull_fraction={source_label_ontology_nonnull_fraction!r}"
    )
print("ready_file_ok shape=", a.shape)
print("cell_type_source=", cell_type_source)
print("batch_source=", a.uns.get("batch_source", ""))
print("onboard_contract.source_label_key=", source_label_key)
print("onboard_contract.label_key=", contract_label_key)
print("onboard_contract.source_label_ontology_key=", source_label_ontology_key)
print("onboard_contract.source_label_ontology_nonnull_fraction=", source_label_ontology_nonnull_fraction)
PY

echo "Ready file written: ${TS2_READY_FILE}"
ls -lh "${TS2_READY_FILE}"
echo "End: $(date)"
echo "=== TS2-LUNG LIGHTWEIGHT ONBOARD DONE ==="
