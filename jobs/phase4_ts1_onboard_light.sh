#!/bin/bash -l
#
# Phase 4 Stage 0: lightweight one-time onboarding for TS1
# Option B canonical contract:
#   - obs['cell_type'] from obs['cell_ontology_class']
#   - obs['donor_id'] as canonical batch key
#   - layers['counts'] canonical count layer
#
# This script intentionally performs only minimal prep:
#   filtering + HVG + normalize/log1p
# and skips PCA/neighbors/clustering.
#
# Submit from project root with:
#   sbatch jobs/phase4_ts1_onboard_light.sh
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=ts1_onboard
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
TS1_SOURCE_FILE="${TS1_SOURCE_FILE:-${WORK_ROOT}/data/processed/ts1_all_cells.h5ad}"
TS1_READY_FILE="${TS1_READY_FILE:-${WORK_ROOT}/data/processed/ts1_all_cells_phase4_ready.h5ad}"

# Minimal preprocessing knobs (override via env if needed)
TS1_MIN_GENES_PER_CELL="${TS1_MIN_GENES_PER_CELL:-200}"
TS1_MIN_CELLS_PER_GENE="${TS1_MIN_CELLS_PER_GENE:-3}"
TS1_N_TOP_GENES="${TS1_N_TOP_GENES:-2000}"
TS1_TARGET_SUM="${TS1_TARGET_SUM:-10000}"

if [[ ! -f "${TS1_SOURCE_FILE}" ]]; then
  echo "ERROR: Missing source TS1 file: ${TS1_SOURCE_FILE}"
  exit 2
fi

mkdir -p "$(dirname "${TS1_READY_FILE}")"

STAGE_ROOT="${TMPDIR:-/tmp}/${USER}/scae_ts1_onboard_${SLURM_JOB_ID}"
mkdir -p "${STAGE_ROOT}"
TS1_STAGE_READY="${STAGE_ROOT}/ts1_all_cells_phase4_ready.h5ad"

export TS1_SOURCE_FILE TS1_READY_FILE TS1_STAGE_READY
export TS1_MIN_GENES_PER_CELL TS1_MIN_CELLS_PER_GENE TS1_N_TOP_GENES TS1_TARGET_SUM

echo "=== TS1 LIGHTWEIGHT ONBOARD START ==="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Source file:   ${TS1_SOURCE_FILE}"
echo "Ready file:    ${TS1_READY_FILE}"
echo "Staged output: ${TS1_STAGE_READY}"
echo "Preprocess:    min_genes/cell=${TS1_MIN_GENES_PER_CELL}, min_cells/gene=${TS1_MIN_CELLS_PER_GENE}, n_top_genes=${TS1_N_TOP_GENES}, target_sum=${TS1_TARGET_SUM}"
echo "Start:         $(date)"

python3 - <<'PY'
import os
import gc
import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

src = os.environ["TS1_SOURCE_FILE"]
out = os.environ["TS1_STAGE_READY"]
min_genes = int(os.environ["TS1_MIN_GENES_PER_CELL"])
min_cells = int(os.environ["TS1_MIN_CELLS_PER_GENE"])
n_top_genes = int(os.environ["TS1_N_TOP_GENES"])
target_sum = float(os.environ["TS1_TARGET_SUM"])

src_adata = ad.read_h5ad(src)

missing = [k for k in ("cell_ontology_class", "donor_id") if k not in src_adata.obs.columns]
if missing:
    raise SystemExit(f"Missing required source obs columns: {missing}")

# Build working matrix from raw.X when available (preferred counts source), otherwise X.
if src_adata.raw is not None:
    X_counts = src_adata.raw.X.copy()
    var = src_adata.raw.var.copy()
    counts_source = "raw.X"
else:
    X_counts = src_adata.X.copy()
    var = src_adata.var.copy()
    counts_source = "X"

obs = src_adata.obs.copy()
adata = ad.AnnData(X=X_counts, obs=obs, var=var)

# Free source object early to reduce peak memory pressure.
del src_adata, X_counts, obs, var
gc.collect()

# Canonical Option B keys
adata.obs["cell_type"] = adata.obs["cell_ontology_class"].astype(str).astype("category")
adata.obs["donor_id"] = adata.obs["donor_id"].astype(str).astype("category")
# Optional mirror only
adata.obs["batch"] = adata.obs["donor_id"].copy()

# Minimal filtering + HVG only (no PCA/neighbors/clustering)
sc.pp.filter_cells(adata, min_genes=min_genes)
sc.pp.filter_genes(adata, min_cells=min_cells)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=n_top_genes,
    flavor="seurat_v3",
    subset=True,
)

# counts layer is created *after* filtering/HVG subsetting so shapes are guaranteed aligned
if sp.issparse(adata.X):
    counts = adata.X.tocsr(copy=True)
    if counts.data.size:
        np.clip(counts.data, 0, None, out=counts.data)
        counts.data = np.rint(counts.data)
        counts.data = counts.data.astype(np.float32, copy=False)
else:
    counts = np.asarray(adata.X)
    counts = np.clip(counts, 0, None)
    counts = np.rint(counts).astype(np.float32, copy=False)
adata.layers["counts"] = counts

# Controlled pipeline-friendly X
sc.pp.normalize_total(adata, target_sum=target_sum)
sc.pp.log1p(adata)

# Hard shape compatibility checks
if adata.layers["counts"].shape != adata.X.shape:
    raise SystemExit(
        f"Shape mismatch after onboarding: counts={adata.layers['counts'].shape}, X={adata.X.shape}"
    )

adata.uns["cell_type_source"] = "provided:cell_ontology_class"
adata.uns["batch_source"] = "provided:donor_id"
adata.uns["phase4_ready_contract"] = {
    "label_key": "cell_type",
    "batch_key": "donor_id",
    "counts_layer": "counts",
    "counts_source": counts_source,
    "minimal_steps": ["filter_cells", "filter_genes", "hvg_subset", "normalize_total", "log1p"],
    "created_by": "jobs/phase4_ts1_onboard_light.sh",
}

adata.write_h5ad(out, compression="lzf")

# Post-write quick contract sanity
b = ad.read_h5ad(out, backed="r")
req_obs = ["cell_type", "donor_id"]
miss_obs = [k for k in req_obs if k not in b.obs.columns]
if miss_obs:
    raise SystemExit(f"Ready file missing required obs keys: {miss_obs}")
if "counts" not in b.layers:
    raise SystemExit("Ready file missing layers['counts']")
if b.layers["counts"].shape != b.X.shape:
    raise SystemExit(
        f"Ready file shape mismatch: counts={b.layers['counts'].shape}, X={b.X.shape}"
    )

print("Ready file contract check passed")
print(f"shape={b.shape}")
print(f"counts_source={counts_source}")
print(f"obs_keys_present={req_obs}")
print("counts_layer_present=True")
print("counts_shape_matches_X=True")
PY

rsync -a "${TS1_STAGE_READY}" "${TS1_READY_FILE}"

echo "Ready file written: ${TS1_READY_FILE}"
ls -lh "${TS1_READY_FILE}"
echo "End: $(date)"
echo "=== TS1 LIGHTWEIGHT ONBOARD DONE ==="
