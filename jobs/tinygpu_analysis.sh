#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=sc_ae_analysis
#SBATCH --output=logs/slurm_analysis_%x_%j.out
#SBATCH --error=logs/slurm_analysis_%x_%j.out
#SBATCH --export=NONE
# tinygpu_analysis.sh — code_analysis pipeline on TinyGPU A100
# Submit:  sbatch jobs/tinygpu_analysis.sh
# Monitor: tail -f logs/slurm_analysis_*.out
#
# Optional env-var overrides (pass with --export=ALL,VAR=val):
#   ANALYSIS_OUT   override output root directory
#   ANALYSIS_ONLY  run only these IDs, comma-separated  e.g. "02,14"
#   ANALYSIS_SKIP  skip these IDs, comma-separated      e.g. "06,07"

unset SLURM_EXPORT_ENV
set -euo pipefail

module load python
conda activate scvi_env 2>/dev/null || echo "WARN: scvi_env not found, using pip packages"
# Uncomment and set if your compute nodes require a proxy for internet access:
# export http_proxy=http://your.proxy.address:port
# export https_proxy=http://your.proxy.address:port
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true

PROJECT_HOME="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_HOME}"

[[ -z "${WORK:-}" ]] && export WORK="${HOME}/work" && echo "WARN: WORK not set, using ${WORK}"
SCAE_WORK="${WORK}/sc_autoencoder_project"
TS="$(date +%Y%m%dT%H%M%S)"
OUT_ROOT="${ANALYSIS_OUT:-${PROJECT_HOME}/results/code_analysis_runs/${TS}}"
mkdir -p logs "${OUT_ROOT}"

echo "=== sc_ae Analysis Pipeline ==="
echo "    job=${SLURM_JOB_ID}  node=${SLURM_JOB_NODELIST}"
echo "    start=$(date)"
echo "    repo=${PROJECT_HOME}"
echo "    output=${OUT_ROOT}"
echo "    ONLY=${ANALYSIS_ONLY:-all}  SKIP=${ANALYSIS_SKIP:-none}"
echo "    Python: $(python3 --version 2>&1)"

# ── Pre-flight checks ────────────────────────────────────────
MAIN="${PROJECT_HOME}/results/global/tables/combined_curated_real_plus_phase4_enriched.csv"
if [[ ! -f "${MAIN}" ]]; then
    echo "ERROR: Main evidence CSV not found: ${MAIN}"
    echo "       Run the experiment pipeline first, then resubmit."
    exit 1
fi
echo "[preflight] Main evidence CSV: OK"

for f in recommended_d_by_K_primary.csv formula_fit_summary_primary.csv \
          formula_fit_loo_primary.csv phase3_splatter_full_completed.csv; do
    fp="${PROJECT_HOME}/results/global/tables/${f}"
    [[ -f "${fp}" ]] \
        && echo "[preflight] ${f}: OK" \
        || echo "[preflight] WARN: ${f} missing — some scripts may warn"
done

# Log GPU for audit trail (analysis is CPU-only)
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader 2>/dev/null || true

# ── Launch ───────────────────────────────────────────────────
ARGS=(--output-root "${OUT_ROOT}")
[[ -n "${ANALYSIS_SKIP:-}" ]] && ARGS+=(--skip "${ANALYSIS_SKIP}")
[[ -n "${ANALYSIS_ONLY:-}" ]] && ARGS+=(--only "${ANALYSIS_ONLY}")

echo "[analysis] Launching code_analysis/analysis.sh ..."
bash "${PROJECT_HOME}/code_analysis/analysis.sh" "${ARGS[@]}"

# ── Post-run inventory ────────────────────────────────────────
echo ""
echo "=== Post-run inventory ==="
printf "%-40s  %6s  %6s\n" "Subfolder" "Figs" "Tables"
find "${OUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | while IFS= read -r d; do
    nf=$(find "${d}/figures" -type f 2>/dev/null | wc -l)
    nt=$(find "${d}/tables"  -type f 2>/dev/null | wc -l)
    printf "%-40s  %6d  %6d\n" "$(basename "${d}")" "${nf}" "${nt}"
done
echo "PNGs+PDFs : $(find "${OUT_ROOT}" \( -name '*.png' -o -name '*.pdf' \) 2>/dev/null | wc -l)"
echo "CSVs      : $(find "${OUT_ROOT}" -name '*.csv' 2>/dev/null | wc -l)"
echo "LaTeX     : $(find "${OUT_ROOT}" -name '*.tex' 2>/dev/null | wc -l)"

# ── Sync to WORK ─────────────────────────────────────────────
if [[ -d "${SCAE_WORK}" ]]; then
    mkdir -p "${SCAE_WORK}/analysis_runs"
    rsync -a "${OUT_ROOT}/" "${SCAE_WORK}/analysis_runs/${TS}/"
    echo "[sync] done -> ${SCAE_WORK}/analysis_runs/${TS}/"
else
    echo "WARN: ${SCAE_WORK} not found, skipping WORK sync"
fi

echo "=== Complete: $(date)"
echo "=== Output: ${OUT_ROOT}"
