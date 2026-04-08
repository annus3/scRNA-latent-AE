#!/usr/bin/env bash
# ============================================================
# analysis.sh — Master runner for the code_analysis pipeline
#
# Usage:
#   cd /path/to/sc_autoencoder_project
#   bash code_analysis/analysis.sh [--output-root /path/to/dir] [--skip 06,07]
#
# Each run writes to a fresh timestamped directory under:
#   results/code_analysis_runs/<YYYYmmddTHHMMSS>/
#
# Flags:
#   --output-root   Override the default timestamped output directory
#   --skip          Comma-separated list of script IDs to skip (e.g. 06,07)
#   --only          Comma-separated list of script IDs to run exclusively
#   --help          Show this help
# ============================================================
set -euo pipefail

# ------------------------------------
# Defaults
# ------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%dT%H%M%S)"
OUT_ROOT="${REPO_ROOT}/results/code_analysis_runs/${TS}"
SKIP_IDS=""
ONLY_IDS=""
PYTHON="${PYTHON:-python3}"

# ------------------------------------
# Parse arguments
# ------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root)
            OUT_ROOT="$2"; shift 2 ;;
        --skip)
            SKIP_IDS="$2"; shift 2 ;;
        --only)
            ONLY_IDS="$2"; shift 2 ;;
        --help|-h)
            grep '^#' "$0" | head -20 | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "[analysis] Unknown argument: $1"; exit 1 ;;
    esac
done

# ------------------------------------
# Setup
# ------------------------------------
mkdir -p "${OUT_ROOT}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
LOG_FILE="${OUT_ROOT}/analysis_run.log"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "  sc_autoencoder_project — Analysis Pipeline"             | tee -a "${LOG_FILE}"
echo "  Timestamp : ${TS}"                                       | tee -a "${LOG_FILE}"
echo "  Repo      : ${REPO_ROOT}"                               | tee -a "${LOG_FILE}"
echo "  Output    : ${OUT_ROOT}"                                 | tee -a "${LOG_FILE}"
echo "  Python    : $(${PYTHON} --version 2>&1)"               | tee -a "${LOG_FILE}"
echo "  WORK      : ${WORK:-<not set>}"                         | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

# ------------------------------------
# Helper: should_run <id>
# ------------------------------------
should_run() {
    local id="$1"
    # If --only is set, run only those
    if [[ -n "${ONLY_IDS}" ]]; then
        IFS=',' read -ra only_arr <<< "${ONLY_IDS}"
        for o in "${only_arr[@]}"; do
            if [[ "${o}" == "${id}" ]]; then return 0; fi
        done
        return 1
    fi
    # If --skip is set, skip those
    if [[ -n "${SKIP_IDS}" ]]; then
        IFS=',' read -ra skip_arr <<< "${SKIP_IDS}"
        for s in "${skip_arr[@]}"; do
            if [[ "${s}" == "${id}" ]]; then return 1; fi
        done
    fi
    return 0
}

# ------------------------------------
# Helper: run_script <id> <script> [extra_args...]
# ------------------------------------
run_script() {
    local id="$1"
    local script="$2"
    shift 2
    local extra_args=("$@")

    if ! should_run "${id}"; then
        echo "[analysis] SKIP  ${id} (${script})" | tee -a "${LOG_FILE}"
        return 0
    fi

    echo "" | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo "[analysis] START ${id} — ${script} $(date '+%H:%M:%S')"       | tee -a "${LOG_FILE}"
    echo "------------------------------------------------------------" | tee -a "${LOG_FILE}"

    local t_start
    t_start=$(date +%s)

    if ${PYTHON} "code_analysis/${script}" \
        --output-root "${OUT_ROOT}" \
        "${extra_args[@]}" \
        2>&1 | tee -a "${LOG_FILE}"; then
        local t_end
        t_end=$(date +%s)
        echo "[analysis] DONE  ${id} — $((t_end - t_start))s" | tee -a "${LOG_FILE}"
    else
        local t_end
        t_end=$(date +%s)
        echo "[analysis] ERROR ${id} — $((t_end - t_start))s (see log above)" | tee -a "${LOG_FILE}"
        # Do NOT abort the full run — continue to next script
    fi
}

# ============================================================
# Master execution sequence
# ============================================================

t_global_start=$(date +%s)

# 00 — Inventory & Readiness (always run first)
run_script "00" "a00_inventory.py"

# 01 — Per-dataset profiles
run_script "01" "a01_dataset_profiles.py"

# 02 — Primary d=f(K) analysis
run_script "02" "a02_primary_d_vs_k.py"

# 03 — Cross-model comparison
run_script "03" "a03_cross_model.py"

# 04 — Per-dataset architecture deep-dive
run_script "04" "a04_per_dataset.py"

# 05 — Batch effect analysis
run_script "05" "a05_batch_effects.py"

# 06 — Splatter K-sweep validation
run_script "06" "a06_splatter_validation.py"

# 07 — Statistical significance tests
run_script "07" "a07_statistical_significance.py"

# 08 — Training dynamics & convergence
run_script "08" "a08_training_dynamics.py"

# 09 — ELBO & reconstruction decomposition
run_script "09" "a09_elbo_analysis.py"

# 10 — Centrality & latent topology
run_script "10" "a10_centrality_topology.py"

# 11 — Seed stability & reproducibility
run_script "11" "a11_seed_stability.py"

# 12 — Metric correlation analysis
run_script "12" "a12_metric_correlation.py"

# 13 — Phase comparison & scale analysis
run_script "13" "a13_phase_comparison.py"

# 14 — Publication-quality figures (run last)
run_script "14" "a14_publication_figures.py"

# ============================================================
# Summary
# ============================================================
t_global_end=$(date +%s)
t_total=$((t_global_end - t_global_start))

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "[analysis] COMPLETE — total time: ${t_total}s"              | tee -a "${LOG_FILE}"
echo "[analysis] Artifacts: ${OUT_ROOT}"                           | tee -a "${LOG_FILE}"
echo "[analysis] Log:       ${LOG_FILE}"                           | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

# Write a short manifest
cat > "${OUT_ROOT}/analysis_manifest.json" <<EOF
{
  "timestamp": "${TS}",
  "repo_root": "${REPO_ROOT}",
  "output_root": "${OUT_ROOT}",
  "total_time_s": ${t_total},
  "python": "$(${PYTHON} --version 2>&1)",
  "skip_ids": "${SKIP_IDS}",
  "only_ids": "${ONLY_IDS}"
}
EOF

echo "[analysis] Done."
