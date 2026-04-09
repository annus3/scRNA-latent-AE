#!/bin/bash -l
#
# Phase 3 full sweep run for splatter_k04/k08/k12 using config/phase3_splatter_k_sweep.yaml
# Submit from project root with: sbatch jobs/phase3_full_sweep.sh
#
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=phase3_full_sweep
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results/global/tables data/processed

if [[ -z "${WORK:-}" ]]; then
  echo "ERROR: WORK is not set. Expected Splatter files under \$WORK/sc_autoencoder_project/data/processed/."
  exit 2
fi

for K in 04 08 12; do
  WORK_FILE="${WORK}/sc_autoencoder_project/data/processed/splatter_k${K}.h5ad"
  REPO_FILE="data/processed/splatter_k${K}.h5ad"

  if [[ ! -f "${WORK_FILE}" ]]; then
    echo "ERROR: Missing Splatter file: ${WORK_FILE}"
    exit 2
  fi

  if [[ ! -e "${REPO_FILE}" ]]; then
    ln -s "${WORK_FILE}" "${REPO_FILE}"
    echo "Linked ${REPO_FILE} -> ${WORK_FILE}"
  elif [[ -L "${REPO_FILE}" ]]; then
    CURRENT_TARGET="$(readlink "${REPO_FILE}")"
    if [[ "${CURRENT_TARGET}" != "${WORK_FILE}" ]]; then
      rm -f "${REPO_FILE}"
      ln -s "${WORK_FILE}" "${REPO_FILE}"
      echo "Updated symlink ${REPO_FILE} -> ${WORK_FILE}"
    fi
  else
    rm -f "${REPO_FILE}"
    ln -s "${WORK_FILE}" "${REPO_FILE}"
    echo "Replaced local file with symlink ${REPO_FILE} -> ${WORK_FILE}"
  fi
done

echo "=== PHASE3 FULL SWEEP START ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      ${SLURM_JOB_NODELIST}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Start:     $(date)"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  python scripts/run_experiment.py \
    --config config/phase3_splatter_k_sweep.yaml \
    --device auto \
    --output results/global/tables/phase3_splatter_full.csv

echo "End: $(date)"
echo "=== PHASE3 FULL SWEEP DONE ==="
