#!/bin/bash
#SBATCH --job-name=phase1_pbmc3k
#SBATCH --output=logs/phase1_pbmc3k_%j.out
#SBATCH --error=logs/phase1_pbmc3k_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

cd "${HOME}/sc_autoencoder_project"
export NUMBA_DISABLE_COVERAGE=1
mkdir -p results/global/tables logs

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
echo "SLURM_STEP_ID=${SLURM_STEP_ID:-<none>}"

python3 scripts/run_experiment.py \
  --config config/phase1_pbmc3k_quick.yaml \
  --output results/global/tables/phase1_pbmc3k_quick_results.csv
