#!/bin/bash
#SBATCH --job-name=phase2_pbmc12k
#SBATCH --output=logs/phase2_pbmc12k_%j.out
#SBATCH --error=logs/phase2_pbmc12k_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

cd "${HOME}/sc_autoencoder_project"
export NUMBA_DISABLE_COVERAGE=1
mkdir -p results/global/tables logs

python3 scripts/run_experiment.py \
  --config config/phase2_scvi_pbmc12k.yaml \
  --output results/global/tables/phase2_scvi_pbmc12k_results.csv
