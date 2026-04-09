#!/bin/bash
#SBATCH --job-name=phase3_k04
#SBATCH --output=logs/phase3_k04_%j.out
#SBATCH --error=logs/phase3_k04_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

cd "${HOME}/sc_autoencoder_project"
export NUMBA_DISABLE_COVERAGE=1
mkdir -p results/global/tables logs

python3 scripts/run_experiment.py \
  --config config/phase3_splatter_k_sweep.yaml \
  --datasets splatter_k04 \
  --loss_matrix ae:nb,vae:nb,scvi:nb \
  --latent_dims 2,8,16 \
  --output results/global/tables/phase3_splatter_k04_pilot.csv
