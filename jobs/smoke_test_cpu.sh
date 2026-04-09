#!/bin/bash -l
#
# Smoke test: CPU-only (no GPU wait, faster queue)
# Submit from sc_autoencoder_project/ with: sbatch jobs/smoke_test_cpu.sh
#
#SBATCH --time=0:10:00
#SBATCH --job-name=sc_smoke_cpu
#SBATCH --output=logs/slurm_smoke_%j.out
#SBATCH --error=logs/slurm_smoke_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate scvi_env

# Uncomment and set if your compute nodes require a proxy for internet access:
# export http_proxy=http://your.proxy.address:port
# export https_proxy=http://your.proxy.address:port

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "=== Smoke Test (CPU) ==="
echo "Node: $SLURM_JOB_NODELIST"
echo "Start: $(date)"
echo "---"

# AE + VAE only (no scVI to avoid Lightning GPU detection issues)
python scripts/smoke_test.py --tensorboard

echo "---"
echo "End: $(date)"
