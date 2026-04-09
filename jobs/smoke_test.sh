#!/bin/bash -l
#
# Smoke test: Quick pipeline validation (~2 min)
# Submit from sc_autoencoder_project/ with: sbatch jobs/smoke_test.sh
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=0:10:00
#SBATCH --job-name=sc_smoke
#SBATCH --output=logs/slurm_smoke_%j.out
#SBATCH --error=logs/slurm_smoke_%j.out
#SBATCH --export=NONE

# Merge stderr into stdout (^^^ same file for --error)

unset SLURM_EXPORT_ENV

module load python
conda activate scvi_env

# Uncomment and set if your compute nodes require a proxy for internet access:
# export http_proxy=http://your.proxy.address:port
# export https_proxy=http://your.proxy.address:port

mkdir -p logs
cd $SLURM_SUBMIT_DIR

echo "=== Smoke Test ==="
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU:"
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
echo "Start: $(date)"
echo "---"

# Run smoke test with all model types including scVI + TensorBoard
python scripts/smoke_test.py --with_scvi --tensorboard 2>&1

echo "---"
echo "TensorBoard files:"
find logs/tensorboard/smoke_test -name "events.out.tfevents*" -exec ls -lh {} \; 2>/dev/null || echo "  No TB files found"
echo "Checkpoint files:"
ls -lh checkpoints/smoke_test/*.pt 2>/dev/null || echo "  No checkpoint files found"
echo "---"
echo "End: $(date)"
