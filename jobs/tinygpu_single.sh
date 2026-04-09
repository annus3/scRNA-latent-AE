#!/bin/bash -l

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=6:00:00
#SBATCH --job-name=sc_ae_exp
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
#SBATCH --export=NONE

# Merge stderr into stdout for unified logging
unset SLURM_EXPORT_ENV

echo "=== EXPERIMENT START ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start:     $(date)"
echo "---"

# Load modules
module load python

# Increase file descriptor limit (TensorBoard + DataLoader need many FDs)
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# Activate conda environment (continue if not found — pip user install fallback)
conda activate scvi_env 2>/dev/null || echo "WARN: conda env 'scvi_env' not found, using pip user packages"

# Uncomment and set if your compute nodes require a proxy for internet access:
# export http_proxy=http://your.proxy.address:port
# export https_proxy=http://your.proxy.address:port

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Ensure directories exist
mkdir -p logs results checkpoints data/raw data/processed

# GPU info
nvidia-smi --query-gpu=gpu_name,memory.total,compute_cap --format=csv,noheader
echo "---"

# Run the full experiment sweep
python scripts/run_experiment.py \
    --config config/default.yaml \
    --device auto \
    --profile_memory \
    --auto_d_report \
    2>&1

echo "---"
echo "=== OUTPUT FILES ==="
echo "Results:"
find results -name "*.csv" -exec ls -lh {} \; 2>/dev/null
echo ""
echo "TensorBoard:"
find logs/tensorboard -name "events.out.tfevents*" 2>/dev/null | wc -l
echo " event files in logs/tensorboard/"
echo ""
echo "Checkpoints:"
find checkpoints -name "*.pt" 2>/dev/null | wc -l
echo " checkpoint files"
echo "---"
echo "End: $(date)"
