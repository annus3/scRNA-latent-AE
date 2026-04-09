#!/bin/bash
# 
# Generic SLURM Job Array Script for sc_autoencoder_project
# 
# Usage: sbatch jobs/slurm_sweep_generic.sh
# 
# This script is designed for the community to adapt it to their own cluster.
# It parallelizes the latent-dimensionality sweep across multiple compute nodes.
# 
#SBATCH --job-name=sc_ae_sweep
#SBATCH --partition=gpu          # <= Change to your cluster's GPU partition
#SBATCH --gres=gpu:1             # <= Request 1 GPU per task
#SBATCH --time=2:00:00           # <= Adjust based on dataset size
#SBATCH --output=logs/slurm_%x_%A_%a.out 
#SBATCH --error=logs/slurm_%x_%A_%a.err 
#SBATCH --array=0-11             # Array indices matching the DIMS length

# Environment Setup (Modify as needed for your cluster)

# 1. Load missing modules (e.g., CUDA, Python)
# module load python cuda

# 2. Activate your virtual environment
# source ~/envs/sc_ae_env/bin/activate
# OR
# conda activate sc_ae_env

# 3. Setup internet proxy if compute nodes do not have internet access
# export http_proxy=http://your.proxy.address:port
# export https_proxy=http://your.proxy.address:port

export PYTHONUNBUFFERED=1

echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Starting Time: $(date)"
echo "============================================================================"

# Move to the project root directory
cd $SLURM_SUBMIT_DIR

# Ensure output directories exist
mkdir -p logs results checkpoints

# Print GPU info
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

# Hyperparameter Grid Definition
# List of latent dimensions to sweep over (length must match SLURM array indexing)
DIMS=(2 4 6 8 10 12 16 20 24 32 48 64)

# Map current SLURM_ARRAY_TASK_ID to a latent dimension
CURRENT_DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}

echo "Starting training sweep for latent_dim=$CURRENT_DIM"

# Execute the runner script
python scripts/run_experiment.py \
    --config config/default.yaml \
    --datasets pbmc3k,paul15 \
    --latent_dims $CURRENT_DIM \
    --model_types ae,vae,scvi \
    --device auto \
    --profile_memory

echo "Task $SLURM_ARRAY_TASK_ID (d=$CURRENT_DIM) completed at $(date)"
