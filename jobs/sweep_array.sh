#!/bin/bash -l
#
# SLURM Job Array: Parallel sweep over latent dimensions
# Each array task trains one latent_dim value across all model types.
#
# Submit from sc_autoencoder_project/ with: sbatch jobs/sweep_array.sh
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --job-name=sc_ae_sweep
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
#SBATCH --array=0-11
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate scvi_env

# Uncomment and set if your compute nodes require a proxy for internet access:
# export http_proxy=http://your.proxy.address:port
# export https_proxy=http://your.proxy.address:port

mkdir -p logs results checkpoints

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------
DIMS=(2 4 6 8 10 12 16 20 24 32 48 64)

# Map SLURM_ARRAY_TASK_ID to a latent dimension
CURRENT_DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Training with latent_dim=$CURRENT_DIM"
echo "Start: $(date)"

nvidia-smi

# Each task trains all model types for one latent dimension
python scripts/run_experiment.py \
    --config config/default.yaml \
    --latent_dims $CURRENT_DIM \
    --datasets pbmc3k \
    --model_types ae,vae \
    --device auto \
    --profile_memory \
    --output results/sweep_d${CURRENT_DIM}.csv

echo "Task $SLURM_ARRAY_TASK_ID (d=$CURRENT_DIM) completed at $(date)"
