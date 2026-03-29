# Latent Dimension Space in Single-Cell RNA Auto-Encoders

This repository contains a research-grade, highly modular, and HPC-ready Python pipeline designed to systematically evaluate how the optimal latent dimensionality ($d$) of an autoencoder scales with the biological complexity ($K$ cell classes) in single-cell RNA-seq datasets.

The pipeline parallelizes the training of models across multiple datasets to generate empirical data, enabling the derivation of the mathematical formula $d = f(K)$.

---

## Project Architecture

The project is structured as a standard, modular Python package to support distributed execution: 

```text
sc_autoencoder_project/
├── config/                  # YAML configurations (hyperparameters, grid sweep)
├── data/                    # Datasets (raw & processed .h5ad)
├── jobs/                    # SLURM scripts for FAU NHR clusters
├── logs/                    # TensorBoard and script execution logs
├── results/                 # Metrics CSVs, recommendation tables, and figures
├── scripts/                 # CLI entry points (train, sweep, preprocess)
├── src/                     # Core Python modules
│   ├── config.py            # Dataclass config loader
│   ├── data/                # Scanpy preprocessing & PyTorch DataLoaders
│   ├── evaluation/          # Metrics (ARI, Silhouette, Centrality Variance)
│   ├── models/              # AE, VAE, scVI wrapper, and specialized losses
│   ├── training/            # PyTorch training loop, Early Stopping, Memory profiling
│   └── utils/               # Loggers and helper functions
├── requirements.txt         # Deep learning & downstream analysis dependencies
└── README.md                # This file
```

---

## Quick Start & Installation

### 1. Environment Setup
I recommend setting up a virtual environment or conda environment: 
```bash
python -m venv ~/envs/sc_ae_env
source ~/envs/sc_ae_env/bin/activate
cd sc_autoencoder_project

# Install the repository as an editable package
pip install -e .

# Install all dependencies (including analysis tools like Jupyter/UMAP)
pip install -r requirements.txt
```

### 2. Verify the Pipeline (Smoke Test)
Before running massive sweeps on the GPU nodes, verify the pipeline integrity locally on the CPU using a tiny, generated synthetic dataset:
```bash
python scripts/smoke_test.py --with_scvi
```

---

## Dataset Tier Strategy

To structurally manage computational load across the HPC clusters, the pipeline groups datasets into tiers. You can run sweeps on specific tiers using `--dataset_tier`.

- **Tier A (Smoke Test)**: `pbmc3k` (Small, local, excellent for verifying code correctness before submitting to GPU nodes).
- **Tier B (Core Analysis)**: `pbmc3k`, `paul15` (Medium size, standard benchmarks for validating scRNA-seq embedding quality).
- **Tier C (Phase-2 Scale)**: Curated massive `cellxgene` datasets. These represent the ultimate goal of the project for large-scale, final validation runs taking advantage of the multi-GPU clusters.

*Note: Avoid using Scanpy's built-in `pbmc68k_reduced` for `nb` or `zinb` likelihood testing, as it comes preprocessed without raw counts.*

---

## CLI Scripts & Workflows

Instead of running notebooks sequentially, interact with the pipeline via the highly configurable CLI scripts. All scripts are driven by `config/default.yaml`.

### 1. Preprocessing (`scripts/preprocess.py`)
Downloads (if necessary) and runs the standard Scanpy pipeline (normalization, log1p, highly variable genes). Importantly, it **keeps raw counts intact** for count-based likelihood models (`nb` and `zinb`).
```bash
python scripts/preprocess.py --config config/default.yaml --dataset pbmc3k
```

### 2. Single Model Training (`scripts/train.py`)
Trains a single model specification. Great for debugging a specific architectural choice or loss function.
```bash
python scripts/train.py \
  --config config/default.yaml \
  --dataset pbmc3k \
  --model ae \
  --latent_dim 16 \
  --loss_type mse \
  --device cuda
```

### 3. Grid Sweep (`scripts/run_experiment.py`)
The primary experiment orchestrator. It sweeps across specified datasets, neural network architectures (`ae`, `vae`, `scvi`), loss functions (`mse`, `nb`, `zinb`), random seeds, and latent dimensions.
```bash
python scripts/run_experiment.py \
  --config config/default.yaml \
  --datasets pbmc3k,paul15 \
  --latent_dims 2,4,8,16,32,64 \
  --auto_d_report
```

---

## Outputs & Logging

The pipeline saves highly structured outputs to the `results/` directory, preventing data overwrites.

### Tables & Recommendations
*   **`experiment_results_all.csv`**: Massive table containing every metric (ARI, Silhouette, Reconstruction Loss, Peak GPU memory, Runtime) for every individual model run.
*   **`recommended_d_by_K.csv`**: Generated when running sweeps with `--auto_d_report`. It automatically strips out the noise and ranks the absolute "best" latent dimension for each dataset size.
*   **`formula_fit_summary.csv`**: Attempts to fit equations (Linear, Power Law, Sqrt, Log) mapping $K \rightarrow d$.

### TensorBoard Monitoring
Run `tensorboard --logdir logs/tensorboard` to monitor training loops live. The pipeline tracks:
*   Train and Validation Losses (Reconstruction, KL-Divergence).
*   Learning Rate scheduling.
*   Gradient Norm tracking (to debug exploding gradients in ZINB loss).
*   Peak Memory profiling (CPU & GPU).

---

## HPC & SLURM Usage

For users running on High-Performance Computing (HPC) clusters, the pipeline is designed to be storage-aware:

1. **Codebase**: Clone the repository into your `$HOME` directory.
2. **Data & Results**: Large `.h5ad` datasets and uncompressed CSV results can quickly exceed `$HOME` storage quotas. I recommend updating `config/default.yaml` to point `paths.data_dir` to your `$WORK` or scratch directory.
3. **Array Jobs**: The `jobs/` directory contains a template SLURM script for submitting parallel batch operations to GPU partitions (e.g., `sbatch jobs/slurm_sweep_generic.sh`). Users should modify this file to fit their cluster's partition architecture and python module/conda systems.

---

## References & Literature

- **`scvi-tools` framework**: [Documentation](https://docs.scvi-tools.org/en/1.3.3/index.html)
- **Scanpy guidelines**: [Documentation](https://scanpy.readthedocs.io/en/stable/)
- **scRNA-seq best practices**: [sc-best-practices.org](https://www.sc-best-practices.org/introduction/scrna_seq.html)
- **Datasets**: [CELLxGENE Census](https://cellxgene.cziscience.com/)
