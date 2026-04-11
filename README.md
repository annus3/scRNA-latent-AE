# scRNA Autoencoder Latent-Dimension Study

**How should latent size (`d`) change as class complexity (`K`) changes in single-cell RNA-seq?**

This project benchmarks AE, VAE, and scVI models on curated scRNA-seq datasets to study whether the optimal latent dimension can be predicted from the number of cell types. It uses a trust-aware evaluation framework that strictly separates ground-truth evidence from exploratory results.

## Main Findings

Using curated evidence from scVI with negative binomial likelihood:

| Dataset | K | Recommended d | ARI |
|---|---|---|---|
| scvi_pbmc12k | 9 | 4 | 0.666 |
| paul15 | 19 | 2 | 0.404 |
| aifi_immune_full | 29 | 4 | 0.342 |
| ts2_lung | 34 | 12 | 0.430 |
| ts1_all_cells | 177 | 10 | 0.270 |

No universal **d = f(K)** scaling law was found. Instead, the data suggest dataset-dependent narrow bands: compact latent spaces (d=2-4) work for moderate-complexity datasets, while atlas-scale data favors d=10-12.

## Repository Layout

```text
sc_autoencoder_project/
├── src/                   # Core package (models, training, evaluation, data)
│   ├── data/              #   Dataset loading, preprocessing, splitting
│   ├── models/            #   AE, VAE, scVI wrapper, loss functions
│   ├── training/          #   Training loop with early stopping, TensorBoard
│   ├── evaluation/        #   Clustering, reconstruction, batch metrics
│   └── utils/             #   Logging and tensor conversion helpers
├── scripts/               # CLI entry points
│   ├── run_experiment.py      # Main experiment orchestrator
│   ├── train.py               # Single model training
│   ├── smoke_test.py          # Pipeline validation
│   ├── preprocess.py          # Dataset preprocessing
│   ├── validate_pipeline.py   # End-to-end pipeline checks
│   ├── onboard_contract_h5ad.py  # Large-atlas onboarding
│   └── ...                    # Audit, inspection, re-evaluation tools
├── code_analysis/         # Post-hoc analysis pipeline (15 modules)
│   ├── common.py              # Shared plotting/loading utilities
│   ├── a00_inventory.py       # Data readiness audit
│   ├── a02_primary_d_vs_k.py  # Core d=f(K) analysis
│   ├── a14_publication_figures.py  # Publication-quality figures
│   └── ...                    # Cross-model, batch, stability, ELBO analyses
├── config/                # YAML experiment configurations
├── jobs/                  # SLURM job scripts for HPC execution
├── data/                  # Datasets (not tracked, downloaded on demand)
├── results/               # Experiment outputs (selectively tracked)
│   ├── global/tables/         # Core evidence & formula-fit CSVs
│   ├── datasets/              # Per-dataset figures, latents, tables
│   ├── analysis/              # Early analysis pipeline output
│   ├── analysis_runs/         # Timestamped analysis snapshots
│   └── code_analysis_runs/    # 15-module analysis pipeline output
├── checkpoints/           # Model weights (not tracked)
└── logs/                  # Training logs (not tracked)
```

## Setup

```bash
# Clone and enter the repository
git clone https://github.com/annus3/scRNA-latent-AE.git
cd scRNA-latent-AE

# Create and activate environment
python -m venv ~/envs/scae
source ~/envs/scae/bin/activate

# Install package and all dependencies
pip install -e .
```

**Requirements:** Python >= 3.10, PyTorch >= 2.0, scvi-tools >= 1.0, Scanpy >= 1.9

## Usage

### 1. Smoke test

```bash
python scripts/smoke_test.py --with_scvi
```

### 2. Preprocess a dataset

```bash
python scripts/preprocess.py --config config/default.yaml --dataset pbmc3k
```

### 3. Run an experiment sweep

```bash
python scripts/run_experiment.py \
  --config config/default.yaml \
  --datasets pbmc3k,paul15
```

### 4. Generate reports from existing results

```bash
python scripts/run_experiment.py \
  --config config/default.yaml \
  --report_only_csv results/global/tables/combined_curated_real_plus_phase4_enriched.csv \
  --auto_d_report
```

### 5. Run the analysis pipeline

```bash
bash code_analysis/analysis.sh
```

Each analysis run creates a timestamped output folder under `results/code_analysis_runs/`.

## Experimental Phases

The project was developed in **four sequential phases**, each building on the previous:

| Phase | Dataset(s) | Purpose | Config | Job | Est. Time |
|---|---|---|---|---|---|
| 1 | pbmc3k | Pipeline baseline validation | `phase1_pbmc3k_quick.yaml` | `phase1_pbmc3k_quick.sbatch` | ~1h |
| 2 | scvi_pbmc12k | Medium real-data benchmark (batch effects) | `phase2_scvi_pbmc12k.yaml` | `phase2_scvi_pbmc12k.sbatch` | ~2h |
| 3 | splatter_k04/k08/k12 | Controlled simulation with known K | `phase3_splatter_k_sweep.yaml` | `phase3_full_sweep.sh` | ~12h |
| 4 | AIFI, TS1, TS2-Lung | Atlas-scale validation on large curated datasets | `phase4_*.yaml` | `phase4_*_onboard_light.sh` → `phase4_*_reduced.sh` | ~12h each |

**Phase 4 prerequisites:** Large-atlas datasets must be downloaded and placed under `$WORK/sc_autoencoder_project/data/raw/` before running. Run the onboarding script first to prepare each dataset, then the sweep:

```bash
# Example for AIFI atlas
sbatch jobs/phase4_aifi_onboard_light.sh
sbatch jobs/phase4_aifi_scvi_nb_reduced.sh
```

## HPC Execution

For large-scale runs on SLURM clusters, use the job scripts in `jobs/`:

```bash
# Validate the pipeline first (GPU, ~10 min)
sbatch jobs/smoke_test.sh

# Generic latent-dimension sweep (adapt partition/modules for your cluster)
sbatch jobs/slurm_sweep_generic.sh

# Parallel array sweep (12 tasks, one per latent dim)
sbatch jobs/sweep_array.sh

# Phase-specific runs (run in order)
sbatch jobs/phase1_pbmc3k_quick.sbatch
sbatch jobs/phase2_scvi_pbmc12k.sbatch
sbatch jobs/phase3_full_sweep.sh
sbatch jobs/phase4_aifi_scvi_nb_reduced.sh
```

Job scripts handle data staging via `$WORK` and `$TMPDIR` environment variables. Phase 4 configs reference `${SCAE_DATA_DIR}` and `${SCAE_RESULTS_DIR}` — these are exported automatically by the job scripts. If running outside SLURM, set them manually before invoking the phase configs.

## Datasets

### Curated (ground-truth labels)

| Dataset | Cells | K | Phase | Source |
|---|---|---|---|---|
| scvi_pbmc12k | 12,039 | 9 | 2 | scvi-tools built-in |
| paul15 | 2,730 | 19 | 2 | Scanpy built-in |
| aifi_immune_full | ~1.8M | 29 | 4 | Allen Institute for Immunology |
| ts2_lung | ~100K | 34 | 4 | Tabula Sapiens |
| ts1_all_cells | ~483K | 177 | 4 | Tabula Sapiens |

### Exploratory

| Dataset | Cells | K | Phase | Source |
|---|---|---|---|---|
| pbmc3k | 2,700 | — | 1 | Scanpy built-in |
| splatter_k04/k08/k12 | ~5K each | 4/8/12 | 3 | Splatter (synthetic) |

## Models & Losses

| Model | Loss Functions | Batch Modeling |
|---|---|---|
| AE (autoencoder) | MSE, NB, ZINB | No |
| VAE (variational AE) | MSE, NB, ZINB | No |
| scVI | NB, ZINB | Yes (conditional) |

## Trust & Evidence Policy

The project enforces a strict split between trusted and exploratory evidence:

- **Curated**: datasets with verified ground-truth cell-type labels. All quantitative claims use only curated evidence.
- **Exploratory**: synthetic data or datasets with inferred labels. Used for pipeline validation only.

Trust is enforced programmatically — external metrics (ARI, AMI) are only computed when labels pass the trust gate.

## Results

The `results/` directory is selectively tracked — only final outputs are included; raw tensorboard exports and intermediate runs are gitignored.

### Key Output Tables

All tables are written to `results/global/tables/`:

| File | Description |
|---|---|
| `combined_curated_real_plus_phase4_enriched.csv` | Full evidence table with trust metadata |
| `recommended_d_by_K_primary.csv` | Recommended d per dataset (curated only) |
| `formula_fit_summary_primary.csv` | d=f(K) curve fitting results |
| `formula_fit_loo_primary.csv` | Leave-one-out cross-validation |
| `phase1_pbmc3k_quick_results.csv` | Phase 1 baseline results |
| `phase2_scvi_pbmc12k_results.csv` | Phase 2 benchmark results |
| `phase3_splatter_full_completed.csv` | Phase 3 simulation sweep results |

### Analysis Pipeline Output

The `code_analysis/` pipeline produces a timestamped run under `results/code_analysis_runs/` with 15 analysis modules:

| Module | Content |
|---|---|
| `00_inventory` | Data readiness audit |
| `01_dataset_profiles` | Dataset scale comparison figures |
| `02_primary_d_vs_k` | Core d=f(K) analysis with formula fits |
| `03_cross_model` | AE vs VAE vs scVI comparison |
| `04_per_dataset` | Per-dataset ARI curves |
| `05_batch_effects` | Batch correction analysis |
| `06_splatter_validation` | Synthetic data validation |
| `07_significance` | Statistical significance tests |
| `08_training_dynamics` | Training convergence analysis |
| `09_elbo_analysis` | ELBO decomposition (VAE/scVI) |
| `10_centrality_topology` | Latent space topology |
| `11_seed_stability` | Cross-seed reproducibility |
| `12_metric_correlation` | Metric agreement analysis |
| `13_phase_comparison` | Cross-phase consistency |
| `14_publication_figures` | Publication-quality figures |

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:

> Mohammad Annus, "Trust-Aware Latent Dimension Selection for Single-Cell RNA-seq Autoencoders," Biomedical Network Science Lab (BioNets), FAU Erlangen-Nürnberg, 2026.
