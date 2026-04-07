# scRNA Autoencoder Latent-Dimension Study

This project asks one main question:

**How should latent size (`d`) change as class complexity (`K`) changes in single-cell RNA-seq?**

Here, `K` means the number of label classes used for evaluation in a dataset.

The pipeline runs AE, VAE, and scVI experiments, records metrics, and builds candidate `d = f(K)` summaries.

## What This Project Is

This is a **fixed-config benchmark study**.

It is not a full hyperparameter search per dataset.  
So results should be read as controlled comparisons under one protocol.

## Scientific Policy (Current)

We keep a strict split:

- **PRIMARY evidence** = trusted real datasets with ground-truth label provenance
- **EXPLORATORY evidence** = inferred-label or synthetic datasets

Trust is enforced in code during experiment/report generation.  
PRIMARY tables only include rows that pass the trust gate.

## Repository Layout

```text
sc_autoencoder_project/
├── config/        # YAML configs (default + phase configs)
├── data/          # raw + processed datasets
├── docs/          # plans, audits, status reports
├── jobs/          # SLURM scripts for HPC
├── logs/          # runtime logs + TensorBoard events
├── code_analysis/ # additional script-based analysis modules
├── results/       # experiment tables, reports, analysis outputs
├── scripts/       # CLI entry points
└── src/           # package code (data, models, training, evaluation)
```

## Quick Setup

```bash
python -m venv ~/envs/scae
source ~/envs/scae/bin/activate

cd /home/hpc/iwbn/iwbn129h/sc_autoencoder_project
pip install -e .
pip install -r requirements.txt
```

## Core Commands

### 1) Smoke test

```bash
python scripts/smoke_test.py --with_scvi
```

### 2) Preprocess one dataset

```bash
python scripts/preprocess.py --config config/default.yaml --dataset pbmc3k
```

### 3) Run a sweep

```bash
python scripts/run_experiment.py \
  --config config/default.yaml \
  --datasets pbmc3k,paul15
```

### 4) Generate reports from an existing CSV (report-only mode)

```bash
python scripts/run_experiment.py \
  --config config/default.yaml \
  --report_only_csv results/global/tables/combined_curated_real_plus_phase4_enriched.csv \
  --auto_d_report
```

### 5) Audit PRIMARY inclusion from a CSV

```bash
python scripts/audit_primary_pool.py \
  --config config/default.yaml \
  --csv results/global/tables/combined_curated_real_plus_phase4_enriched.csv
```

## HPC Run Pattern

For large jobs on FAU TinyGPU/A100, use job scripts in `jobs/`.

Example:

```bash
cd /home/hpc/iwbn/iwbn129h/sc_autoencoder_project
sbatch.tinygpu jobs/tinygpu_large.sh
```

`jobs/tinygpu_large.sh` handles path staging with:

- `$WORK` for persistent run outputs
- `$TMPDIR` for local staged data
- `SCAE_*` environment variables for phase configs that use `${SCAE_DATA_DIR}` style paths

If you run those phase configs directly (outside job scripts), set the required `SCAE_*` env vars first.

## Analysis Workflow (Script-First on HPC)

On HPC, use script mode instead of interactive notebooks:

```bash
cd /home/hpc/iwbn/iwbn129h/sc_autoencoder_project
bash notebooks/analysis.sh
```

Each analysis run creates a new timestamped folder:

`results/analysis_runs/<YYYYmmddTHHMMSS>/`

So old analysis outputs are preserved.

## Main Output Tables

Global merged evidence:

- `results/global/tables/combined_curated_real_plus_phase4_enriched.csv`

PRIMARY reports:

- `results/global/tables/recommended_d_by_K_primary.csv`
- `results/global/tables/formula_fit_summary_primary.csv`
- `results/global/tables/formula_fit_loo_primary.csv`

Exploratory reports (if enabled):

- `results/global/tables/recommended_d_by_K_exploratory.csv`
- `results/global/tables/formula_fit_summary_exploratory.csv`

## Notes on Loss Scale

NB/ZINB loss values can look large (hundreds or thousands).  
That is normal because they are count-likelihood losses, not MSE.

What to watch instead:

- stable train/val trend
- no NaN/inf
- downstream metrics and report outputs

The analysis scripts also export per-gene normalized NB/ZINB loss summaries for fairer comparison across datasets.

## Useful Phase Configs

- `config/phase1_pbmc3k_quick.yaml`
- `config/phase2_scvi_pbmc12k.yaml`
- `config/phase3_splatter_k_sweep.yaml`
- `config/phase4_ts1_stageA_tiny.yaml`
- `config/phase4_ts1_stageB_reduced.yaml`
- `config/phase4_aifi_scvi_nb_reduced.yaml`
- `config/phase4_ts2_lung_scvi_nb_reduced.yaml`

## Quick Troubleshooting

1. Schema checks:  
   `python scripts/inspect_dataset_schema.py --help`
2. PRIMARY trust/status checks:  
   `python scripts/audit_primary_pool.py --help`
3. Run-level errors:  
   `logs/slurm_*.out`, `logs/slurm_*.err`
4. Run manifests:  
   `results/runs/run_*/manifest.json`
5. Analysis inventories:  
   `results/analysis_runs/<timestamp>/00_inventory/`

---

If you are new to the project: run smoke test first, then one small sweep, then `notebooks/analysis.sh`.
