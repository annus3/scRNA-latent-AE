# scRNA Autoencoder Latent-Dimension Study

This project asks one main question:

**How should latent size (`d`) change as class complexity (`K`) changes in single-cell RNA-seq?**

Here, `K` means the number of label classes used for evaluation in a dataset.

The pipeline runs AE, VAE, and scVI experiments, records metrics, and builds candidate `d = f(K)` summaries.

## Project Description

This is a fixed-config benchmark project for single-cell representation learning.

The work has two connected parts:

- **Scientific part:** test whether latent size can be explained by label complexity (`K`) in a practical way.
- **Engineering part:** build a stable HPC pipeline that can run from small datasets to atlas-scale datasets.

What this project is designed to do:

- compare AE, VAE, and scVI under one consistent protocol
- keep PRIMARY evidence strict (trusted real labels only)
- keep synthetic/inferred-label evidence separate as exploratory
- produce reproducible output tables that can be re-audited in report-only mode

What this project is not trying to do right now:

- full per-dataset hyperparameter optimization
- claiming one universal law that works perfectly for every dataset

### Current Scope and Main Findings

Using the current cleaned PRIMARY evidence for `scvi:nb`, the recommended latent dimensions are:

| Dataset | `K` | `d` |
|---|---|---|
| `scvi_pbmc12k` | 09  | 04 |
| `paul15` | 19 | 02 |
| `aifi_immune_full` | 29 | 04 |
| `ts2_lung` | 34 | 12 |
| `ts1_all_cells` | 177| 10 |

Formula-fit summary for PRIMARY `scvi:nb` is moderate (not perfect), and leave-one-dataset-out rows are available (`status=ok`).

So the current picture is useful but still conservative: behavior looks dataset-dependent, with larger atlases often favoring a modest higher latent band.

## Datasets Used in This Study

The project uses a mix of small/medium benchmarks and large real atlases.

### Trusted real datasets (PRIMARY candidates)

| Dataset | Typical use | Label key | Batch key | Trust role | Notes |
|---|---|---|---|---|---|
| `scvi_pbmc12k` | core real benchmark | `labels` | `batch` | ground_truth | curated scvi benchmark PBMC set |
| `paul15` | core real benchmark | `paul15_clusters` | none | ground_truth | myeloid progenitor benchmark |
| `ts1_all_cells` | large atlas phase | `cell_type` | `donor_id` | ground_truth (via onboarding contract) | Tabula Sapiens all-cells processed contract |
| `aifi_immune_full` | large atlas phase | `cell_type` | `batch_id` | ground_truth (via onboarding contract) | Allen Immune atlas, large-scale stress test |
| `ts2_lung` | large non-blood atlas phase | `cell_type` | `donor_id` | strict provenance-gated | enters PRIMARY only with explicit ontology-backed provenance evidence |

### Exploratory / non-primary datasets

| Dataset group | Why used | PRIMARY role |
|---|---|---|
| `pbmc3k` | smoke and pipeline checks | exploratory / utility |
| `pbmc68k_reduced` | quick small benchmark | untrusted labels |
| `splatter_k*` synthetic sets | controlled scaling experiments | exploratory only |

## Scientific Policy (Current)

I keep a strict split:

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
python scripts/run_experiment.py 
  --config config/default.yaml 
  --datasets pbmc3k,paul15
```

### 4) Generate reports from an existing CSV (report-only mode)

```bash
python scripts/run_experiment.py 
  --config config/default.yaml 
  --report_only_csv results/global/tables/combined_curated_real_plus_phase4_enriched.csv 
  --auto_d_report
```

### 5) Audit PRIMARY inclusion from a CSV

```bash
python scripts/audit_primary_pool.py 
  --config config/default.yaml 
  --csv results/global/tables/combined_curated_real_plus_phase4_enriched.csv
```

## HPC Run Pattern

For large jobs on FAU TinyGPU/A100, use job scripts in `jobs/`.

Example:

```bash
cd /home/hpc/user/sc_autoencoder_project
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
cd /home/hpc/user/sc_autoencoder_project
bash code_analysis/analysis.sh
```

Each analysis run creates a new timestamped folder:

`results/analysis_runs/<YYYYmmddTHHMMSS>/`


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

If you are new to the project: run smoke test first, then one small sweep, then `code_analysis/analysis.sh`.
