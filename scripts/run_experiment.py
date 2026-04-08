#!/usr/bin/env python3
"""CLI: Full experiment sweep over latent dimensions."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import make_dataloaders, split_adata
from src.data.loader import (
    get_label_trust,
    list_datasets_by_tiers,
    load_dataset,
    resolve_batch_key,
    resolve_label_key,
)
from src.data.preprocessor import (
    check_cached_preprocess_fingerprint,
    load_processed,
    preprocess,
    save_processed,
)
from src.evaluation.metrics import (
    compute_centrality_variance,
    compute_reconstruction_loss,
    compute_vae_elbo,
    evaluate_latent,
    extract_latent,
    extract_original_data,
)
from src.models.autoencoder import Autoencoder
from src.models.scvi_wrapper import ScVIWrapper
from src.models.vae import VAE
from src.training.trainer import Trainer
from src.utils.logging_utils import (
    make_tensorboard_run_subdir,
    setup_logger,
    write_tensorboard_run_metadata,
)


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    """Format timezone-aware UTC datetime as ISO string with trailing Z."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_run_id() -> str:
    """Build a collision-resistant run id, including SLURM context when available."""
    base = _utc_now().strftime("run_%Y%m%dT%H%M%SZ")
    suffix = []
    job_id = os.getenv("SLURM_JOB_ID")
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if job_id:
        suffix.append(f"j{job_id}")
    if task_id:
        suffix.append(f"a{task_id}")
    suffix.append(f"p{os.getpid()}")
    return "_".join([base] + suffix)


def parse_loss_matrix(spec) -> List[Tuple[str, str]]:
    """Parse model/loss matrix from CSV string or list entries."""
    if isinstance(spec, str):
        tokens = spec.split(",")
    else:
        tokens = list(spec or [])

    items = []
    for raw in tokens:
        token = str(raw).strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid loss matrix item {token}. Expected model:loss")
        model, loss = [x.strip() for x in token.split(":", 1)]
        items.append((model, loss))

    if not items:
        raise ValueError("Empty loss matrix")
    return items


def validate_combo(model_type: str, loss_type: str) -> None:
    valid = {
        "ae": {"mse", "nb", "zinb"},
        "vae": {"mse", "nb", "zinb"},
        "scvi": {"nb", "zinb"},
    }
    if model_type not in valid or loss_type not in valid[model_type]:
        raise ValueError(
            f"Unsupported model/loss combination: {model_type}:{loss_type}. "
            f"Allowed: {valid}"
        )


def get_git_commit() -> str:
    """Return short git commit hash when available."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(os.path.join(project_root, ".git")):
            return "unknown"
        out = subprocess.check_output(
            ["git", "-C", project_root, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _fit_formula_params(K: np.ndarray, d: np.ndarray, name: str) -> np.ndarray:
    """Fit candidate formula and return fitted parameters."""
    if name == "linear":
        return np.polyfit(K, d, 1)
    if name == "power":
        Kp = np.clip(K, 1e-8, None)
        dp = np.clip(d, 1e-8, None)
        return np.polyfit(np.log(Kp), np.log(dp), 1)
    if name == "sqrt":
        return np.polyfit(np.sqrt(K), d, 1)
    if name == "logk":
        X = K * np.log(np.clip(K, 1e-8, None))
        return np.polyfit(X, d, 1)
    raise ValueError(f"Unknown formula candidate: {name}")


def _predict_formula(K: np.ndarray, name: str, params: np.ndarray) -> np.ndarray:
    """Predict d from fitted formula parameters for given K values."""
    if name == "linear":
        return params[0] * K + params[1]
    if name == "power":
        Kp = np.clip(K, 1e-8, None)
        return np.exp(params[1]) * np.power(Kp, params[0])
    if name == "sqrt":
        return params[0] * np.sqrt(K) + params[1]
    if name == "logk":
        X = K * np.log(np.clip(K, 1e-8, None))
        return params[0] * X + params[1]
    raise ValueError(f"Unknown formula candidate: {name}")


def _fit_model(K: np.ndarray, d: np.ndarray, name: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fit candidate formula and return predictions + scores."""
    params = _fit_formula_params(K, d, name)
    pred = _predict_formula(K, name, params)

    resid = d - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((d - np.mean(d)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid**2)))

    return pred, {"r2": r2, "rmse": rmse}


def _resolve_label_source(adata, label_key: Optional[str]) -> str:
    """Resolve label provenance string with a conservative fallback."""
    source = adata.uns.get("cell_type_source")
    if isinstance(source, str) and source:
        return source
    if label_key:
        return f"provided:{label_key}"
    return "unknown"


def _label_source_category(label_source: str) -> str:
    """Map label_source string to coarse category."""
    src = str(label_source or "").strip().lower()
    if src.startswith("provided:"):
        return "provided"
    if src.startswith("inferred:"):
        return "inferred"
    if src in {"", "nan", "none", "unknown"}:
        return "unknown"
    return "other"


def _get_onboard_contract(adata) -> Dict:
    """Safely read onboarding provenance metadata from adata.uns."""
    if adata is None:
        return {}
    try:
        contract = adata.uns.get("onboard_contract", {})
    except Exception:
        return {}
    return contract if isinstance(contract, dict) else {}


def _ts2_has_ground_truth_provenance(
    adata,
    label_key: Optional[str],
    label_source: str,
) -> bool:
    """Strict TS2 provenance gate for promotion to trusted ground truth.

    Accepted evidence:
      1) canonical label key is cell_ontology_class
      2) label_source explicitly records provided:cell_ontology_class
      3) onboard_contract records source_label_key=cell_ontology_class and label_key=cell_type
      4) fallback: cell_type matches cell_ontology_class >=99% exactly (if both columns exist)
    """
    key = str(label_key or "")
    src = str(label_source or "")

    if key == "cell_ontology_class":
        return True
    if src.startswith("provided:cell_ontology_class"):
        return True

    contract = _get_onboard_contract(adata)
    source_label_key = str(contract.get("source_label_key", ""))
    contract_label_key = str(contract.get("label_key", ""))
    source_label_ontology_key = str(contract.get("source_label_ontology_key", ""))
    try:
        source_label_ontology_nonnull_fraction = float(contract.get("source_label_ontology_nonnull_fraction", float("nan")))
    except Exception:
        source_label_ontology_nonnull_fraction = float("nan")

    if source_label_key == "cell_ontology_class" and contract_label_key == "cell_type":
        return True

    # Strict alternative for TS2 exports that use cell_type + ontology term ids.
    if (
        source_label_key == "cell_type"
        and contract_label_key == "cell_type"
        and source_label_ontology_key == "cell_type_ontology_term_id"
        and np.isfinite(source_label_ontology_nonnull_fraction)
        and source_label_ontology_nonnull_fraction >= 0.99
    ):
        return True

    if adata is not None and "cell_ontology_class" in getattr(adata, "obs", {}) and "cell_type" in getattr(adata, "obs", {}):
        try:
            same = (
                adata.obs["cell_type"].astype(str).values
                == adata.obs["cell_ontology_class"].astype(str).values
            )
            if float(np.mean(same)) >= 0.99:
                return True
        except Exception:
            pass

    return False


def _resolve_label_trust(
    dataset_name: str,
    label_key: Optional[str],
    label_source: str,
    adata=None,
) -> str:
    """Resolve label trust semantics.

    Trust categories:
      - ground_truth: trusted curated labels for primary analyses
      - untrusted: pseudo/synthetic labels unsuitable for primary claims
      - unknown: no explicit trust guarantee
    """
    dataset = str(dataset_name or "")
    base = str(get_label_trust(dataset) or "unknown")

    if base in {"ground_truth", "untrusted"}:
        return base

    ds_lower = dataset.lower()
    if ds_lower.startswith("splatter_"):
        return "untrusted"

    # TS2 family: strict promotion only with explicit ontology provenance evidence.
    if ds_lower.startswith("ts2_"):
        if _ts2_has_ground_truth_provenance(adata=adata, label_key=label_key, label_source=label_source):
            return "ground_truth"
        return "unknown"

    # TS1 family: curated ontology labels are accepted.
    if dataset == "ts1_all_cells" or ds_lower.startswith("ts1_"):
        key = str(label_key or "")
        src = str(label_source or "")
        if key == "cell_ontology_class":
            return "ground_truth"
        if key == "cell_type" and "cell_ontology_class" in src:
            return "ground_truth"

    # Allen Immune Health Atlas: trust only official AIFI label columns.
    if dataset == "aifi_immune_full" or ds_lower.startswith("aifi_"):
        key = str(label_key or "")
        src = str(label_source or "")
        allowed = {"AIFI_L1", "AIFI_L2", "AIFI_L3"}
        if key in allowed:
            return "ground_truth"
        if key == "cell_type":
            if any(f"provided:{k}" in src for k in allowed):
                return "ground_truth"
            if adata is not None:
                try:
                    for k in allowed:
                        if k in adata.obs:
                            same = (
                                adata.obs["cell_type"].astype(str).values
                                == adata.obs[k].astype(str).values
                            )
                            if float(np.mean(same)) >= 0.99:
                                return "ground_truth"
                except Exception:
                    pass

    # Strict default: do not elevate plain provided:* to ground truth automatically.
    return "unknown"


def _derive_ground_truth_flag_from_row(row: pd.Series) -> bool:
    """Best-effort ground-truth flag for report-only compatibility with old CSVs."""
    trust = str(row.get("label_trust", "")).strip().lower()
    if trust:
        return trust == "ground_truth"

    dataset = str(row.get("dataset", ""))
    mapped = str(get_label_trust(dataset) or "unknown")
    if mapped in {"ground_truth", "untrusted"}:
        return mapped == "ground_truth"

    ds_lower = dataset.lower()
    if ds_lower.startswith("ts2_"):
        label_source = str(row.get("label_source", ""))
        if label_source.startswith("provided:cell_ontology_class"):
            return True
        source_label_key = str(row.get("onboard_source_label_key", ""))
        contract_label_key = str(row.get("onboard_contract_label_key", row.get("label_key", "")))
        source_label_ontology_key = str(row.get("onboard_source_label_ontology_key", ""))
        try:
            source_label_ontology_nonnull_fraction = float(row.get("onboard_source_label_ontology_nonnull_fraction", float("nan")))
        except Exception:
            source_label_ontology_nonnull_fraction = float("nan")

        if source_label_key == "cell_ontology_class" and contract_label_key in {"cell_type", "cell_ontology_class"}:
            return True
        if (
            source_label_key == "cell_type"
            and contract_label_key in {"cell_type", "cell_ontology_class"}
            and source_label_ontology_key == "cell_type_ontology_term_id"
            and np.isfinite(source_label_ontology_nonnull_fraction)
            and source_label_ontology_nonnull_fraction >= 0.99
        ):
            return True
        evidence = row.get("ts2_ground_truth_evidence", False)
        if pd.notna(evidence):
            if isinstance(evidence, str):
                return evidence.strip().lower() in {"1", "true", "yes"}
            return bool(evidence)
        return False

    # Legacy fallback for unknown non-TS2 datasets only.
    label_source = str(row.get("label_source", ""))
    return label_source.startswith("provided:cell_ontology_class")


def _metric_selection_column(df: pd.DataFrame, mode: str = "current_fallback") -> Optional[str]:
    """Choose ranking metric based on mode."""
    if mode != "current_fallback":
        if mode in df.columns and df[mode].notna().any():
            return mode
        return None

    for col in ("ari", "silhouette_kmeans", "silhouette"):
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _matches_any_pattern(value: str, patterns: List[str]) -> bool:
    """Return True when value matches at least one glob-style pattern."""
    if not patterns:
        return False
    val = str(value)
    return any(fnmatch(val, pat) for pat in patterns)


def _should_preload(adata, config) -> bool:
    """Choose dataset materialization mode for DataLoader construction."""
    if config.data.dense_conversion_policy == "never":
        return False
    if config.data.use_backed_mode and adata.n_obs >= config.data.backed_threshold_cells:
        return False
    return True


def _select_true_labels_for_external_metrics(
    adata_eval,
    label_source: str,
    label_trust: str,
    config,
):
    """Return true labels array when external metrics are valid; otherwise None."""
    if not config.experiment.require_ground_truth_for_external_metrics:
        return adata_eval.obs["cell_type"].values, True, "enabled"

    is_ground_truth = str(label_trust or "").strip().lower() == "ground_truth"
    if is_ground_truth:
        return adata_eval.obs["cell_type"].values, True, "enabled_ground_truth"

    note = (
        "disabled_no_ground_truth("
        f"label_source={label_source},label_trust={label_trust}"
        ")"
    )
    return None, False, note


def _compute_zero_fraction(adata_eval, layer: Optional[str]) -> float:
    """Compute zero-entry fraction for selected matrix layer (safe fallback to NaN)."""
    try:
        X = adata_eval.layers[layer] if layer else adata_eval.X
        n_obs = int(adata_eval.n_obs)
        n_vars = int(adata_eval.n_vars)
        total = float(n_obs * n_vars)
        if total <= 0:
            return float("nan")
        if hasattr(X, "nnz"):
            nnz = float(X.nnz)
            return float(max(0.0, min(1.0, 1.0 - (nnz / total))))
        arr = np.asarray(X)
        nnz = float(np.count_nonzero(arr))
        return float(max(0.0, min(1.0, 1.0 - (nnz / total))))
    except Exception:
        return float("nan")


def _warn_large_run_storage_paths(config, logger) -> None:
    """Warn when large-run mode points active paths under HOME-like storage."""
    if not config.data.large_run_mode:
        return

    home_real = os.path.realpath(os.path.expanduser("~"))
    for label, path in {
        "data_dir": config.paths.data_dir,
        "results_dir": config.paths.results_dir,
        "log_dir": config.paths.log_dir,
        "checkpoint_dir": config.paths.checkpoint_dir,
    }.items():
        path_real = os.path.realpath(path)
        if path_real == home_real or path_real.startswith(home_real + os.sep):
            logger.warning(
                "Large-run mode path '%s' is under HOME-like storage: %s. "
                "Use WORK-backed paths for active data/results/checkpoints/logs.",
                label,
                path,
            )


def _warn_large_run_data_settings(adata, config, logger, dataset_name: str) -> None:
    """Warn on large-run-hostile settings without mutating config."""
    if not config.data.large_run_mode:
        return

    is_sparse_like = hasattr(adata.X, "toarray")
    preload = _should_preload(adata, config)

    if is_sparse_like and config.data.dense_conversion_policy != "never":
        logger.warning(
            "Large-run mode on dataset '%s': dense_conversion_policy=%s may densify sparse data. "
            "Recommended: dense_conversion_policy='never'.",
            dataset_name,
            config.data.dense_conversion_policy,
        )

    if is_sparse_like and preload and adata.n_obs >= config.data.backed_threshold_cells:
        logger.warning(
            "Large-run mode on dataset '%s': preload=True with sparse-like matrix and n_obs=%d can increase memory pressure.",
            dataset_name,
            int(adata.n_obs),
        )


def _warn_array_output_isolation(config, logger, output_path: Optional[str]) -> None:
    """Warn when SLURM array execution appears to use non-isolated output paths."""
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if not array_task_id:
        return

    # Path hints expected for task isolation.
    hint_tokens = {
        f"task_{array_task_id}",
        f"_{array_task_id}",
        os.sep + str(array_task_id) + os.sep,
    }

    def _has_task_hint(path: str) -> bool:
        norm = os.path.realpath(path)
        return any(token in norm for token in hint_tokens)

    for label, path in {
        "results_dir": config.paths.results_dir,
        "log_dir": config.paths.log_dir,
        "checkpoint_dir": config.paths.checkpoint_dir,
    }.items():
        if not _has_task_hint(path):
            logger.warning(
                "SLURM array task detected but '%s' does not appear task-isolated: %s. "
                "Prefer per-task directories (e.g., .../task_%s/...).",
                label,
                path,
                array_task_id,
            )

    if output_path and not _has_task_hint(output_path):
        logger.warning(
            "SLURM array task detected but output path may be shared across tasks: %s. "
            "Prefer task-local output paths to avoid concurrent write collisions.",
            output_path,
        )

def _warn_hpc_without_slurm_allocation(logger) -> None:
    """Warn when likely on FAU HPC but not inside an active SLURM allocation."""
    has_slurm_context = bool(os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_STEP_ID"))
    if has_slurm_context:
        return

    hostname = (os.getenv("HOSTNAME") or "").lower()
    home_path = os.path.realpath(os.path.expanduser("~")).lower()
    looks_fau_hpc = (
        "/home/hpc/" in home_path
        or "nhr.fau.de" in hostname
        or hostname.startswith("tiny")
        or hostname.startswith("alex")
    )

    if looks_fau_hpc:
        logger.warning(
            "No active SLURM allocation detected (SLURM_JOB_ID/SLURM_STEP_ID missing). "
            "On FAU HPC, long training runs should be launched via sbatch/srun to avoid "
            "session/policy termination in plain shells."
        )


def _write_manifest(manifest_path: str, manifest: dict, logger=None) -> None:
    """Write run manifest safely."""
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to write manifest %s: %s", manifest_path, exc)


def _persist_partial_results(all_results: List[dict], output_path: str, run_dir: str, logger=None) -> None:
    """Persist currently completed rows so interruptions do not lose all progress."""
    if not all_results:
        return
    try:
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        df.to_csv(os.path.join(run_dir, "experiment_results_all.csv"), index=False)
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to persist partial results: %s", exc)


def _probe_backed_lazy_compatibility(
    adata_backed,
    config,
    combos: List[Tuple[str, str]],
    seed: int,
    batch_key: Optional[str],
) -> Tuple[bool, str]:
    """Run a lightweight side-effect-free compatibility probe for backed/lazy mode."""
    if int(adata_backed.n_obs) < 16:
        return False, "probe_failed: too few cells for compatibility check"

    if "cell_type" not in adata_backed.obs.columns:
        return False, "probe_failed: missing obs['cell_type']"

    needs_counts = any(loss in {"nb", "zinb"} for _, loss in combos)
    if needs_counts and "counts" not in adata_backed.layers:
        return False, "probe_failed: missing layers['counts'] for NB/ZINB"

    probe_cells = max(16, min(int(config.data.backed_probe_cells), int(adata_backed.n_obs)))
    probe_idx = np.arange(probe_cells)

    try:
        adata_probe = adata_backed[probe_idx].to_memory()
    except Exception as exc:
        return False, f"probe_failed: to_memory on probe slice failed ({exc})"

    try:
        adata_train, adata_val, adata_test = split_adata(
            adata_probe,
            config.training.train_frac,
            config.training.val_frac,
            seed=seed,
        )
    except Exception as exc:
        return False, f"probe_failed: split_adata failed ({exc})"

    if min(int(adata_train.n_obs), int(adata_val.n_obs), int(adata_test.n_obs)) < 1:
        return False, "probe_failed: split produced empty subset"

    try:
        data_layer = "counts" if needs_counts else None
        train_loader, _, _ = make_dataloaders(
            adata_train,
            adata_val,
            adata_test,
            batch_size=max(4, min(int(config.training.batch_size), 32)),
            layer=data_layer,
            dense_policy=config.data.dense_conversion_policy,
            preload=False,
            num_workers=0,
            pin_memory=False,
        )
        batch = next(iter(train_loader))
    except Exception as exc:
        return False, f"probe_failed: dataloader construction/fetch failed ({exc})"

    try:
        input_dim = int(adata_probe.n_vars)
        latent_dim = max(1, min(2, input_dim))
        probe_loss = "nb" if needs_counts else "mse"
        probe_hidden = [16]

        ae_probe = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=probe_hidden,
            activation=config.model.activation,
            dropout=config.model.dropout,
            loss_type=probe_loss,
        )
        _ = ae_probe(batch)

        vae_probe = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=probe_hidden,
            activation=config.model.activation,
            dropout=config.model.dropout,
            loss_type=probe_loss,
        )
        _ = vae_probe(batch, deterministic=True)
    except Exception as exc:
        return False, f"probe_failed: AE/VAE forward compatibility failed ({exc})"

    if any(model == "scvi" for model, _ in combos):
        try:
            import scvi
        except Exception as exc:
            return False, f"probe_failed: scvi import failed ({exc})"

        try:
            scvi_like = next((loss for model, loss in combos if model == "scvi"), "nb")
            adata_scvi_probe = adata_train[: min(64, int(adata_train.n_obs))].copy()
            if "counts" not in adata_scvi_probe.layers:
                return False, "probe_failed: scVI probe missing counts layer"

            if batch_key:
                if batch_key not in adata_scvi_probe.obs.columns:
                    return False, f"probe_failed: scVI probe missing batch_key={batch_key}"
                scvi.model.SCVI.setup_anndata(adata_scvi_probe, layer="counts", batch_key=batch_key)
            else:
                scvi.model.SCVI.setup_anndata(adata_scvi_probe, layer="counts")

            _ = scvi.model.SCVI(
                adata_scvi_probe,
                n_latent=max(1, min(2, int(adata_probe.n_vars))),
                n_hidden=16,
                n_layers=1,
                gene_likelihood=scvi_like,
            )
        except Exception as exc:
            return False, f"probe_failed: scVI setup/instantiation failed ({exc})"

    return True, f"probe_passed(n_cells={probe_cells})"


def _build_recommendations(
    df: pd.DataFrame,
    selection_mode: str,
    logger,
    scope_name: str,
) -> pd.DataFrame:
    """Build per-dataset recommendations of best latent dimension by selected metric policy."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model_type",
                "loss_type",
                "K",
                "recommended_d",
                "selection_metric",
                "selection_score",
                "mean_ari",
                "mean_silhouette_kmeans",
                "report_scope",
            ]
        )

    metric_candidates = [
        col for col in ("ari", "ami", "silhouette_kmeans", "silhouette")
        if col in df.columns
    ]
    if not metric_candidates:
        logger.warning("No score columns available for %s recommendation build.", scope_name)
        return pd.DataFrame()

    grouped = (
        df.groupby(
            ["dataset", "model_type", "loss_type", "latent_dim", "K"],
            as_index=False,
        )[list(dict.fromkeys(metric_candidates + ["ari", "silhouette_kmeans"]))]
        .mean(numeric_only=True)
    )

    best_rows = []
    for (dataset, model_type, loss_type), sub in grouped.groupby(["dataset", "model_type", "loss_type"]):
        score_col = _metric_selection_column(sub, mode=selection_mode)
        if score_col is None:
            logger.warning(
                "No valid ranking metric for %s scope=%s | %s:%s (mode=%s); skipping.",
                dataset,
                scope_name,
                model_type,
                loss_type,
                selection_mode,
            )
            continue

        ranked = sub.dropna(subset=[score_col])
        if ranked.empty:
            logger.warning(
                "Ranking metric %s is all-NaN for %s scope=%s | %s:%s; skipping.",
                score_col,
                dataset,
                scope_name,
                model_type,
                loss_type,
            )
            continue

        best_idx = ranked[score_col].idxmax()
        best = ranked.loc[best_idx]
        best_rows.append(
            {
                "dataset": dataset,
                "model_type": model_type,
                "loss_type": loss_type,
                "K": int(best["K"]),
                "recommended_d": int(best["latent_dim"]),
                "selection_metric": score_col,
                "selection_score": float(best[score_col]),
                "mean_ari": float(best["ari"]) if "ari" in best else float("nan"),
                "mean_silhouette_kmeans": (
                    float(best["silhouette_kmeans"])
                    if "silhouette_kmeans" in best
                    else float("nan")
                ),
                "report_scope": scope_name,
            }
        )

    return pd.DataFrame(best_rows)


def _fit_formula_summary(rec_df: pd.DataFrame, config, logger, scope_name: str) -> pd.DataFrame:
    """Fit formula candidates on recommendation table."""
    if rec_df.empty:
        return pd.DataFrame(
            columns=["model_type", "loss_type", "fit_formula_name", "fit_score_r2", "fit_rmse", "report_scope"]
        )

    fit_rows = []
    for (model_type, loss_type), sub in rec_df.groupby(["model_type", "loss_type"]):
        n_unique_k = int(sub["K"].nunique())
        if n_unique_k < config.experiment.min_k_for_fit:
            fit_rows.append(
                {
                    "model_type": model_type,
                    "loss_type": loss_type,
                    "fit_formula_name": "insufficient_k_coverage",
                    "fit_score_r2": float("nan"),
                    "fit_rmse": float("nan"),
                    "report_scope": scope_name,
                }
            )
            logger.info(
                "Skipping fit for %s:%s (%s, unique K=%d < min_k_for_fit=%d)",
                model_type,
                loss_type,
                scope_name,
                n_unique_k,
                config.experiment.min_k_for_fit,
            )
            continue

        K = sub["K"].to_numpy(dtype=float)
        d = sub["recommended_d"].to_numpy(dtype=float)

        for formula in config.experiment.formula_candidates:
            try:
                _, score = _fit_model(K, d, formula)
            except Exception as exc:
                logger.warning(
                    "Formula fit failed (%s, %s:%s, formula=%s): %s",
                    scope_name,
                    model_type,
                    loss_type,
                    formula,
                    exc,
                )
                score = {"r2": float("nan"), "rmse": float("nan")}

            fit_rows.append(
                {
                    "model_type": model_type,
                    "loss_type": loss_type,
                    "fit_formula_name": formula,
                    "fit_score_r2": score["r2"],
                    "fit_rmse": score["rmse"],
                    "report_scope": scope_name,
                }
            )

    return pd.DataFrame(fit_rows)


def _fit_formula_loo_primary(rec_primary: pd.DataFrame, config, logger) -> pd.DataFrame:
    """Leave-one-dataset-out validation for primary recommendations only."""
    loo_rows = []

    if rec_primary.empty:
        return pd.DataFrame(
            columns=[
                "model_type",
                "loss_type",
                "fit_formula_name",
                "held_out_dataset",
                "held_out_K",
                "predicted_d",
                "predicted_d_continuous",
                "predicted_d_nearest_grid",
                "observed_d",
                "abs_error",
                "squared_error",
                "train_n_datasets",
                "train_n_unique_k",
                "status",
            ]
        )

    for (model_type, loss_type), sub in rec_primary.groupby(["model_type", "loss_type"]):
        sub = sub[["dataset", "K", "recommended_d"]].drop_duplicates().reset_index(drop=True)
        if len(sub) < 2:
            continue

        for formula in config.experiment.formula_candidates:
            for _, held_row in sub.iterrows():
                held_ds = held_row["dataset"]
                held_k = float(held_row["K"])
                held_d = float(held_row["recommended_d"])

                train = sub[sub["dataset"] != held_ds]
                train_n = int(train["dataset"].nunique())
                train_k = int(train["K"].nunique())

                if len(train) < 2 or train_k < config.experiment.min_k_for_fit:
                    loo_rows.append(
                        {
                            "model_type": model_type,
                            "loss_type": loss_type,
                            "fit_formula_name": formula,
                            "held_out_dataset": held_ds,
                            "held_out_K": held_k,
                            "predicted_d": float("nan"),
                            "predicted_d_continuous": float("nan"),
                            "predicted_d_nearest_grid": float("nan"),
                            "observed_d": held_d,
                            "abs_error": float("nan"),
                            "squared_error": float("nan"),
                            "train_n_datasets": train_n,
                            "train_n_unique_k": train_k,
                            "status": "insufficient_train_k_coverage",
                        }
                    )
                    continue

                try:
                    params = _fit_formula_params(
                        train["K"].to_numpy(dtype=float),
                        train["recommended_d"].to_numpy(dtype=float),
                        formula,
                    )
                    pred = float(_predict_formula(np.asarray([held_k], dtype=float), formula, params)[0])
                    grid = np.asarray(sorted(set(config.experiment.latent_dims)), dtype=float)
                    if grid.size > 0 and np.isfinite(pred):
                        pred_nearest = float(grid[np.argmin(np.abs(grid - pred))])
                    else:
                        pred_nearest = float("nan")
                    abs_err = abs(pred - held_d)
                    sq_err = (pred - held_d) ** 2
                    status = "ok"
                except Exception as exc:
                    logger.warning(
                        "Primary LOO fit failed for %s:%s formula=%s held_out=%s: %s",
                        model_type,
                        loss_type,
                        formula,
                        held_ds,
                        exc,
                    )
                    pred = float("nan")
                    pred_nearest = float("nan")
                    abs_err = float("nan")
                    sq_err = float("nan")
                    status = "fit_failed"

                loo_rows.append(
                    {
                        "model_type": model_type,
                        "loss_type": loss_type,
                        "fit_formula_name": formula,
                        "held_out_dataset": held_ds,
                        "held_out_K": held_k,
                        "predicted_d": pred,
                        "predicted_d_continuous": pred,
                        "predicted_d_nearest_grid": pred_nearest,
                        "observed_d": held_d,
                        "abs_error": abs_err,
                        "squared_error": sq_err,
                        "train_n_datasets": train_n,
                        "train_n_unique_k": train_k,
                        "status": status,
                    }
                )

    loo_df = pd.DataFrame(loo_rows)
    if not loo_df.empty:
        stats = (
            loo_df[loo_df["status"] == "ok"]
            .groupby(["model_type", "loss_type", "fit_formula_name"], as_index=False)
            .agg(
                loo_mae=("abs_error", "mean"),
                loo_rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
                loo_n=("held_out_dataset", "count"),
            )
        )
        loo_df = loo_df.merge(stats, on=["model_type", "loss_type", "fit_formula_name"], how="left")

    return loo_df


def generate_auto_d_report(df: pd.DataFrame, config, out_dir: str, logger) -> None:
    """Generate legacy + primary/exploratory latent-d recommendations and formula fits."""
    success = df[df["status"] == "success"].copy()
    if success.empty:
        logger.warning("No successful runs; skipping auto_d_report")
        return

    # Legacy behavior retained for backward compatibility.
    rec_legacy = _build_recommendations(
        success,
        selection_mode="current_fallback",
        logger=logger,
        scope_name="legacy",
    )
    fit_legacy = _fit_formula_summary(rec_legacy, config, logger, scope_name="legacy")
    rec_legacy_path = os.path.join(out_dir, "recommended_d_by_K.csv")
    fit_legacy_path = os.path.join(out_dir, "formula_fit_summary.csv")
    rec_legacy.to_csv(rec_legacy_path, index=False)
    fit_legacy.to_csv(fit_legacy_path, index=False)

    # Primary scope: curated labels + source/dataset pattern controls.
    primary = success.copy()

    exclude_patterns = list(config.experiment.auto_d_primary_exclude_dataset_patterns or [])
    if exclude_patterns:
        keep_mask = ~primary["dataset"].astype(str).apply(lambda x: _matches_any_pattern(x, exclude_patterns))
        primary = primary[keep_mask].copy()

    if config.experiment.auto_d_primary_require_curated_labels:
        if "is_ground_truth_labels" in primary.columns:
            primary = primary[primary["is_ground_truth_labels"] == True].copy()  # noqa: E712
        elif "label_trust" in primary.columns:
            primary = primary[primary["label_trust"].astype(str).str.lower() == "ground_truth"].copy()
        else:
            gt_mask = primary.apply(_derive_ground_truth_flag_from_row, axis=1)
            primary = primary[gt_mask].copy()

    include_source_patterns = list(config.experiment.auto_d_primary_include_label_sources or [])
    if include_source_patterns:
        source_keep = primary["label_source"].astype(str).apply(
            lambda x: _matches_any_pattern(x, include_source_patterns)
        )
        primary = primary[source_keep].copy()

    rec_primary = _build_recommendations(
        primary,
        selection_mode=config.experiment.auto_d_primary_metric,
        logger=logger,
        scope_name="primary",
    )
    fit_primary = _fit_formula_summary(rec_primary, config, logger, scope_name="primary")

    rec_primary_path = os.path.join(out_dir, "recommended_d_by_K_primary.csv")
    fit_primary_path = os.path.join(out_dir, "formula_fit_summary_primary.csv")
    rec_primary.to_csv(rec_primary_path, index=False)
    fit_primary.to_csv(fit_primary_path, index=False)

    if config.experiment.auto_d_loo_validation:
        loo_primary = _fit_formula_loo_primary(rec_primary, config, logger)
        loo_primary_path = os.path.join(out_dir, "formula_fit_loo_primary.csv")
        loo_primary.to_csv(loo_primary_path, index=False)
    else:
        loo_primary_path = ""

    # Exploratory scope: optional, includes inferred/synthetic if present.
    if config.experiment.auto_d_generate_exploratory_report:
        rec_expl = _build_recommendations(
            success,
            selection_mode=config.experiment.auto_d_secondary_metric_mode,
            logger=logger,
            scope_name="exploratory",
        )
        fit_expl = _fit_formula_summary(rec_expl, config, logger, scope_name="exploratory")
    else:
        rec_expl = pd.DataFrame(columns=rec_legacy.columns)
        fit_expl = pd.DataFrame(columns=fit_legacy.columns)

    rec_expl_path = os.path.join(out_dir, "recommended_d_by_K_exploratory.csv")
    fit_expl_path = os.path.join(out_dir, "formula_fit_summary_exploratory.csv")
    rec_expl.to_csv(rec_expl_path, index=False)
    fit_expl.to_csv(fit_expl_path, index=False)

    logger.info(
        "Auto-d reports saved: %s, %s, %s, %s%s",
        rec_legacy_path,
        fit_legacy_path,
        rec_primary_path,
        fit_primary_path,
        f", {loo_primary_path}" if loo_primary_path else "",
    )


def run_single_experiment(
    adata,
    dataset_name,
    model_type,
    latent_dim,
    config,
    device,
    logger,
    seed,
    loss_type,
    batch_key,
    label_key,
    requested_data_mode,
    effective_data_mode,
    backed_fallback_reason,
    git_commit,
    tb_dir=None,
):
    """Run one (dataset, model, loss, d, seed) experiment."""
    n_genes = adata.n_vars
    K = int(adata.obs["cell_type"].nunique())
    label_source = _resolve_label_source(adata, label_key)
    label_source_category = _label_source_category(label_source)
    label_trust = _resolve_label_trust(dataset_name, label_key, label_source, adata=adata)
    is_ground_truth_labels = label_trust == "ground_truth"
    is_untrusted_labels = label_trust == "untrusted"
    # Backward-compatible alias retained for older analysis code.
    is_curated_labels = is_ground_truth_labels
    is_inferred_labels = label_source_category == "inferred"
    is_synthetic = dataset_name.lower().startswith("splatter_") or "synthetic" in dataset_name.lower()

    onboard_contract = _get_onboard_contract(adata)
    onboard_source_label_key = str(onboard_contract.get("source_label_key", ""))
    onboard_source_batch_key = str(onboard_contract.get("source_batch_key", ""))
    onboard_source_label_ontology_key = str(onboard_contract.get("source_label_ontology_key", ""))
    try:
        onboard_source_label_ontology_nonnull_fraction = float(
            onboard_contract.get("source_label_ontology_nonnull_fraction", float("nan"))
        )
    except Exception:
        onboard_source_label_ontology_nonnull_fraction = float("nan")
    onboard_contract_label_key = str(onboard_contract.get("label_key", ""))
    onboard_contract_batch_key = str(onboard_contract.get("batch_key", ""))
    ts2_ground_truth_evidence = bool(
        dataset_name.lower().startswith("ts2_")
        and _ts2_has_ground_truth_provenance(
            adata=adata,
            label_key=label_key,
            label_source=label_source,
        )
    )

    result = {
        "dataset": dataset_name,
        "model_type": model_type,
        "loss_type": loss_type,
        "latent_dim": latent_dim,
        "K": K,
        "n_cells": adata.n_obs,
        "n_cells_eval": float("nan"),
        "eval_split": "test",
        "n_genes": n_genes,
        "seed": seed,
        "batch_key": batch_key,
        "label_key": label_key,
        "label_source": label_source,
        "label_source_category": label_source_category,
        "label_trust": label_trust,
        "onboard_source_label_key": onboard_source_label_key,
        "onboard_source_batch_key": onboard_source_batch_key,
        "onboard_source_label_ontology_key": onboard_source_label_ontology_key,
        "onboard_source_label_ontology_nonnull_fraction": onboard_source_label_ontology_nonnull_fraction,
        "onboard_contract_label_key": onboard_contract_label_key,
        "onboard_contract_batch_key": onboard_contract_batch_key,
        "ts2_ground_truth_evidence": bool(ts2_ground_truth_evidence),
        "is_ground_truth_labels": bool(is_ground_truth_labels),
        "is_untrusted_labels": bool(is_untrusted_labels),
        "is_curated_labels": bool(is_curated_labels),
        "is_inferred_labels": bool(is_inferred_labels),
        "is_synthetic": bool(is_synthetic),
        "n_batches": float("nan"),
        "zero_fraction": float("nan"),
        "external_metrics_enabled": False,
        "external_metrics_note": "not_evaluated",
        "data_mode": effective_data_mode,
        "requested_data_mode": requested_data_mode,
        "effective_data_mode": effective_data_mode,
        "backed_fallback_reason": backed_fallback_reason,
        "git_commit": git_commit,
        "fit_formula_name": "",
        "fit_score_r2": float("nan"),
        "fit_rmse": float("nan"),
        "elbo": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
        "trustworthiness": float("nan"),
        "continuity": float("nan"),
        "silhouette_sampled": False,
        "silhouette_n_used": float("nan"),
        "silhouette_n_total": float("nan"),
        "batch_silhouette": float("nan"),
        "batch_silhouette_mixing": float("nan"),
        "batch_knn_entropy": float("nan"),
        "batch_metrics_mode": "not_available",
        "batch_metrics_n_used": float("nan"),
        "batch_metrics_n_total": float("nan"),
        "centrality_mode": "unknown",
        "centrality_n_used": float("nan"),
        "centrality_n_total": float("nan"),
        "error": "",
        "error_type": "none",
    }

    start = time.time()
    trainer = None

    try:
        # Holdout split for evaluation (prevents train/eval leakage)
        adata_train, adata_val, adata_test = split_adata(
            adata,
            config.training.train_frac,
            config.training.val_frac,
            seed=seed,
        )
        adata_eval = adata_test
        result["n_cells_eval"] = adata_eval.n_obs

        batch_labels_for_metrics = None
        if batch_key and batch_key in adata_eval.obs.columns:
            batch_labels_for_metrics = adata_eval.obs[batch_key].astype(str).values
        elif "batch" in adata_eval.obs.columns:
            batch_labels_for_metrics = adata_eval.obs["batch"].astype(str).values

        if batch_labels_for_metrics is not None:
            result["n_batches"] = int(pd.Series(batch_labels_for_metrics).nunique())

        eval_layer = "counts" if loss_type in {"nb", "zinb"} else None
        result["zero_fraction"] = _compute_zero_fraction(adata_eval, eval_layer)

        true_labels, external_enabled, external_note = _select_true_labels_for_external_metrics(
            adata_eval,
            label_source,
            label_trust,
            config,
        )
        result["external_metrics_enabled"] = external_enabled
        result["external_metrics_note"] = external_note

        n_clusters_eval = int(adata_eval.obs["cell_type"].nunique())
        if n_clusters_eval < 2:
            raise ValueError(
                "Evaluation split has <2 clusters; cannot run KMeans/silhouette metrics. "
                f"Try larger dataset or adjust split fractions. Observed K_eval={n_clusters_eval}."
            )

        tb_run_subdir = make_tensorboard_run_subdir(
            dataset=dataset_name,
            model_type=model_type,
            loss_type=loss_type,
            latent_dim=latent_dim,
            seed=seed,
        )
        if tb_dir:
            write_tensorboard_run_metadata(
                tb_dir,
                tb_run_subdir,
                {
                    "dataset": dataset_name,
                    "model_type": model_type,
                    "loss_type": loss_type,
                    "latent_dim": int(latent_dim),
                    "seed": int(seed),
                },
            )

        # Extract original test data for trustworthiness/continuity metrics
        import scipy.sparse as sp
        _X = adata_eval.X
        original_test_data = np.asarray(_X.toarray() if sp.issparse(_X) else _X)

        if model_type == "scvi":
            run_tag = tb_run_subdir

            scvi_wrapper = ScVIWrapper(
                latent_dim=latent_dim,
                n_hidden=config.model.hidden_dims[0] if config.model.hidden_dims else 128,
                n_layers=len(config.model.hidden_dims),
                gene_likelihood=loss_type,
                max_epochs=config.experiment.scvi_max_epochs,
                learning_rate=config.training.learning_rate,
                seed=seed,
                batch_key=batch_key,
                tensorboard_dir=tb_dir,
                run_tag=run_tag,
                log_every_n_epochs=config.training.log_every_n_epochs,
            )
            _ = scvi_wrapper.setup_and_train(adata_train)
            latent = scvi_wrapper.get_latent(adata_eval)
            recon_loss = scvi_wrapper.get_reconstruction_loss(adata_eval)
            result["elbo"] = scvi_wrapper.get_elbo(adata_eval)

            result["best_val_loss"] = float("nan")
            result["total_epochs"] = scvi_wrapper.actual_epochs
            result["peak_cpu_mem_mb"] = float("nan")
            result["peak_gpu_mem_mb"] = float("nan")
        else:
            if loss_type in {"nb", "zinb"} and "counts" not in adata.layers:
                raise ValueError("NB/ZINB loss requested but counts layer is missing")

            # Resolve pin_memory based on device
            use_pin = config.training.pin_memory and (device.type == "cuda")
            preload = _should_preload(adata, config)

            data_layer = "counts" if loss_type in {"nb", "zinb"} else None
            train_loader, val_loader, test_loader = make_dataloaders(
                adata_train,
                adata_val,
                adata_test,
                batch_size=config.training.batch_size,
                layer=data_layer,
                dense_policy=config.data.dense_conversion_policy,
                preload=preload,
                num_workers=config.training.num_workers,
                pin_memory=use_pin,
            )

            ModelClass = Autoencoder if model_type == "ae" else VAE
            model = ModelClass(
                input_dim=n_genes,
                latent_dim=latent_dim,
                hidden_dims=config.model.hidden_dims,
                activation=config.model.activation,
                dropout=config.model.dropout,
                loss_type=loss_type,
            )

            ckpt_name = f"{dataset_name}_{model_type}_{loss_type}_d{latent_dim}_s{seed}"
            dataset_dirs = config.ensure_dataset_dirs(dataset_name)

            trainer = Trainer(
                model=model,
                device=device,
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                early_stopping_patience=config.training.early_stopping_patience,
                vae_beta=config.model.vae_beta,
                checkpoint_dir=dataset_dirs["checkpoints"],
                loss_type=loss_type,
                profile_memory=config.training.profile_memory,
                max_grad_norm=config.training.max_grad_norm,
                tensorboard_dir=tb_dir,
                run_tag=tb_run_subdir,
                log_every_n_epochs=config.training.log_every_n_epochs,
            )
            history = trainer.train(
                train_loader,
                val_loader,
                max_epochs=config.training.max_epochs,
                checkpoint_name=ckpt_name,
            )

            recon_loss = compute_reconstruction_loss(model, test_loader, device, loss_type=loss_type)
            latent = extract_latent(model, test_loader, device)

            if model_type == "vae":
                result["elbo"] = compute_vae_elbo(
                    model,
                    test_loader,
                    device,
                    beta=config.model.vae_beta,
                    loss_type=loss_type,
                )

            result["best_val_loss"] = history.get("best_val_loss", float("nan"))
            result["total_epochs"] = history.get("total_epochs", 0)
            result["peak_cpu_mem_mb"] = history.get("peak_cpu_mem_mb", float("nan"))
            result["peak_gpu_mem_mb"] = history.get("peak_gpu_mem_mb", float("nan"))

        metrics = evaluate_latent(
            latent=latent,
            n_clusters=n_clusters_eval,
            true_labels=true_labels,
            n_init=config.experiment.kmeans_n_init,
            seed=seed,
            silhouette_max_cells=config.experiment.silhouette_max_cells,
            batch_labels=batch_labels_for_metrics,
            batch_metrics_enabled=config.experiment.batch_metrics_enabled,
            batch_metrics_max_cells=config.experiment.batch_metrics_max_cells,
            batch_metrics_knn_k=config.experiment.batch_metrics_knn_k,
            original_data=original_test_data,
        )

        centrality_var, centrality_meta = compute_centrality_variance(
            latent,
            n_neighbors=config.experiment.centrality_n_neighbors,
            policy=config.experiment.centrality_policy,
            threshold_cells=config.experiment.centrality_threshold_cells,
            sample_size=config.experiment.centrality_sample_size,
            seed=seed,
            return_metadata=True,
        )

        result.update(metrics)
        result.update(centrality_meta)
        result["reconstruction_loss"] = recon_loss
        result["centrality_variance"] = centrality_var
        result["status"] = "success"

    except Exception as exc:
        logger.error(
            "  FAILED: dataset=%s model=%s loss=%s d=%s seed=%s error=%s",
            dataset_name,
            model_type,
            loss_type,
            latent_dim,
            seed,
            exc,
            exc_info=True,
        )
        result["status"] = "failed"
        result["error"] = str(exc)
        result["error_type"] = type(exc).__name__
        result.update(
            {
                "ari": float("nan"),
                "ami": float("nan"),
                "silhouette": float("nan"),
                "silhouette_kmeans": float("nan"),
                "silhouette_true_labels": float("nan"),
                "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan"),
                "trustworthiness": float("nan"),
                "continuity": float("nan"),
                "silhouette_sampled": False,
                "silhouette_n_used": float("nan"),
                "silhouette_n_total": float("nan"),
                "batch_silhouette": float("nan"),
                "batch_silhouette_mixing": float("nan"),
                "batch_knn_entropy": float("nan"),
                "batch_metrics_mode": "failed",
                "batch_metrics_n_used": float("nan"),
                "batch_metrics_n_total": float("nan"),
                "reconstruction_loss": float("nan"),
                "centrality_variance": float("nan"),
                "centrality_mode": "failed",
                "centrality_n_used": float("nan"),
                "centrality_n_total": float("nan"),
                "best_val_loss": float("nan"),
                "total_epochs": 0,
                "peak_cpu_mem_mb": float("nan"),
                "peak_gpu_mem_mb": float("nan"),
            }
        )

    finally:
        if trainer is not None:
            try:
                trainer.close()
            except Exception as close_exc:
                logger.warning("  Failed to close trainer cleanly: %s", close_exc)

    result["runtime_seconds"] = time.time() - start
    return result


def main():
    parser = argparse.ArgumentParser(description="Run full latent dimension sweep experiment")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--model_types", type=str, default=None)
    parser.add_argument("--latent_dims", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--profile_memory", action="store_true")
    parser.add_argument("--auto_d_report", action="store_true")
    parser.add_argument("--dataset_tier", type=str, default=None, help="tier_a,tier_b,tier_c")
    parser.add_argument(
        "--loss_matrix",
        type=str,
        default=None,
        help="Comma-separated model:loss pairs, e.g. ae:mse,ae:nb,vae:mse,scvi:nb",
    )
    parser.add_argument("--batch_key", type=str, default=None)
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--report_only_csv", type=str, default=None, help="Generate auto-d reports from an existing CSV and exit")
    parser.add_argument("--auto_d_primary_metric", type=str, default=None)
    parser.add_argument("--auto_d_primary_include_label_sources", type=str, default=None)
    parser.add_argument("--auto_d_primary_exclude_dataset_patterns", type=str, default=None)
    parser.add_argument(
        "--auto_d_primary_require_curated_labels",
        dest="auto_d_primary_require_curated_labels",
        action="store_true",
    )
    parser.add_argument(
        "--no_auto_d_primary_require_curated_labels",
        dest="auto_d_primary_require_curated_labels",
        action="store_false",
    )
    parser.add_argument(
        "--auto_d_generate_exploratory_report",
        dest="auto_d_generate_exploratory_report",
        action="store_true",
    )
    parser.add_argument(
        "--no_auto_d_generate_exploratory_report",
        dest="auto_d_generate_exploratory_report",
        action="store_false",
    )
    parser.add_argument(
        "--auto_d_loo_validation",
        dest="auto_d_loo_validation",
        action="store_true",
    )
    parser.add_argument(
        "--no_auto_d_loo_validation",
        dest="auto_d_loo_validation",
        action="store_false",
    )
    parser.add_argument("--auto_d_secondary_metric_mode", type=str, default=None)
    parser.add_argument(
        "--batch_metrics_enabled",
        dest="batch_metrics_enabled",
        action="store_true",
    )
    parser.add_argument(
        "--no_batch_metrics_enabled",
        dest="batch_metrics_enabled",
        action="store_false",
    )
    parser.set_defaults(
        auto_d_primary_require_curated_labels=None,
        auto_d_generate_exploratory_report=None,
        auto_d_loo_validation=None,
        batch_metrics_enabled=None,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_dirs()
    global_dirs = config.ensure_global_result_dirs()

    if args.device:
        config.training.device = args.device
    if args.max_epochs:
        config.training.max_epochs = args.max_epochs
        config.experiment.scvi_max_epochs = args.max_epochs
    if args.profile_memory:
        config.training.profile_memory = True
    if args.auto_d_report:
        config.experiment.auto_d_report = True
    if args.batch_key:
        config.data.batch_key = args.batch_key
    if args.label_key:
        config.data.label_key = args.label_key

    if args.auto_d_primary_metric:
        config.experiment.auto_d_primary_metric = args.auto_d_primary_metric
    if args.auto_d_primary_include_label_sources is not None:
        config.experiment.auto_d_primary_include_label_sources = [
            x.strip() for x in args.auto_d_primary_include_label_sources.split(",") if x.strip()
        ]
    if args.auto_d_primary_exclude_dataset_patterns is not None:
        config.experiment.auto_d_primary_exclude_dataset_patterns = [
            x.strip() for x in args.auto_d_primary_exclude_dataset_patterns.split(",") if x.strip()
        ]
    if args.auto_d_primary_require_curated_labels is not None:
        config.experiment.auto_d_primary_require_curated_labels = bool(args.auto_d_primary_require_curated_labels)
    if args.auto_d_generate_exploratory_report is not None:
        config.experiment.auto_d_generate_exploratory_report = bool(args.auto_d_generate_exploratory_report)
    if args.auto_d_loo_validation is not None:
        config.experiment.auto_d_loo_validation = bool(args.auto_d_loo_validation)
    if args.auto_d_secondary_metric_mode:
        config.experiment.auto_d_secondary_metric_mode = args.auto_d_secondary_metric_mode
    if args.batch_metrics_enabled is not None:
        config.experiment.batch_metrics_enabled = bool(args.batch_metrics_enabled)

    # Re-validate after CLI overrides.
    config.validate()

    if args.dataset_tier:
        selected_tiers = [x.strip() for x in args.dataset_tier.split(",") if x.strip()]
        datasets = list_datasets_by_tiers(selected_tiers)
    elif args.datasets:
        datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    else:
        datasets = config.experiment.datasets

    if args.model_types:
        model_types = [x.strip() for x in args.model_types.split(",") if x.strip()]
    else:
        model_types = config.experiment.model_types

    if args.latent_dims:
        latent_dims = [int(x.strip()) for x in args.latent_dims.split(",") if x.strip()]
    else:
        latent_dims = config.experiment.latent_dims

    if args.loss_matrix:
        combos = parse_loss_matrix(args.loss_matrix)
    elif config.experiment.loss_matrix:
        combos = parse_loss_matrix(config.experiment.loss_matrix)
    else:
        combos = []
        for model in model_types:
            if model == "scvi":
                combos.append((model, config.experiment.scvi_gene_likelihood))
            else:
                combos.append((model, config.training.loss_type))

    model_set = set(model_types)
    combos = [(model, loss) for model, loss in combos if model in model_set]
    if not combos:
        raise ValueError(
            "No model/loss combinations remain after applying model_types filter. "
            f"model_types={model_types}"
        )

    for model_type, loss_type in combos:
        validate_combo(model_type, loss_type)

    device = config.resolve_device()

    # Optimize A100 Tensor Core performance
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    logger = setup_logger("experiment", config.paths.log_dir)
    _warn_hpc_without_slurm_allocation(logger)
    _warn_large_run_storage_paths(config, logger)
    default_output = os.path.join(global_dirs["tables"], "experiment_results_all.csv")
    output_path = args.output or default_output
    _warn_array_output_isolation(config, logger, output_path)

    if args.report_only_csv:
        report_csv = os.path.abspath(args.report_only_csv)
        if not os.path.exists(report_csv):
            raise FileNotFoundError(f"report_only_csv not found: {report_csv}")
        logger.info("Report-only mode: generating auto-d reports from %s", report_csv)
        df_report = pd.read_csv(report_csv)
        generate_auto_d_report(df_report, config, global_dirs["tables"], logger)
        logger.info("Report-only mode complete. Reports written under %s", global_dirs["tables"])
        return

    if os.getenv("SLURM_ARRAY_TASK_ID") and not config.data.large_run_mode:
        logger.warning(
            "Detected SLURM array task without large_run_mode enabled. "
            "Ensure results_dir/checkpoint_dir/log_dir are task-isolated to avoid collisions."
        )

    git_commit = get_git_commit()
    run_id = _build_run_id()
    run_dir = os.path.join(global_dirs["runs"], run_id)
    os.makedirs(run_dir, exist_ok=True)

    # TensorBoard dir for this run (timestamped)
    tb_dir = None
    if config.training.tensorboard:
        tb_dir = os.path.join(config.paths.log_dir, "tensorboard", run_id)
        os.makedirs(tb_dir, exist_ok=True)
        logger.info(f"  TensorBoard: {tb_dir}")
        logger.info(f"  View live:   tensorboard --logdir {os.path.join(config.paths.log_dir, 'tensorboard')}")

    logger.info("=" * 70)
    logger.info("EXPERIMENT: Latent Dimension Sweep")
    logger.info(f"  Datasets:       {datasets}")
    logger.info(f"  Model/losses:   {combos}")
    logger.info(f"  Latent dims:    {latent_dims}")
    logger.info(f"  Seeds:          {config.experiment.n_seeds}")
    logger.info(f"  Device:         {device}")
    logger.info(f"  Git commit:     {git_commit}")
    logger.info(f"  Run ID:         {run_id}")
    logger.info(f"  Large run mode: {config.data.large_run_mode}")
    logger.info("=" * 70)

    run_manifest = {
        "run_id": run_id,
        "git_commit": git_commit,
        "config_path": args.config,
        "datasets": datasets,
        "model_loss_combos": combos,
        "latent_dims": latent_dims,
        "n_seeds": config.experiment.n_seeds,
        "device": str(device),
        "large_run_mode": bool(config.data.large_run_mode),
        "requested_backed_mode": bool(config.data.use_backed_mode),
        "timestamp_utc": _iso_utc(_utc_now()),
        "status": "running",
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    _write_manifest(manifest_path, run_manifest, logger)

    all_results = []
    interrupted_state = {"signal": ""}

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)

    def _handle_interrupt(signum, _frame):
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = f"SIGNAL_{signum}"
        interrupted_state["signal"] = sig_name
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_interrupt)
    signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        for ds_name in datasets:
            dataset_dirs = config.ensure_dataset_dirs(ds_name)
            processed_path = os.path.join(config.paths.data_dir, "processed", f"{ds_name}.h5ad")

            requested_data_mode = "backed" if config.data.use_backed_mode else "in_memory"
            effective_data_mode = "in_memory"
            backed_fallback_reason = ""

            if not os.path.exists(processed_path):
                logger.info(f"Preprocessing {ds_name}...")
                raw_adata = load_dataset(ds_name, data_dir=config.paths.data_dir)
                label_key = resolve_label_key(raw_adata, config.data.label_key, ds_name)
                batch_key = resolve_batch_key(raw_adata, config.data.batch_key, ds_name)
                adata = preprocess(
                    raw_adata,
                    config.preprocessing,
                    cell_type_key=label_key,
                    batch_key=batch_key,
                )
                save_processed(adata, processed_path)
                effective_data_mode = "in_memory"
            else:
                backed_mode = "r" if config.data.use_backed_mode else None
                adata_loaded = load_processed(processed_path, backed=backed_mode)

                label_key = resolve_label_key(adata_loaded, config.data.label_key, ds_name)
                batch_key = resolve_batch_key(adata_loaded, config.data.batch_key, ds_name)

                cache_ok, cache_msg = check_cached_preprocess_fingerprint(
                    adata_loaded,
                    config.preprocessing,
                    cell_type_key=label_key,
                    batch_key=batch_key,
                )
                if not cache_ok:
                    logger.warning(
                        "Processed cache check for %s: %s. "
                        "Continuing with cached file %s; rerun preprocess to refresh.",
                        ds_name,
                        cache_msg,
                        processed_path,
                    )

                if backed_mode:
                    if config.data.large_run_mode and config.data.keep_backed_until_split:
                        probe_ok, probe_msg = _probe_backed_lazy_compatibility(
                            adata_loaded,
                            config,
                            combos,
                            seed=config.seed,
                            batch_key=batch_key,
                        )
                        if probe_ok:
                            adata = adata_loaded
                            effective_data_mode = "backed_preserved"
                            logger.info("Backed compatibility probe passed for %s: %s", ds_name, probe_msg)
                        else:
                            adata = adata_loaded.to_memory()
                            effective_data_mode = "in_memory_fallback"
                            backed_fallback_reason = probe_msg
                            logger.warning(
                                "Backed preservation disabled for %s: %s. Falling back to in-memory mode.",
                                ds_name,
                                probe_msg,
                            )
                    else:
                        adata = adata_loaded.to_memory()
                        effective_data_mode = "in_memory_from_backed"
                        if config.data.large_run_mode:
                            logger.info(
                                "Large-run mode on %s using in-memory path (keep_backed_until_split=%s).",
                                ds_name,
                                config.data.keep_backed_until_split,
                            )
                else:
                    adata = adata_loaded
                    effective_data_mode = "in_memory"

            # Backfill label provenance for older processed files.
            if not adata.uns.get("cell_type_source"):
                adata.uns["cell_type_source"] = f"provided:{label_key}" if label_key else "unknown"

            _warn_large_run_data_settings(adata, config, logger, ds_name)

            K = int(adata.obs["cell_type"].nunique())
            logger.info(
                f"\nDataset: {ds_name} ({adata.n_obs} cells, K={K}, "
                f"requested_mode={requested_data_mode}, effective_mode={effective_data_mode}, "
                f"label_source={adata.uns.get('cell_type_source', 'unknown')})"
            )
            if backed_fallback_reason:
                logger.info("  Backed fallback reason: %s", backed_fallback_reason)

            dataset_results = []

            for model_type, loss_type in combos:
                for d in latent_dims:
                    for seed_idx in range(config.experiment.n_seeds):
                        seed = config.seed + seed_idx
                        _ = config.set_seed(seed)

                        logger.info(
                            f"\n--- {ds_name} | {model_type}:{loss_type} | d={d} | seed={seed} ---"
                        )

                        result = run_single_experiment(
                            adata=adata,
                            dataset_name=ds_name,
                            model_type=model_type,
                            latent_dim=d,
                            config=config,
                            device=device,
                            logger=logger,
                            seed=seed,
                            loss_type=loss_type,
                            batch_key=batch_key,
                            label_key=label_key,
                            requested_data_mode=requested_data_mode,
                            effective_data_mode=effective_data_mode,
                            backed_fallback_reason=backed_fallback_reason,
                            git_commit=git_commit,
                            tb_dir=tb_dir,
                        )
                        all_results.append(result)
                        dataset_results.append(result)
                        _persist_partial_results(all_results, output_path, run_dir, logger)

                        if result["status"] == "success":
                            ari_txt = f"{result['ari']:.4f}" if np.isfinite(result["ari"]) else "nan"
                            ami_txt = f"{result['ami']:.4f}" if np.isfinite(result["ami"]) else "nan"
                            logger.info(
                                f"  -> ARI={ari_txt}, AMI={ami_txt}, "
                                f"Sil={result['silhouette']:.4f}, Recon={result['reconstruction_loss']:.6f}, "
                                f"CentralityMode={result.get('centrality_mode', 'unknown')}, "
                                f"CPUmem={result['peak_cpu_mem_mb']:.1f}MB, "
                                f"Time={result['runtime_seconds']:.1f}s"
                            )

            # Save per-dataset outputs
            ds_df = pd.DataFrame(dataset_results)
            ds_results_path = os.path.join(dataset_dirs["tables"], "experiment_results.csv")
            ds_df.to_csv(ds_results_path, index=False)
            logger.info(f"  Dataset results saved: {ds_results_path} ({len(ds_df)} rows)")

            if config.experiment.auto_d_report:
                generate_auto_d_report(ds_df, config, dataset_dirs["tables"], logger)

        df = pd.DataFrame(all_results)
        out_path = output_path
        df.to_csv(out_path, index=False)
        df.to_csv(os.path.join(run_dir, "experiment_results_all.csv"), index=False)

        # Save run manifest with completion time
        run_manifest["status"] = "completed"
        run_manifest["completed_utc"] = _iso_utc(_utc_now())
        run_manifest["total_experiments"] = len(df)
        run_manifest["successful"] = int((df["status"] == "success").sum())
        run_manifest["failed"] = int((df["status"] == "failed").sum())
        _write_manifest(manifest_path, run_manifest, logger)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"EXPERIMENT COMPLETE — {run_id}")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Total runs:    {len(df)}")
        logger.info(f"  Successful:    {run_manifest['successful']}")
        logger.info(f"  Failed:        {run_manifest['failed']}")
        logger.info(f"  Completed:     {run_manifest['completed_utc']}")
        logger.info("")
        logger.info(f"  Output files:")
        logger.info(f"    Global CSV:  {out_path}")
        logger.info(f"    Run dir:     {run_dir}/")
        if tb_dir:
            n_tb = sum(1 for _, _, files in os.walk(tb_dir) for f in files if "tfevents" in f)
            logger.info(f"    TensorBoard: {tb_dir}/ ({n_tb} event files)")
            logger.info(f"    View TB:     tensorboard --logdir {os.path.dirname(tb_dir)}")
        logger.info(f"{'=' * 70}")

        if config.experiment.auto_d_report:
            generate_auto_d_report(df, config, global_dirs["tables"], logger)

    except KeyboardInterrupt:
        reason = interrupted_state["signal"] or "KeyboardInterrupt"
        logger.warning(
            "Run interrupted (%s). Writing partial outputs and interruption metadata.",
            reason,
        )
        _persist_partial_results(all_results, output_path, run_dir, logger)
        run_manifest["status"] = "interrupted"
        run_manifest["interrupted_utc"] = _iso_utc(_utc_now())
        run_manifest["interrupted_reason"] = reason
        run_manifest["total_experiments_partial"] = len(all_results)
        if all_results:
            partial_df = pd.DataFrame(all_results)
            run_manifest["successful_partial"] = int((partial_df["status"] == "success").sum())
            run_manifest["failed_partial"] = int((partial_df["status"] == "failed").sum())
        else:
            run_manifest["successful_partial"] = 0
            run_manifest["failed_partial"] = 0
        _write_manifest(manifest_path, run_manifest, logger)
        raise SystemExit(130)

    except Exception as exc:
        logger.exception("Run failed with exception. Writing partial outputs.")
        _persist_partial_results(all_results, output_path, run_dir, logger)
        run_manifest["status"] = "failed"
        run_manifest["failed_utc"] = _iso_utc(_utc_now())
        run_manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        run_manifest["total_experiments_partial"] = len(all_results)
        if all_results:
            partial_df = pd.DataFrame(all_results)
            run_manifest["successful_partial"] = int((partial_df["status"] == "success").sum())
            run_manifest["failed_partial"] = int((partial_df["status"] == "failed").sum())
        else:
            run_manifest["successful_partial"] = 0
            run_manifest["failed_partial"] = 0
        _write_manifest(manifest_path, run_manifest, logger)
        raise

    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)


if __name__ == "__main__":
    main()
