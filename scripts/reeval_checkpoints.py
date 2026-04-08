#!/usr/bin/env python3
"""
Re-evaluate existing model checkpoints to compute new metrics
(Calinski-Harabasz, Davies-Bouldin, trustworthiness, continuity)
without retraining.

Usage:
    python scripts/reeval_checkpoints.py [--checkpoint-dir checkpoints] [--output reeval_results.csv]
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import split_adata, make_dataloaders
from src.evaluation.metrics import (
    evaluate_latent,
    extract_latent,
    extract_original_data,
)
from src.models.autoencoder import Autoencoder
from src.models.vae import VAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Regex to parse checkpoint filenames: {dataset}_{model}_{loss}_d{dim}_s{seed}.pt
CKPT_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<model>ae|vae)_(?P<loss>mse|nb|zinb)_d(?P<dim>\d+)_s(?P<seed>\d+)\.pt$"
)


def load_processed_adata(dataset_name: str):
    """Load processed h5ad for a dataset."""
    import anndata as ad

    path = PROJECT_ROOT / "data" / "processed" / f"{dataset_name}.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    return ad.read_h5ad(str(path))


def build_model(model_type: str, loss_type: str, input_dim: int, latent_dim: int,
                hidden_dims=None, activation="relu", dropout=0.0):
    """Build an AE or VAE model matching training config."""
    if hidden_dims is None:
        hidden_dims = [128, 64]
    if model_type == "ae":
        return Autoencoder(input_dim, latent_dim, hidden_dims, activation, dropout, loss_type)
    elif model_type == "vae":
        return VAE(input_dim, latent_dim, hidden_dims, activation, dropout, loss_type)
    else:
        raise ValueError(f"Unsupported model_type for checkpoint re-eval: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate checkpoints for new metrics")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "results" / "global" / "tables" / "reeval_new_metrics.csv"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--neighborhood-max-cells", type=int, default=5000)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    logger.info(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")

    # Parse all checkpoint filenames
    tasks = []
    for f in ckpt_files:
        m = CKPT_RE.match(f.name)
        if m:
            tasks.append({
                "path": f,
                "dataset": m.group("dataset"),
                "model_type": m.group("model"),
                "loss_type": m.group("loss"),
                "latent_dim": int(m.group("dim")),
                "seed": int(m.group("seed")),
            })
    logger.info(f"Parsed {len(tasks)} valid checkpoints")

    # Group by dataset to avoid reloading data
    from collections import defaultdict
    by_dataset = defaultdict(list)
    for t in tasks:
        by_dataset[t["dataset"]].append(t)

    results = []
    for ds_name, ds_tasks in by_dataset.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name} ({len(ds_tasks)} checkpoints)")
        logger.info(f"{'='*60}")

        try:
            adata = load_processed_adata(ds_name)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {ds_name}: {e}")
            continue

        input_dim = adata.n_vars
        label_key = None
        for candidate in ["cell_type", "louvain", "leiden", "cluster"]:
            if candidate in adata.obs.columns:
                label_key = candidate
                break

        if label_key is None:
            logger.warning(f"No label key found for {ds_name}, skipping")
            continue

        true_labels_all = adata.obs[label_key].values
        n_clusters = len(np.unique(true_labels_all))

        for task in ds_tasks:
            seed = task["seed"]
            latent_dim = task["latent_dim"]
            model_type = task["model_type"]
            loss_type = task["loss_type"]
            ckpt_path = task["path"]

            tag = f"{ds_name}/{model_type}:{loss_type}/d={latent_dim}/s={seed}"
            logger.info(f"  Re-evaluating: {tag}")
            t0 = time.time()

            try:
                # Split with the same seed used during training
                _, _, adata_test = split_adata(adata, args.train_frac, args.val_frac, seed=seed)

                # Build model and load checkpoint
                model = build_model(model_type, loss_type, input_dim, latent_dim)
                state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

                # Data layer for NB/ZINB
                data_layer = "counts" if loss_type in {"nb", "zinb"} else None

                _, _, test_loader = make_dataloaders(
                    adata_test, adata_test, adata_test,  # only test_loader matters
                    batch_size=args.batch_size,
                    layer=data_layer,
                )

                # Extract latent representations
                latent = extract_latent(model, test_loader, device)

                # Get original high-dim test data
                import scipy.sparse as sp
                X_test = adata_test.X
                original_data = np.asarray(X_test.toarray() if sp.issparse(X_test) else X_test)

                # Get true labels for test split
                test_labels = adata_test.obs[label_key].values

                # Run evaluation with new metrics
                metrics = evaluate_latent(
                    latent=latent,
                    n_clusters=n_clusters,
                    true_labels=test_labels,
                    seed=seed,
                    original_data=original_data,
                    neighborhood_max_cells=args.neighborhood_max_cells,
                )

                elapsed = time.time() - t0
                row = {
                    "dataset": ds_name,
                    "model_type": model_type,
                    "loss_type": loss_type,
                    "latent_dim": latent_dim,
                    "seed": seed,
                    "calinski_harabasz": metrics["calinski_harabasz"],
                    "davies_bouldin": metrics["davies_bouldin"],
                    "trustworthiness": metrics["trustworthiness"],
                    "continuity": metrics["continuity"],
                    "ari": metrics["ari"],
                    "ami": metrics["ami"],
                    "silhouette_kmeans": metrics["silhouette_kmeans"],
                    "reeval_time_s": round(elapsed, 2),
                    "status": "success",
                }
                results.append(row)
                logger.info(
                    f"    CH={metrics['calinski_harabasz']:.1f}, "
                    f"DB={metrics['davies_bouldin']:.3f}, "
                    f"Trust={metrics['trustworthiness']:.4f}, "
                    f"Cont={metrics['continuity']:.4f} "
                    f"({elapsed:.1f}s)"
                )

            except Exception as exc:
                logger.error(f"    FAILED: {exc}", exc_info=True)
                results.append({
                    "dataset": ds_name,
                    "model_type": model_type,
                    "loss_type": loss_type,
                    "latent_dim": latent_dim,
                    "seed": seed,
                    "calinski_harabasz": float("nan"),
                    "davies_bouldin": float("nan"),
                    "trustworthiness": float("nan"),
                    "continuity": float("nan"),
                    "ari": float("nan"),
                    "ami": float("nan"),
                    "silhouette_kmeans": float("nan"),
                    "reeval_time_s": float("nan"),
                    "status": f"failed: {exc}",
                })

    if results:
        df = pd.DataFrame(results)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"\nSaved {len(df)} results to {out_path}")
        logger.info(f"  Success: {(df['status'] == 'success').sum()}")
        logger.info(f"  Failed:  {(df['status'] != 'success').sum()}")

        # Print summary
        success = df[df["status"] == "success"]
        if len(success):
            print("\n" + "=" * 70)
            print("RE-EVALUATION SUMMARY (new metrics from existing checkpoints)")
            print("=" * 70)
            summary = success.groupby(["dataset", "model_type", "loss_type"]).agg(
                n=("seed", "count"),
                mean_CH=("calinski_harabasz", "mean"),
                mean_DB=("davies_bouldin", "mean"),
                mean_trust=("trustworthiness", "mean"),
                mean_cont=("continuity", "mean"),
            ).round(4)
            print(summary.to_string())
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    main()
