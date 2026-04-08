#!/usr/bin/env python3
"""Inspect dataset schema against onboarding checklist requirements.

Mode behavior:
- --processed_name / --h5ad: read-only (loads existing files)
- --dataset with registry dataset: may trigger dataset download/cache via upstream loaders
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

import anndata as ad
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.loader import load_dataset, DATASET_REGISTRY


def _sample_matrix_values(
    matrix,
    max_cells: int,
    max_genes: int,
    seed: int,
) -> np.ndarray:
    n_obs, n_vars = matrix.shape
    rng = np.random.default_rng(seed)

    rows = np.arange(n_obs)
    cols = np.arange(n_vars)
    if n_obs > max_cells:
        rows = rng.choice(n_obs, size=max_cells, replace=False)
    if n_vars > max_genes:
        cols = rng.choice(n_vars, size=max_genes, replace=False)

    sampled = matrix[rows][:, cols]
    if hasattr(sampled, "toarray"):
        sampled = sampled.toarray()

    arr = np.asarray(sampled, dtype=np.float64)
    return arr.reshape(-1)


def _count_like_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "n_values": 0,
            "finite_fraction": float("nan"),
            "negative_fraction": float("nan"),
            "non_integer_fraction": float("nan"),
            "looks_count_like": False,
        }

    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return {
            "n_values": int(values.size),
            "finite_fraction": float(finite.mean()),
            "negative_fraction": float("nan"),
            "non_integer_fraction": float("nan"),
            "looks_count_like": False,
        }

    negative_fraction = float((finite_values < 0).mean())
    non_integer_fraction = float((np.abs(finite_values - np.round(finite_values)) > 1e-6).mean())
    looks_count_like = negative_fraction == 0.0 and non_integer_fraction <= 0.05

    return {
        "n_values": int(values.size),
        "finite_fraction": float(finite.mean()),
        "negative_fraction": negative_fraction,
        "non_integer_fraction": non_integer_fraction,
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "looks_count_like": bool(looks_count_like),
    }


def _detect_count_source(adata: ad.AnnData) -> Tuple[str, Any]:
    if "counts" in adata.layers:
        return "layers['counts']", adata.layers["counts"]

    if adata.raw is not None and getattr(adata.raw, "X", None) is not None:
        return "raw.X", adata.raw.X

    return "X", adata.X


def _load_target_adata(config, args) -> Tuple[ad.AnnData, str, str, bool]:
    if args.h5ad:
        return ad.read_h5ad(args.h5ad), f"h5ad:{args.h5ad}", "h5ad", True

    if args.processed_name:
        path = os.path.join(config.paths.data_dir, "processed", f"{args.processed_name}.h5ad")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed file not found: {path}")
        return ad.read_h5ad(path), f"processed:{path}", "processed_name", True

    if args.dataset:
        if args.dataset in DATASET_REGISTRY:
            # Registry dataset loaders may download/cache data (e.g., scanpy/scvi built-ins).
            adata = load_dataset(args.dataset, data_dir=config.paths.data_dir)
            return adata, f"registry:{args.dataset}", "dataset_registry", False

        path = os.path.join(config.paths.data_dir, "processed", f"{args.dataset}.h5ad")
        if os.path.exists(path):
            return ad.read_h5ad(path), f"processed:{path}", "dataset_processed_fallback", True

        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset '{args.dataset}' and no processed file at {path}. "
            f"Registry datasets: {available}"
        )

    raise ValueError("One of --dataset, --processed_name, or --h5ad is required")


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect dataset schema for onboarding readiness. "
            "Note: --dataset may trigger dataset download/cache via registry loaders; "
            "--processed_name/--h5ad are read-only file inspections."
        )
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Registry dataset name or processed dataset stem. "
            "Registry mode MAY download/cache data."
        ),
    )
    source_group.add_argument(
        "--processed_name",
        type=str,
        default=None,
        help="Processed dataset stem under <data_dir>/processed/<name>.h5ad (read-only).",
    )
    source_group.add_argument(
        "--h5ad",
        type=str,
        default=None,
        help="Explicit .h5ad file path (read-only).",
    )

    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--batch_key", type=str, default=None)
    parser.add_argument("--sample_cells", type=int, default=512)
    parser.add_argument("--sample_genes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Print JSON summary only")
    args = parser.parse_args()

    config = load_config(args.config)
    adata, source, source_mode, is_read_only_mode = _load_target_adata(config, args)

    label_key = args.label_key
    batch_key = args.batch_key

    count_source, count_matrix = _detect_count_source(adata)
    sampled_values = _sample_matrix_values(
        count_matrix,
        max_cells=max(1, int(args.sample_cells)),
        max_genes=max(1, int(args.sample_genes)),
        seed=int(args.seed),
    )
    count_stats = _count_like_stats(sampled_values)

    result: Dict[str, Any] = {
        "source": source,
        "source_mode": source_mode,
        "is_read_only_mode": bool(is_read_only_mode),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "obs_columns": sorted(list(map(str, adata.obs.columns))),
        "layers": sorted(list(map(str, adata.layers.keys()))),
        "label_key_requested": label_key,
        "label_key_present": bool(label_key and label_key in adata.obs.columns),
        "batch_key_requested": batch_key,
        "batch_key_present": bool(batch_key and batch_key in adata.obs.columns),
        "count_source_detected": count_source,
        "count_like_stats": count_stats,
        "preprocess_assumptions": {
            "has_cell_type_column": "cell_type" in adata.obs.columns,
            "has_counts_layer": "counts" in adata.layers,
            "has_cell_type_source_uns": bool(adata.uns.get("cell_type_source")),
        },
        "nb_zinb_ready": bool(
            ("counts" in adata.layers) and count_stats.get("looks_count_like", False)
        ),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("=== Dataset Schema Inspection ===")
    print(f"Source:             {result['source']}")
    print(f"Source mode:        {result['source_mode']}")
    print(f"Read-only mode:     {_yes_no(result['is_read_only_mode'])}")
    if not result["is_read_only_mode"]:
        print("WARNING: This mode may download/cache data via dataset registry loaders.")
    print(f"Shape:              {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Count source:       {count_source}")
    print(f"Count-like sample:  {_yes_no(count_stats.get('looks_count_like', False))}")
    print(f"NB/ZINB ready:      {_yes_no(result['nb_zinb_ready'])}")
    print("")
    print("Checklist")
    print(f"- label key present: {_yes_no(result['label_key_present'])} ({label_key})")
    print(f"- batch key present: {_yes_no(result['batch_key_present'])} ({batch_key})")
    print(f"- has obs['cell_type']: {_yes_no(result['preprocess_assumptions']['has_cell_type_column'])}")
    print(f"- has layers['counts']: {_yes_no(result['preprocess_assumptions']['has_counts_layer'])}")
    print(f"- has uns['cell_type_source']: {_yes_no(result['preprocess_assumptions']['has_cell_type_source_uns'])}")
    print("")
    print("Count sample stats")
    for k in ("n_values", "finite_fraction", "negative_fraction", "non_integer_fraction", "min", "max", "mean"):
        if k in count_stats:
            print(f"- {k}: {count_stats[k]}")


if __name__ == "__main__":
    main()
