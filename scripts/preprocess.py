#!/usr/bin/env python3
"""
CLI: Preprocess a scRNA-seq dataset.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.utils.logging_utils import setup_logger
from src.data.loader import (
    load_dataset,
    subsample_cell_types,
    resolve_label_key,
    resolve_batch_key,
)
from src.data.preprocessor import preprocess, save_processed


def main():
    parser = argparse.ArgumentParser(description="Preprocess scRNA-seq dataset")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subsample_k", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch_key", type=str, default=None)
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--scale_policy", type=str, default=None, choices=["none", "pca_only", "full"])
    parser.add_argument("--use_backed_mode", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_dirs()
    dataset_dirs = config.ensure_dataset_dirs(args.dataset)

    if args.scale_policy:
        config.preprocessing.scale_policy = args.scale_policy
    if args.batch_key:
        config.data.batch_key = args.batch_key
    if args.label_key:
        config.data.label_key = args.label_key
    if args.use_backed_mode:
        config.data.use_backed_mode = True

    logger = setup_logger("preprocess", dataset_dirs["logs"])
    logger.info(f"=== Preprocessing: {args.dataset} ===")

    adata = load_dataset(args.dataset, data_dir=config.paths.data_dir)
    label_key = resolve_label_key(adata, config.data.label_key, args.dataset)
    batch_key = resolve_batch_key(adata, config.data.batch_key, args.dataset)

    if args.subsample_k is not None:
        if label_key is None:
            logger.error(
                "Cannot subsample without label key. Use --label_key or a dataset with known labels."
            )
            sys.exit(1)
        adata = subsample_cell_types(adata, label_key, k=args.subsample_k, seed=config.seed)

    adata = preprocess(
        adata,
        config.preprocessing,
        cell_type_key=label_key,
        batch_key=batch_key,
    )

    if args.output:
        out_path = args.output
    else:
        suffix = f"_k{args.subsample_k}" if args.subsample_k else ""
        out_path = os.path.join(config.paths.data_dir, "processed", f"{args.dataset}{suffix}.h5ad")

    save_processed(adata, out_path)
    K = adata.obs["cell_type"].nunique()
    logger.info(
        f"=== Done. {adata.n_obs} cells, {adata.n_vars} genes, K={K}, "
        f"label_key={label_key}, batch_key={batch_key} ==="
    )


if __name__ == "__main__":
    main()
