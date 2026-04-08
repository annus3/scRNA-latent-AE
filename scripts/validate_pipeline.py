#!/usr/bin/env python3
"""Validation harness for small-scope pipeline checks."""

import argparse
import os
import subprocess
import sys


def run(cmd: str) -> None:
    print(f"\n[RUN] {cmd}")
    completed = subprocess.run(cmd, shell=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    parser = argparse.ArgumentParser(description="Validate scRNA pipeline with smoke tests")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--smoke_dataset",
        type=str,
        default="pbmc3k",
        help=(
            "Dataset used for preprocess/train smoke checks. "
            "Default pbmc3k because pbmc68k_reduced is preprocessed and not raw-count suitable."
        ),
    )
    args = parser.parse_args()

    py = args.python

    run(
        f"{py} scripts/preprocess.py --config {args.config} --dataset {args.smoke_dataset}"
    )
    run(
        f"{py} scripts/train.py --config {args.config} --dataset {args.smoke_dataset} "
        "--model ae --latent_dim 4 --max_epochs 2 --device cpu --loss_type mse"
    )
    run(
        f"{py} scripts/train.py --config {args.config} --dataset {args.smoke_dataset} "
        "--model vae --latent_dim 4 --max_epochs 2 --device cpu --loss_type mse"
    )
    run(
        f"{py} scripts/run_experiment.py --config {args.config} --datasets {args.smoke_dataset} "
        "--model_types ae,vae --latent_dims 2,4,8 --max_epochs 2 --device cpu "
        "--loss_matrix ae:mse,vae:mse"
    )

    print("\nValidation pipeline completed successfully.")


if __name__ == "__main__":
    main()
