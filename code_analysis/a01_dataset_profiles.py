#!/usr/bin/env python3
"""
01 — Per-Dataset Profile & Characterization

Deep profile of each dataset: cell counts, gene counts, K, zero fraction,
label trust, batch structure, and preprocessing fingerprint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    display, find_repo_root, resolve_output_root,
    load_main_results, success_only, save_fig, save_table,
    section_header, dataset_label, PALETTE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="01 — Per-Dataset Profiles")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir = out_root / "01_dataset_profiles"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("01 — Per-Dataset Profile & Characterization")

    df = success_only(load_main_results(repo))

    # ---- 1. Dataset summary table ----
    section_header("1. Dataset Summary Table")
    ds_profile = df.groupby("dataset", as_index=False).agg(
        K=("K", "first"),
        n_cells=("n_cells", "first"),
        n_genes=("n_genes", "first"),
        n_runs=("status", "count"),
        n_seeds=("seed", "nunique"),
        label_trust=("label_trust", "first"),
        label_source=("label_source", "first"),
        n_batches=("n_batches", lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan),
        zero_fraction_mean=("zero_fraction", "mean"),
        external_metrics=("external_metrics_enabled", "first"),
        latent_dims_tested=("latent_dim", lambda x: len(x.unique())),
        model_loss_combos=("model_loss", lambda x: len(x.unique())),
    ).sort_values("K")

    ds_profile["display_name"] = ds_profile["dataset"].map(dataset_label)
    display(ds_profile)
    save_table(ds_profile, tab_dir / "dataset_summary.csv")

    # ---- 2. Dataset comparison bar chart ----
    section_header("2. Dataset Scale Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ds_sorted = ds_profile.sort_values("n_cells")

    axes[0].barh(ds_sorted["display_name"], ds_sorted["n_cells"], color="#2274A5")
    axes[0].set_xlabel("Number of Cells")
    axes[0].set_title("Cell Count")
    axes[0].set_xscale("log")

    axes[1].barh(ds_sorted["display_name"], ds_sorted["K"], color="#E85D04")
    axes[1].set_xlabel("K (Cell Types)")
    axes[1].set_title("Biological Complexity")

    axes[2].barh(ds_sorted["display_name"], ds_sorted["zero_fraction_mean"], color="#7B2D8E")
    axes[2].set_xlabel("Zero Fraction")
    axes[2].set_title("Data Sparsity")
    axes[2].set_xlim(0, 1)

    plt.suptitle("Dataset Characteristics", fontsize=14, fontweight="bold")
    save_fig(fig_dir / "dataset_scale_comparison.png")

    # ---- 3. Per-dataset metric distributions ----
    section_header("3. Per-Dataset Metric Distributions")
    metrics_long = []
    for metric in ["ari", "silhouette_kmeans", "reconstruction_loss"]:
        if metric in df.columns:
            tmp = df[["dataset", "model_loss", "latent_dim", metric]].copy()
            tmp = tmp.rename(columns={metric: "value"})
            tmp["metric"] = metric
            metrics_long.append(tmp)

    if metrics_long:
        metrics_df = pd.concat(metrics_long, ignore_index=True)
        metrics_df = metrics_df.dropna(subset=["value"])

        g = sns.catplot(
            data=metrics_df, x="dataset", y="value", col="metric",
            kind="box", sharey=False, col_wrap=2, height=4, aspect=1.5,
        )
        g.set_xticklabels(rotation=45, ha="right")
        g.fig.suptitle("Metric Distributions by Dataset", y=1.02)
        g.fig.savefig(fig_dir / "metric_distributions_by_dataset.png", dpi=300, bbox_inches="tight")
        plt.close(g.fig)
        print(f"  [fig] metric_distributions_by_dataset.png")

    # ---- 4. Per-dataset model coverage heatmap ----
    section_header("4. Model/Loss Coverage per Dataset")
    cov = df.groupby(["dataset", "model_loss"]).size().reset_index(name="n_runs")
    cov_pivot = cov.pivot(index="dataset", columns="model_loss", values="n_runs").fillna(0).astype(int)

    plt.figure(figsize=(12, 5))
    sns.heatmap(cov_pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5)
    plt.title("Experiment Coverage: Runs per Dataset × Model/Loss")
    plt.ylabel("")
    save_fig(fig_dir / "coverage_heatmap.png")

    # ---- 5. Label trust distribution ----
    section_header("5. Label Trust Distribution")
    trust_df = df.groupby(["dataset", "label_trust"]).size().reset_index(name="count")
    trust_pivot = trust_df.pivot(index="dataset", columns="label_trust", values="count").fillna(0).astype(int)
    save_table(trust_pivot.reset_index(), tab_dir / "label_trust_matrix.csv")

    print(f"\n  Saved to: {out_root / '01_dataset_profiles'}")


if __name__ == "__main__":
    main()
