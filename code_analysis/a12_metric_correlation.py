#!/usr/bin/env python3
"""
12 — Metric Correlation & Multi-Objective Analysis

Analyzes correlations between all metrics (ARI, AMI, silhouette, reconstruction,
centrality, batch), identifies redundant vs complementary metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    display, find_repo_root, resolve_output_root,
    load_main_results, success_only, save_fig, save_table,
    section_header, dataset_label,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="12 — Metric Correlations")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "12_metric_correlation", \
        out_root / "12_metric_correlation" / "figures", \
        out_root / "12_metric_correlation" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("12 — Metric Correlation & Multi-Objective Analysis")

    df = success_only(load_main_results(repo))

    # ---- 1. Full correlation matrix ----
    section_header("1. Metric Correlation Matrix")
    metric_cols = [
        "ari", "ami", "silhouette_kmeans", "silhouette_true_labels",
        "reconstruction_loss", "best_val_loss",
        "centrality_variance", "batch_silhouette", "batch_knn_entropy",
        "total_epochs", "runtime_seconds",
    ]
    available = [c for c in metric_cols if c in df.columns]
    corr = df[available].corr(method="spearman")
    save_table(corr.reset_index(), tab_dir / "spearman_correlation_matrix.csv")

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Spearman Correlation Between All Metrics")
    save_fig(fig_dir / "metric_correlation_matrix.png")

    # ---- 2. Per-dataset correlations ----
    section_header("2. Per-Dataset ARI vs Silhouette Agreement")
    agree_rows = []
    for ds, sub in df.groupby("dataset"):
        sub_clean = sub[["ari", "silhouette_kmeans"]].dropna()
        if len(sub_clean) < 5:
            continue
        r, p = spearmanr(sub_clean["ari"], sub_clean["silhouette_kmeans"])
        agree_rows.append({
            "dataset": ds, "spearman_r": round(r, 4), "p_value": round(p, 6),
            "n": len(sub_clean),
        })
    agree_df = pd.DataFrame(agree_rows)
    display(agree_df)
    save_table(agree_df, tab_dir / "ari_silhouette_agreement.csv")

    # ---- 3. ARI vs AMI scatter ----
    section_header("3. ARI vs AMI Agreement")
    ari_ami = df[["ari", "ami"]].dropna()
    if len(ari_ami) >= 5:
        r, _ = spearmanr(ari_ami["ari"], ari_ami["ami"])
        plt.figure(figsize=(7, 6))
        plt.scatter(ari_ami["ari"], ari_ami["ami"], alpha=0.4, s=15, c="#2274A5")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
        plt.xlabel("ARI")
        plt.ylabel("AMI")
        plt.title(f"ARI vs AMI Agreement (Spearman r={r:.3f})")
        save_fig(fig_dir / "ari_vs_ami.png")

    # ---- 4. Multi-metric ranking ----
    section_header("4. Multi-Metric Model Ranking")
    rank_metrics = ["ari", "silhouette_kmeans"]
    avail_rank = [c for c in rank_metrics if c in df.columns and df[c].notna().any()]

    if avail_rank:
        model_summary = df.groupby("model_loss", as_index=False).agg(
            **{f"mean_{c}": (c, "mean") for c in avail_rank},
            n=("seed", "count"),
        )
        # Rank by each metric
        for c in avail_rank:
            model_summary[f"rank_{c}"] = model_summary[f"mean_{c}"].rank(ascending=False)

        if len(avail_rank) > 1:
            rank_cols = [f"rank_{c}" for c in avail_rank]
            model_summary["mean_rank"] = model_summary[rank_cols].mean(axis=1)
            model_summary = model_summary.sort_values("mean_rank")
        display(model_summary)
        save_table(model_summary, tab_dir / "multi_metric_ranking.csv")

    # ---- 5. Silhouette (KMeans) vs Silhouette (True Labels) ----
    section_header("5. KMeans vs True-Labels Silhouette")
    if "silhouette_true_labels" in df.columns:
        both = df[["silhouette_kmeans", "silhouette_true_labels"]].dropna()
        if len(both) >= 5:
            r, _ = spearmanr(both["silhouette_kmeans"], both["silhouette_true_labels"])
            plt.figure(figsize=(7, 6))
            plt.scatter(both["silhouette_kmeans"], both["silhouette_true_labels"],
                       alpha=0.4, s=15, c="#E85D04")
            plt.plot([-1, 1], [-1, 1], "k--", alpha=0.3)
            plt.xlabel("Silhouette (KMeans)")
            plt.ylabel("Silhouette (True Labels)")
            plt.title(f"KMeans vs True-Labels Silhouette (r={r:.3f})")
            save_fig(fig_dir / "sil_kmeans_vs_true.png")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
