#!/usr/bin/env python3
"""
05 — Batch Effect Analysis

Analyzes batch correction quality: batch silhouette, batch KNN entropy,
and the biological-vs-batch trade-off across models.
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
    section_header, dataset_label, get_model_palette,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="05 — Batch Effect Analysis")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "05_batch_effects", \
        out_root / "05_batch_effects" / "figures", \
        out_root / "05_batch_effects" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("05 — Batch Effect Analysis")

    df = success_only(load_main_results(repo))
    palette = get_model_palette(df)

    # ---- 1. Batch metrics availability ----
    section_header("1. Batch Metrics Availability")
    batch_avail = df.groupby("dataset")["batch_metrics_mode"].value_counts(dropna=False).rename("count").reset_index()
    display(batch_avail)
    save_table(batch_avail, tab_dir / "batch_metrics_availability.csv")

    # Filter to datasets with batch metrics
    has_batch = df[
        df["batch_metrics_mode"].isin(["computed", "full", "sampled"]) |
        (df["batch_silhouette"].notna() & np.isfinite(df["batch_silhouette"]))
    ].copy()

    if len(has_batch) == 0:
        print("  No batch metrics available in any dataset. Generating n_batches report only.")
        nbatch = df.groupby("dataset")["n_batches"].first().reset_index()
        save_table(nbatch, tab_dir / "n_batches_by_dataset.csv")
        print(f"\n  Saved to: {out_dir}")
        return

    # ---- 2. Batch metrics summary table ----
    section_header("2. Batch Metrics Summary")
    batch_summary = has_batch.groupby(["dataset", "model_loss"], as_index=False).agg(
        mean_batch_sil=("batch_silhouette", "mean"),
        mean_batch_mixing=("batch_silhouette_mixing", "mean"),
        mean_batch_entropy=("batch_knn_entropy", "mean"),
        n_batches=("n_batches", "first"),
        n=("seed", "count"),
    )
    display(batch_summary)
    save_table(batch_summary, tab_dir / "batch_metrics_summary.csv")

    # ---- 3. Batch mixing vs latent dim ----
    section_header("3. Batch Mixing vs Latent Dimension")
    batch_by_d = has_batch.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
        mean_mixing=("batch_silhouette_mixing", "mean"),
        mean_entropy=("batch_knn_entropy", "mean"),
    )

    datasets_with_batch = batch_by_d["dataset"].unique()
    for ds in datasets_with_batch:
        sub = batch_by_d[batch_by_d["dataset"] == ds]
        if len(sub) < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ml in sorted(sub["model_loss"].unique()):
            ml_sub = sub[sub["model_loss"] == ml].sort_values("latent_dim")
            color = palette.get(ml, "#999999")
            axes[0].plot(ml_sub["latent_dim"], ml_sub["mean_mixing"], "o-",
                        label=ml, color=color, markersize=4)
            axes[1].plot(ml_sub["latent_dim"], ml_sub["mean_entropy"], "o-",
                        label=ml, color=color, markersize=4)

        axes[0].set_title("Batch Silhouette Mixing")
        axes[0].set_xlabel("Latent Dim (d)")
        axes[0].legend(fontsize=7)
        axes[1].set_title("Batch KNN Entropy (higher = better mixing)")
        axes[1].set_xlabel("Latent Dim (d)")
        axes[1].legend(fontsize=7)
        plt.suptitle(f"Batch Mixing — {dataset_label(ds)}", fontsize=12)
        plt.tight_layout()
        fig.savefig(fig_dir / f"batch_mixing_{ds}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig] batch_mixing_{ds}.png")

    # ---- 4. Bio vs Batch trade-off ----
    section_header("4. Biological vs Batch Trade-off")
    tradeoff = has_batch.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
        mean_ari=("ari", "mean"),
        mean_sil=("silhouette_kmeans", "mean"),
        mean_batch_mixing=("batch_silhouette_mixing", "mean"),
    )
    tradeoff["bio_score"] = np.where(tradeoff["mean_ari"].notna(), tradeoff["mean_ari"], tradeoff["mean_sil"])

    if tradeoff["bio_score"].notna().any() and tradeoff["mean_batch_mixing"].notna().any():
        plt.figure(figsize=(10, 6))
        for ml in sorted(tradeoff["model_loss"].unique()):
            sub = tradeoff[tradeoff["model_loss"] == ml]
            color = palette.get(ml, "#999999")
            plt.scatter(sub["mean_batch_mixing"], sub["bio_score"],
                       label=ml, color=color, alpha=0.7, s=50)
        plt.xlabel("Batch Mixing Score (higher = better integration)")
        plt.ylabel("Biological Score (ARI or Silhouette)")
        plt.title("Biological Accuracy vs Batch Integration Trade-off")
        plt.legend(fontsize=8)
        save_fig(fig_dir / "bio_vs_batch_tradeoff.png")

    save_table(tradeoff, tab_dir / "bio_vs_batch_tradeoff.csv")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
