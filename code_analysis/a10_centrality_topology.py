#!/usr/bin/env python3
"""
10 — Centrality & Latent Topology Analysis

Analyzes centrality_variance metric: relationship with d, correlation
with ARI/silhouette, and dataset-specific patterns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    display, find_repo_root, resolve_output_root,
    load_main_results, success_only, save_fig, save_table,
    section_header, dataset_label, get_model_palette,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="10 — Centrality & Topology")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "10_centrality_topology", \
        out_root / "10_centrality_topology" / "figures", \
        out_root / "10_centrality_topology" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("10 — Centrality & Latent Topology Analysis")

    df = success_only(load_main_results(repo))
    palette = get_model_palette(df)

    # ---- 1. Centrality variance availability ----
    section_header("1. Centrality Metrics Availability")
    if "centrality_variance" not in df.columns:
        print("  centrality_variance column not found. Exiting.")
        return

    cv_avail = df.groupby("dataset").agg(
        total=("centrality_variance", "count"),
        non_null=("centrality_variance", lambda x: x.notna().sum()),
    ).reset_index()
    cv_avail["pct_available"] = (cv_avail["non_null"] / cv_avail["total"] * 100).round(1)
    display(cv_avail)
    save_table(cv_avail, tab_dir / "centrality_availability.csv")

    # Filter to rows with centrality
    has_cv = df[df["centrality_variance"].notna()].copy()
    if len(has_cv) == 0:
        print("  No centrality_variance values. Exiting.")
        return

    # Centrality mode distribution
    if "centrality_mode" in has_cv.columns:
        mode_dist = has_cv["centrality_mode"].value_counts().to_dict()
        print(f"  Centrality modes: {mode_dist}")

    # ---- 2. Centrality variance vs latent dim ----
    section_header("2. Centrality Variance vs Latent Dimension")
    cv_agg = has_cv.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
        mean_cv=("centrality_variance", "mean"),
        std_cv=("centrality_variance", "std"),
    )

    datasets = cv_agg["dataset"].unique()
    n = len(datasets)
    cols = min(3, n)
    rows_n = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 4.5 * rows_n), squeeze=False)
    for i, ds in enumerate(sorted(datasets)):
        ax = axes[i // cols][i % cols]
        sub = cv_agg[cv_agg["dataset"] == ds]
        for ml in sorted(sub["model_loss"].unique()):
            ml_sub = sub[sub["model_loss"] == ml].sort_values("latent_dim")
            color = palette.get(ml, "#999999")
            ax.plot(ml_sub["latent_dim"], ml_sub["mean_cv"], "o-", label=ml,
                    color=color, markersize=4)
        ax.set_title(dataset_label(ds), fontsize=10)
        ax.set_xlabel("Latent Dim (d)")
        ax.set_ylabel("Centrality Variance")
        ax.legend(fontsize=6, ncol=2)
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)
    plt.suptitle("Centrality Variance vs Latent Dimension", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(fig_dir / "centrality_vs_d.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] centrality_vs_d.png")

    save_table(cv_agg, tab_dir / "centrality_vs_d.csv")

    # ---- 3. Centrality vs ARI correlation ----
    section_header("3. Centrality vs ARI Correlation")
    corr_df = has_cv[has_cv["ari"].notna()].copy()
    if len(corr_df) >= 5:
        r, p = stats.spearmanr(corr_df["centrality_variance"], corr_df["ari"])
        print(f"  Spearman correlation (centrality_variance vs ARI): r={r:.4f}, p={p:.6f}")

        plt.figure(figsize=(8, 6))
        for ml in sorted(corr_df["model_loss"].unique()):
            sub = corr_df[corr_df["model_loss"] == ml]
            color = palette.get(ml, "#999999")
            plt.scatter(sub["centrality_variance"], sub["ari"], label=ml,
                       color=color, alpha=0.6, s=30)
        plt.xlabel("Centrality Variance")
        plt.ylabel("ARI")
        plt.title(f"Centrality Variance vs ARI (Spearman r={r:.3f}, p={p:.4f})")
        plt.legend(fontsize=7)
        save_fig(fig_dir / "centrality_vs_ari.png")

        # Per-dataset correlation
        corr_rows = []
        for ds, sub in corr_df.groupby("dataset"):
            if len(sub) < 5:
                continue
            r_ds, p_ds = stats.spearmanr(sub["centrality_variance"], sub["ari"])
            corr_rows.append({"dataset": ds, "spearman_r": round(r_ds, 4),
                              "p_value": round(p_ds, 6), "n": len(sub)})
        save_table(pd.DataFrame(corr_rows), tab_dir / "centrality_ari_correlation.csv")

    # ---- 4. Centrality vs Silhouette ----
    section_header("4. Centrality vs Silhouette")
    sil_corr = has_cv[has_cv["silhouette_kmeans"].notna()].copy()
    if len(sil_corr) >= 5:
        r, p = stats.spearmanr(sil_corr["centrality_variance"], sil_corr["silhouette_kmeans"])
        print(f"  Spearman (centrality vs silhouette): r={r:.4f}, p={p:.6f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(sil_corr["centrality_variance"], sil_corr["silhouette_kmeans"],
                   alpha=0.4, s=20, c="#2274A5")
        plt.xlabel("Centrality Variance")
        plt.ylabel("Silhouette (KMeans)")
        plt.title(f"Centrality vs Silhouette (r={r:.3f})")
        save_fig(fig_dir / "centrality_vs_silhouette.png")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
