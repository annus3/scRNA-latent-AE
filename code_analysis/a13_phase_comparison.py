#!/usr/bin/env python3
"""
13 — Phase Comparison & Data Scale Analysis

Compares results across phases and analyzes how
model performance changes with dataset scale.
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
    load_main_results, load_phase_results, load_splatter_results,
    success_only, save_fig, save_table,
    section_header, dataset_label, get_model_palette,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="13 — Phase & Scale Analysis")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "13_phase_comparison", \
        out_root / "13_phase_comparison" / "figures", \
        out_root / "13_phase_comparison" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("13 — Phase Comparison & Data Scale Analysis")

    df = success_only(load_main_results(repo))
    palette = get_model_palette(df)

    # ---- 1. Phase summary ----
    section_header("1. Phase Summary")
    phase_map = {
        "pbmc3k": "Phase 1",
        "scvi_pbmc12k": "Phase 2",
        "splatter_k04": "Phase 3",
        "splatter_k08": "Phase 3",
        "splatter_k12": "Phase 3",
        "paul15": "Phase 4",
        "ts1_all_cells": "Phase 4+",
        "ts2_lung": "Phase 4+",
        "aifi_immune_full": "Phase 4+",
    }
    df["phase"] = df["dataset"].map(phase_map).fillna("Other")
    phase_summary = df.groupby(["phase", "dataset"], as_index=False).agg(
        n_runs=("seed", "count"),
        K=("K", "first"),
        n_cells=("n_cells", "first"),
        model_types=("model_loss", "nunique"),
    )
    display(phase_summary.sort_values(["phase", "n_cells"]))
    save_table(phase_summary, tab_dir / "phase_summary.csv")

    # ---- 2. scvi:nb performance across scales ----
    section_header("2. scvi:nb Performance Across Dataset Scales")
    scvi = df[(df["model_type"] == "scvi") & (df["loss_type"] == "nb")].copy()
    scale = scvi.groupby("dataset", as_index=False).agg(
        K=("K", "first"),
        n_cells=("n_cells", "first"),
        mean_ari=("ari", "mean"),
        mean_sil=("silhouette_kmeans", "mean"),
        mean_runtime=("runtime_seconds", "mean"),
    ).sort_values("n_cells")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(scale["n_cells"], scale["mean_ari"], "o-", color="#2274A5", markersize=8)
    for _, r in scale.iterrows():
        axes[0].annotate(dataset_label(r["dataset"]), (r["n_cells"], r["mean_ari"]),
                        fontsize=7, xytext=(5, 5), textcoords="offset points")
    axes[0].set_xlabel("Number of Cells")
    axes[0].set_ylabel("Mean ARI")
    axes[0].set_title("ARI vs Dataset Scale")
    axes[0].set_xscale("log")

    axes[1].plot(scale["n_cells"], scale["mean_sil"], "o-", color="#E85D04", markersize=8)
    axes[1].set_xlabel("Number of Cells")
    axes[1].set_ylabel("Mean Silhouette")
    axes[1].set_title("Silhouette vs Dataset Scale")
    axes[1].set_xscale("log")

    axes[2].plot(scale["n_cells"], scale["mean_runtime"], "o-", color="#7B2D8E", markersize=8)
    axes[2].set_xlabel("Number of Cells")
    axes[2].set_ylabel("Mean Runtime (s)")
    axes[2].set_title("Runtime vs Dataset Scale")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")

    plt.suptitle("scvi:nb — Scaling Behavior", fontsize=13)
    plt.tight_layout()
    save_fig(fig_dir / "scvi_nb_scaling.png")
    save_table(scale, tab_dir / "scvi_nb_scaling.csv")

    # ---- 3. Model rank stability across scales ----
    section_header("3. Model Rank Stability Across Dataset Scale")
    rank_by_ds = []
    for ds, sub in df.groupby("dataset"):
        ds_rank = sub.groupby("model_loss", as_index=False).agg(
            mean_score=("ari" if sub["ari"].notna().any() else "silhouette_kmeans", "mean"),
        )
        ds_rank["rank"] = ds_rank["mean_score"].rank(ascending=False)
        ds_rank["dataset"] = ds
        ds_rank["n_cells"] = int(sub["n_cells"].iloc[0])
        rank_by_ds.append(ds_rank)
    rank_df = pd.concat(rank_by_ds, ignore_index=True)
    save_table(rank_df, tab_dir / "model_rank_by_dataset.csv")

    # Rank evolution plot
    rank_pivot = rank_df.pivot_table(index="dataset", columns="model_loss", values="rank")
    if rank_pivot.shape[0] >= 2:
        plt.figure(figsize=(12, 6))
        sns.heatmap(rank_pivot, annot=True, fmt=".0f", cmap="YlOrRd_r", linewidths=0.5)
        plt.title("Model Ranking by Dataset (1 = best)")
        plt.ylabel("")
        save_fig(fig_dir / "model_rank_heatmap.png")

    # ---- 4. Splatter vs Real data trend comparison ----
    section_header("4. Splatter vs Real Data d-Trend")
    try:
        splatter = success_only(load_splatter_results(repo))
        splatter["model_loss"] = splatter["model_type"].astype(str) + ":" + splatter["loss_type"].astype(str)
        splatter["score"] = np.where(splatter["ari"].notna(), splatter["ari"], splatter["silhouette_kmeans"])
        # Recommended d per K for scvi:nb
        synth_rec = []
        for k, sub in splatter[splatter["model_loss"] == "scvi:nb"].groupby("K"):
            by_d = sub.groupby("latent_dim")["score"].mean()
            synth_rec.append({"K": k, "best_d": int(by_d.idxmax()), "source": "Splatter"})

        real_rec = []
        scvi_real = df[(df["model_type"] == "scvi") & (df["loss_type"] == "nb")]
        for ds, sub in scvi_real.groupby("dataset"):
            score_col = "ari" if sub["ari"].notna().any() else "silhouette_kmeans"
            by_d = sub.groupby("latent_dim")[score_col].mean()
            real_rec.append({"K": int(sub["K"].iloc[0]), "best_d": int(by_d.idxmax()),
                             "source": "Real"})

        cmp = pd.DataFrame(synth_rec + real_rec)
        if len(cmp) >= 3:
            plt.figure(figsize=(8, 5))
            for src, grp in cmp.groupby("source"):
                grp = grp.sort_values("K")
                marker = "s" if src == "Splatter" else "o"
                plt.plot(grp["K"], grp["best_d"], marker + "-", label=src, markersize=8, linewidth=2)
            plt.xlabel("K")
            plt.ylabel("Recommended d")
            plt.title("Synthetic vs Real Data: d = f(K) Comparison (scvi:nb)")
            plt.legend()
            save_fig(fig_dir / "splatter_vs_real_d_trend.png")
            save_table(cmp, tab_dir / "splatter_vs_real_d_comparison.csv")
    except FileNotFoundError:
        print("  Splatter results not found. Skipping comparison.")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
