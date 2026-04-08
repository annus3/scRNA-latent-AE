#!/usr/bin/env python3
"""
04 — Per-Dataset Architecture Deep-Dive

For each dataset, generates detailed architecture comparison:
ARI/silhouette vs d curves, reconstruction trends, and optimal d analysis.
This addresses the gap: individual datasets analyzed w.r.t. their own architectures.
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
    section_header, dataset_label, get_model_palette, MODEL_ORDER,
)


def analyze_dataset(df_ds: pd.DataFrame, ds_name: str, fig_dir: Path, tab_dir: Path, palette: dict):
    """Produce per-dataset analysis outputs."""
    section_header(f"Dataset: {dataset_label(ds_name)}")
    K = int(df_ds["K"].iloc[0])
    n_cells = int(df_ds["n_cells"].iloc[0])
    trust = df_ds["label_trust"].iloc[0]
    print(f"  K={K}, n_cells={n_cells}, label_trust={trust}")
    print(f"  Model/loss combos: {sorted(df_ds['model_loss'].unique())}")
    print(f"  Latent dims: {sorted(df_ds['latent_dim'].unique())}")
    print(f"  Seeds: {sorted(df_ds['seed'].unique())}")

    # 1. ARI vs d with error bars
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("ari", "ARI"), ("silhouette_kmeans", "Silhouette (KMeans)"),
               ("reconstruction_loss", "Reconstruction Loss")]

    for ax, (metric, title) in zip(axes, metrics):
        if metric not in df_ds.columns or df_ds[metric].isna().all():
            ax.set_title(f"{title}\n(not available)")
            continue
        agg = df_ds.groupby(["model_loss", "latent_dim"], as_index=False).agg(
            mean=(metric, "mean"), std=(metric, "std"), n=("seed", "count"),
        )
        for ml in sorted(agg["model_loss"].unique(), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99):
            sub = agg[agg["model_loss"] == ml].sort_values("latent_dim")
            color = palette.get(ml, "#999999")
            ax.errorbar(sub["latent_dim"], sub["mean"], yerr=sub["std"],
                       fmt="o-", label=ml, color=color, markersize=4, linewidth=1.2,
                       capsize=3, capthick=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Latent Dim (d)")
        ax.legend(fontsize=7, ncol=2)

    plt.suptitle(f"{dataset_label(ds_name)} — Architecture Comparison (K={K})", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / f"{ds_name}_architecture_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {ds_name}_architecture_comparison.png")

    # 2. Detailed table
    detail = df_ds.groupby(["model_loss", "latent_dim"], as_index=False).agg(
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        ami_mean=("ami", "mean"),
        sil_mean=("silhouette_kmeans", "mean"), sil_std=("silhouette_kmeans", "std"),
        recon_mean=("reconstruction_loss", "mean"), recon_std=("reconstruction_loss", "std"),
        n_seeds=("seed", "nunique"),
    ).sort_values(["model_loss", "latent_dim"])
    save_table(detail, tab_dir / f"{ds_name}_detail.csv")

    # 3. Best d per model for this dataset
    best = []
    for ml, sub in df_ds.groupby("model_loss"):
        # Use ARI if available, else silhouette
        metric_col = "ari" if sub["ari"].notna().any() else "silhouette_kmeans"
        by_d = sub.groupby("latent_dim", as_index=False)[metric_col].mean()
        by_d = by_d.sort_values(metric_col, ascending=False)
        if by_d.empty:
            continue
        best.append({
            "model_loss": ml, "best_d": int(by_d.iloc[0]["latent_dim"]),
            "best_score": round(by_d.iloc[0][metric_col], 4),
            "metric_used": metric_col,
        })
    best_df = pd.DataFrame(best)
    save_table(best_df, tab_dir / f"{ds_name}_best_d.csv")

    # 4. Model ranking for this dataset
    ranking = df_ds.groupby("model_loss", as_index=False).agg(
        mean_ari=("ari", "mean"), mean_sil=("silhouette_kmeans", "mean"),
        mean_recon=("reconstruction_loss", "mean"),
    ).sort_values("mean_ari" if df_ds["ari"].notna().any() else "mean_sil", ascending=False)
    save_table(ranking, tab_dir / f"{ds_name}_model_ranking.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="04 — Per-Dataset Analysis")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir = out_root / "04_per_dataset"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("04 — Per-Dataset Architecture Deep-Dive")

    df = success_only(load_main_results(repo))
    palette = get_model_palette(df)

    for ds_name in sorted(df["dataset"].unique()):
        df_ds = df[df["dataset"] == ds_name].copy()
        analyze_dataset(df_ds, ds_name, fig_dir, tab_dir, palette)

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
