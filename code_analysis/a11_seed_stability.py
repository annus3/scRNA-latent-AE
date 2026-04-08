#!/usr/bin/env python3
"""
11 — Seed Stability & Reproducibility Analysis

Analyzes variance across seeds: coefficient of variation, agreement rates,
and identifies unstable experiments.
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
    parser = argparse.ArgumentParser(description="11 — Seed Stability")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "11_seed_stability", \
        out_root / "11_seed_stability" / "figures", \
        out_root / "11_seed_stability" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("11 — Seed Stability & Reproducibility")

    df = success_only(load_main_results(repo))
    palette = get_model_palette(df)

    # ---- 1. Seed variance summary ----
    section_header("1. Seed Variance Summary")
    stab = df.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        sil_mean=("silhouette_kmeans", "mean"), sil_std=("silhouette_kmeans", "std"),
        recon_mean=("reconstruction_loss", "mean"), recon_std=("reconstruction_loss", "std"),
    )
    # CV = std/mean (coefficient of variation)
    stab["ari_cv"] = np.where(stab["ari_mean"] > 0, stab["ari_std"] / stab["ari_mean"], np.nan)
    stab["sil_cv"] = np.where(stab["sil_mean"] > 0, stab["sil_std"] / stab["sil_mean"], np.nan)
    stab["recon_cv"] = np.where(stab["recon_mean"] > 0, stab["recon_std"] / stab["recon_mean"], np.nan)

    save_table(stab, tab_dir / "seed_variance_detail.csv")

    # ---- 2. Average CV by model ----
    section_header("2. Average CV by Model")
    cv_by_model = stab.groupby("model_loss", as_index=False).agg(
        mean_ari_cv=("ari_cv", "mean"),
        mean_sil_cv=("sil_cv", "mean"),
        mean_recon_cv=("recon_cv", "mean"),
        n_configs=("latent_dim", "count"),
    )
    display(cv_by_model.sort_values("mean_ari_cv"))
    save_table(cv_by_model, tab_dir / "cv_by_model.csv")

    # ---- 3. CV distribution by model ----
    section_header("3. CV Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, title) in zip(axes, [
        ("ari_cv", "ARI CV"), ("sil_cv", "Silhouette CV"), ("recon_cv", "Reconstruction CV")
    ]):
        data = stab[stab[metric].notna() & np.isfinite(stab[metric])]
        if len(data):
            sns.boxplot(data=data, x="model_loss", y=metric, ax=ax, palette=palette)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(title)
    plt.suptitle("Seed Stability: Coefficient of Variation", fontsize=13)
    plt.tight_layout()
    save_fig(fig_dir / "cv_distributions.png")

    # ---- 4. Unstable configurations (high CV) ----
    section_header("4. Unstable Configurations")
    unstable = stab[
        (stab["ari_cv"] > 0.5) |
        (stab["sil_cv"] > 0.5) |
        (stab["recon_cv"] > 0.5)
    ].copy()
    if len(unstable):
        print(f"  {len(unstable)} configurations with CV > 0.5:")
        display(unstable.head(20))
    else:
        print("  All configurations have CV ≤ 0.5 (stable).")
    save_table(unstable, tab_dir / "unstable_configs.csv")

    # ---- 5. Best-d agreement across seeds ----
    section_header("5. Best-d Agreement Across Seeds")
    agree_rows = []
    for (ds, ml), sub in df.groupby(["dataset", "model_loss"]):
        metric_col = "ari" if sub["ari"].notna().any() else "silhouette_kmeans"
        best_d_per_seed = []
        for seed, seed_sub in sub.groupby("seed"):
            by_d = seed_sub.groupby("latent_dim")[metric_col].mean()
            if len(by_d) == 0:
                continue
            best_d_per_seed.append(int(by_d.idxmax()))

        if len(best_d_per_seed) < 2:
            continue

        unique_best = len(set(best_d_per_seed))
        agree_rows.append({
            "dataset": ds, "model_loss": ml,
            "n_seeds": len(best_d_per_seed),
            "unique_best_d": unique_best,
            "best_d_values": sorted(set(best_d_per_seed)),
            "agreement_rate": round(1.0 - (unique_best - 1) / len(best_d_per_seed), 3),
        })
    agree_df = pd.DataFrame(agree_rows).sort_values("agreement_rate")
    agree_df["best_d_values"] = agree_df["best_d_values"].apply(lambda x: ",".join(map(str, x)))
    display(agree_df)
    save_table(agree_df, tab_dir / "best_d_agreement.csv")

    # ---- 6. Heatmap of ARI std ----
    section_header("6. ARI Std Heatmap")
    std_pivot = stab.groupby(["dataset", "model_loss"], as_index=False)["ari_std"].mean()
    std_heatmap = std_pivot.pivot(index="dataset", columns="model_loss", values="ari_std")
    if std_heatmap.notna().any().any():
        plt.figure(figsize=(12, 5))
        sns.heatmap(std_heatmap, annot=True, fmt=".3f", cmap="Reds", linewidths=0.5)
        plt.title("Mean ARI Std across Seeds (lower = more stable)")
        plt.ylabel("")
        save_fig(fig_dir / "ari_std_heatmap.png")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
