#!/usr/bin/env python3
"""
03 — Cross-Model Comparison

Compares all model/loss combinations across datasets on ARI, AMI, silhouette,
and reconstruction. Properly separates loss families for reconstruction comparison.
Flags posterior collapse in VAE:MSE.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="03 — Cross-Model Comparison")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "03_cross_model", \
        out_root / "03_cross_model" / "figures", \
        out_root / "03_cross_model" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("03 — Cross-Model Comparison")

    df = success_only(load_main_results(repo))
    palette = get_model_palette(df)

    # ---- 1. ARI heatmap ----
    section_header("1. Mean ARI Heatmap")
    ari_agg = df.groupby(["dataset", "model_loss"], as_index=False).agg(
        mean_ari=("ari", "mean"), mean_ami=("ami", "mean"), n=("seed", "count"),
    )
    ari_pivot = ari_agg.pivot(index="dataset", columns="model_loss", values="mean_ari")

    plt.figure(figsize=(14, 5))
    sns.heatmap(ari_pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5,
                mask=ari_pivot.isna())
    plt.title("Mean ARI by Dataset × Model/Loss (NaN = external metrics disabled)")
    plt.ylabel("")
    save_fig(fig_dir / "mean_ari_heatmap.png")
    save_table(ari_agg, tab_dir / "mean_ari_ami_by_dataset_model.csv")

    # ---- 2. Silhouette heatmap (always available) ----
    section_header("2. Mean Silhouette Heatmap")
    sil_agg = df.groupby(["dataset", "model_loss"], as_index=False).agg(
        mean_sil_kmeans=("silhouette_kmeans", "mean"),
        mean_sil_true=("silhouette_true_labels", "mean"),
        n=("seed", "count"),
    )
    sil_pivot = sil_agg.pivot(index="dataset", columns="model_loss", values="mean_sil_kmeans")

    plt.figure(figsize=(14, 5))
    sns.heatmap(sil_pivot, annot=True, fmt=".3f", cmap="magma", linewidths=0.5)
    plt.title("Mean Silhouette (KMeans) by Dataset × Model/Loss")
    plt.ylabel("")
    save_fig(fig_dir / "mean_silhouette_heatmap.png")
    save_table(sil_agg, tab_dir / "mean_silhouette_by_dataset_model.csv")

    # ---- 3. Score vs d for ALL model/loss combos ----
    section_header("3. Score vs Latent Dimension (All Models)")
    df["score_for_rank"] = np.where(df["ari"].notna() & np.isfinite(df["ari"]),
                                     df["ari"], df["silhouette_kmeans"])
    score_agg = df.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
        mean_score=("score_for_rank", "mean"),
        std_score=("score_for_rank", "std"),
        n=("seed", "count"),
    )

    datasets = score_agg["dataset"].unique()
    n_ds = len(datasets)
    cols = min(3, n_ds)
    rows_n = int(np.ceil(n_ds / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 4.5 * rows_n), squeeze=False)

    for i, ds in enumerate(sorted(datasets)):
        ax = axes[i // cols][i % cols]
        sub = score_agg[score_agg["dataset"] == ds]
        for ml in sorted(sub["model_loss"].unique(), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99):
            ml_sub = sub[sub["model_loss"] == ml].sort_values("latent_dim")
            color = palette.get(ml, "#999999")
            ax.plot(ml_sub["latent_dim"], ml_sub["mean_score"], "o-", label=ml,
                    color=color, markersize=4, linewidth=1.5)
        ax.set_title(dataset_label(ds), fontsize=11)
        ax.set_xlabel("Latent Dim (d)")
        ax.set_ylabel("Score")
        ax.legend(fontsize=7, ncol=2)

    # Hide unused axes
    for j in range(n_ds, rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.suptitle("Score vs Latent Dimension (ARI if available, else Silhouette)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(fig_dir / "score_vs_d_all_models.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] score_vs_d_all_models.png")

    save_table(score_agg, tab_dir / "score_by_latent_all_models.csv")

    # ---- 4. Reconstruction loss (SPLIT BY LOSS FAMILY) ----
    section_header("4. Reconstruction Loss (Separated by Loss Family)")

    # MSE family
    mse_models = df[df["loss_type"] == "mse"].copy()
    if len(mse_models):
        mse_agg = mse_models.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
            mean_recon=("reconstruction_loss", "mean"),
        )
        g = sns.relplot(data=mse_agg, x="latent_dim", y="mean_recon",
                        hue="model_loss", col="dataset", kind="line", marker="o",
                        col_wrap=3, height=3.5, facet_kws={"sharey": False},
                        palette=palette)
        g.fig.suptitle("MSE Reconstruction Loss vs d (MSE models only)", y=1.02)
        g.fig.savefig(fig_dir / "recon_mse_family.png", dpi=300, bbox_inches="tight")
        plt.close(g.fig)
        print(f"  [fig] recon_mse_family.png")

    # NB/ZINB/scVI family
    count_models = df[df["loss_type"].isin(["nb", "zinb"])].copy()
    if len(count_models):
        count_agg = count_models.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
            mean_recon=("reconstruction_loss", "mean"),
        )
        g = sns.relplot(data=count_agg, x="latent_dim", y="mean_recon",
                        hue="model_loss", col="dataset", kind="line", marker="o",
                        col_wrap=3, height=3.5, facet_kws={"sharey": False},
                        palette=palette)
        g.fig.suptitle("Count-Likelihood Reconstruction Loss vs d (NB/ZINB models)", y=1.02)
        g.fig.savefig(fig_dir / "recon_count_family.png", dpi=300, bbox_inches="tight")
        plt.close(g.fig)
        print(f"  [fig] recon_count_family.png")

    # ---- 5. Posterior collapse detection ----
    section_header("5. Posterior Collapse Detection")
    model_means = df.groupby(["dataset", "model_loss"], as_index=False).agg(
        mean_ari=("ari", "mean"),
        mean_sil=("silhouette_kmeans", "mean"),
    )
    model_means["posterior_collapse_flag"] = (
        (model_means["mean_ari"] < 0.05) & model_means["mean_ari"].notna()
    )
    collapsed = model_means[model_means["posterior_collapse_flag"]].copy()
    if len(collapsed):
        print("  WARNING: Likely posterior collapse detected:")
        display(collapsed[["dataset", "model_loss", "mean_ari", "mean_sil"]])
    else:
        print("  No posterior collapse detected.")
    save_table(model_means, tab_dir / "model_means_with_collapse_flag.csv")

    # ---- 6. Best d per model/loss ----
    section_header("6. Best d per Dataset × Model/Loss")
    best_rows = []
    for (ds, ml), sub in df.groupby(["dataset", "model_loss"]):
        ranked = sub.groupby("latent_dim", as_index=False)["score_for_rank"].mean()
        ranked = ranked.sort_values("score_for_rank", ascending=False)
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        best_rows.append({
            "dataset": ds, "model_loss": ml,
            "best_d": int(best["latent_dim"]), "best_score": round(best["score_for_rank"], 4),
        })
    best_df = pd.DataFrame(best_rows).sort_values(["dataset", "model_loss"])
    display(best_df)
    save_table(best_df, tab_dir / "best_d_per_dataset_model.csv")

    # Best d pivot for visual comparison
    bp = best_df.pivot(index="dataset", columns="model_loss", values="best_d")
    plt.figure(figsize=(12, 5))
    sns.heatmap(bp, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Best Latent Dimension per Dataset × Model/Loss")
    plt.ylabel("")
    save_fig(fig_dir / "best_d_heatmap.png")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
