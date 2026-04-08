#!/usr/bin/env python3
"""
09 — ELBO & Reconstruction Decomposition

Analyzes ELBO (for VAE/scVI), reconstruction-KL trade-off,
and evidence of posterior collapse.
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
    parser = argparse.ArgumentParser(description="09 — ELBO Analysis")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "09_elbo_analysis", \
        out_root / "09_elbo_analysis" / "figures", \
        out_root / "09_elbo_analysis" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("09 — ELBO & Reconstruction Decomposition")

    df = success_only(load_main_results(repo))
    palette = get_model_palette(df)

    # ---- 1. Identify VAE/scVI runs with ELBO ----
    section_header("1. ELBO Availability")
    variational = df[df["model_type"].isin(["vae", "scvi"])].copy()
    print(f"  Variational runs: {len(variational)}")

    has_elbo = variational["elbo"].notna().sum() if "elbo" in variational.columns else 0
    print(f"  Runs with ELBO column: {has_elbo}")

    # Even if ELBO column is missing/NaN, we can use reconstruction_loss and best_val_loss
    # as proxies for the reconstruction quality analysis

    # ---- 2. Reconstruction loss comparison: VAE vs AE (same loss type) ----
    section_header("2. VAE vs AE Reconstruction Comparison (Same Loss Family)")
    nb_models = df[df["loss_type"].isin(["nb", "zinb"])].copy()
    if len(nb_models):
        comp = nb_models.groupby(["dataset", "model_type", "loss_type", "latent_dim"], as_index=False).agg(
            mean_recon=("reconstruction_loss", "mean"),
            mean_val=("best_val_loss", "mean"),
        )
        save_table(comp, tab_dir / "vae_vs_ae_reconstruction.csv")

        for ds in comp["dataset"].unique():
            sub = comp[comp["dataset"] == ds]
            if len(sub) < 2:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for (mt, lt), grp in sub.groupby(["model_type", "loss_type"]):
                ml = f"{mt}:{lt}"
                color = palette.get(ml, "#999999")
                grp = grp.sort_values("latent_dim")
                ax.plot(grp["latent_dim"], grp["mean_recon"], "o-", label=ml, color=color)
            ax.set_xlabel("Latent Dim (d)")
            ax.set_ylabel("Mean Reconstruction Loss")
            ax.set_title(f"Reconstruction Loss — {dataset_label(ds)}")
            ax.legend(fontsize=8)
            save_fig(fig_dir / f"recon_comparison_{ds}.png")

    # ---- 3. Posterior collapse analysis ----
    section_header("3. Posterior Collapse Analysis")
    # A model has collapsed if: reconstruction is good (low loss) but
    # latent space is uninformative (low ARI/silhouette)
    collapse = df.groupby(["dataset", "model_loss"], as_index=False).agg(
        mean_ari=("ari", "mean"),
        mean_sil=("silhouette_kmeans", "mean"),
        mean_recon=("reconstruction_loss", "mean"),
        mean_val=("best_val_loss", "mean"),
    )
    collapse["bio_score"] = np.where(collapse["mean_ari"].notna(), collapse["mean_ari"], collapse["mean_sil"])

    # Flag: model_type is vae AND bio_score < 0.05
    collapse["is_vae"] = collapse["model_loss"].str.startswith("vae:")
    collapse["collapsed"] = collapse["is_vae"] & (collapse["bio_score"] < 0.05)

    collapsed_runs = collapse[collapse["collapsed"]]
    if len(collapsed_runs):
        print("  ⚠ Potential posterior collapse detected:")
        display(collapsed_runs[["dataset", "model_loss", "bio_score", "mean_recon"]])
    else:
        print("  No posterior collapse cases detected.")
    save_table(collapse, tab_dir / "posterior_collapse_analysis.csv")

    # ---- 4. ARI vs Reconstruction scatter (efficiency frontier) ----
    section_header("4. ARI vs Reconstruction Efficiency Frontier")
    ari_recon = df.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
        mean_ari=("ari", "mean"),
        mean_recon=("reconstruction_loss", "mean"),
    ).dropna(subset=["mean_ari", "mean_recon"])

    if len(ari_recon):
        datasets = ari_recon["dataset"].unique()
        n = len(datasets)
        cols = min(3, n)
        rows_n = max(1, int(np.ceil(n / cols)))
        fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 5 * rows_n), squeeze=False)

        for i, ds in enumerate(sorted(datasets)):
            ax = axes[i // cols][i % cols]
            sub = ari_recon[ari_recon["dataset"] == ds]
            for ml in sorted(sub["model_loss"].unique()):
                ml_sub = sub[sub["model_loss"] == ml]
                color = palette.get(ml, "#999999")
                ax.scatter(ml_sub["mean_recon"], ml_sub["mean_ari"],
                          label=ml, color=color, s=40, alpha=0.7)
            ax.set_xlabel("Reconstruction Loss")
            ax.set_ylabel("ARI")
            ax.set_title(dataset_label(ds), fontsize=10)
            ax.legend(fontsize=6)

        for j in range(n, rows_n * cols):
            axes[j // cols][j % cols].set_visible(False)
        plt.suptitle("ARI vs Reconstruction Loss (Efficiency Frontier)", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(fig_dir / "ari_vs_recon_frontier.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig] ari_vs_recon_frontier.png")

    # ---- 5. Val loss convergence gap ----
    section_header("5. Generalization Gap (Train - Val Loss)")
    if "reconstruction_loss" in df.columns and "best_val_loss" in df.columns:
        df_gap = df[df["reconstruction_loss"].notna() & df["best_val_loss"].notna()].copy()
        df_gap["gen_gap"] = df_gap["best_val_loss"] - df_gap["reconstruction_loss"]
        gap_agg = df_gap.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
            mean_gap=("gen_gap", "mean"),
        )
        save_table(gap_agg, tab_dir / "generalization_gap.csv")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
