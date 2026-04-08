#!/usr/bin/env python3
"""
07 — Statistical Significance Tests

Provides statistical rigor: Wilcoxon signed-rank tests for model comparison,
effect sizes, bootstrap confidence intervals, and multi-dataset significance
summary figures.
"""
from __future__ import annotations

import argparse
import itertools
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


def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if np.isclose(pooled_std, 0):
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def main() -> None:
    parser = argparse.ArgumentParser(description="07 — Statistical Significance")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "07_significance", \
        out_root / "07_significance" / "figures", \
        out_root / "07_significance" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("07 — Statistical Significance Tests")

    df = success_only(load_main_results(repo))

    # ---- 1. Pairwise model comparison (Wilcoxon on ARI) ----
    section_header("1. Pairwise Model Comparison (ARI)")
    model_losses = sorted(df["model_loss"].unique())
    paired_rows = []

    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        if ds_df["ari"].isna().all():
            continue

        for ml_a, ml_b in itertools.combinations(model_losses, 2):
            a_df = ds_df[ds_df["model_loss"] == ml_a]
            b_df = ds_df[ds_df["model_loss"] == ml_b]
            if len(a_df) == 0 or len(b_df) == 0:
                continue

            merged = a_df[["latent_dim", "seed", "ari"]].merge(
                b_df[["latent_dim", "seed", "ari"]],
                on=["latent_dim", "seed"], suffixes=("_a", "_b"),
            ).dropna()

            if len(merged) < 5:
                continue

            vals_a = merged["ari_a"].values
            vals_b = merged["ari_b"].values

            try:
                stat_w, p_w = stats.wilcoxon(vals_a, vals_b, alternative="two-sided")
            except Exception:
                stat_w, p_w = float("nan"), float("nan")

            cd = cohens_d(vals_a, vals_b)
            mean_diff = float(np.mean(vals_a) - np.mean(vals_b))

            paired_rows.append({
                "dataset": ds, "model_a": ml_a, "model_b": ml_b,
                "n_pairs": len(merged),
                "mean_a": round(np.mean(vals_a), 4),
                "mean_b": round(np.mean(vals_b), 4),
                "mean_diff": round(mean_diff, 4),
                "cohens_d": round(cd, 3),
                "wilcoxon_stat": round(stat_w, 2) if np.isfinite(stat_w) else np.nan,
                "p_value": round(p_w, 6) if np.isfinite(p_w) else np.nan,
                "significant_005": p_w < 0.05 if np.isfinite(p_w) else False,
            })

    paired_df = pd.DataFrame(paired_rows).sort_values(["dataset", "p_value"])
    display(paired_df.head(30))
    save_table(paired_df, tab_dir / "pairwise_wilcoxon_ari.csv")

    # ---- 2. Same for silhouette (always available) ----
    section_header("2. Pairwise Model Comparison (Silhouette)")
    sil_rows = []
    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        for ml_a, ml_b in itertools.combinations(model_losses, 2):
            a_df = ds_df[ds_df["model_loss"] == ml_a]
            b_df = ds_df[ds_df["model_loss"] == ml_b]
            if len(a_df) == 0 or len(b_df) == 0:
                continue
            merged = a_df[["latent_dim", "seed", "silhouette_kmeans"]].merge(
                b_df[["latent_dim", "seed", "silhouette_kmeans"]],
                on=["latent_dim", "seed"], suffixes=("_a", "_b"),
            ).dropna()
            if len(merged) < 5:
                continue
            vals_a = merged["silhouette_kmeans_a"].values
            vals_b = merged["silhouette_kmeans_b"].values
            try:
                _, p_w = stats.wilcoxon(vals_a, vals_b)
            except Exception:
                p_w = float("nan")
            sil_rows.append({
                "dataset": ds, "model_a": ml_a, "model_b": ml_b,
                "n_pairs": len(merged),
                "mean_diff": round(float(np.mean(vals_a) - np.mean(vals_b)), 4),
                "cohens_d": round(cohens_d(vals_a, vals_b), 3),
                "p_value": round(p_w, 6) if np.isfinite(p_w) else np.nan,
                "significant_005": p_w < 0.05 if np.isfinite(p_w) else False,
            })
    sil_df = pd.DataFrame(sil_rows).sort_values(["dataset", "p_value"])
    save_table(sil_df, tab_dir / "pairwise_wilcoxon_silhouette.csv")

    # ---- 3. Bootstrap CI on mean ARI per model ----
    section_header("3. Bootstrap CI on Mean ARI per Model")
    rng = np.random.default_rng(42)
    boot_rows = []
    for (ds, ml), sub in df.groupby(["dataset", "model_loss"]):
        vals = sub["ari"].dropna().values
        if len(vals) < 3:
            continue
        boot_means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(2000)]
        boot_rows.append({
            "dataset": ds, "model_loss": ml,
            "mean_ari": round(np.mean(vals), 4),
            "ci_2.5": round(np.percentile(boot_means, 2.5), 4),
            "ci_97.5": round(np.percentile(boot_means, 97.5), 4),
            "ci_width": round(np.percentile(boot_means, 97.5) - np.percentile(boot_means, 2.5), 4),
            "n": len(vals),
        })
    boot_df = pd.DataFrame(boot_rows)
    display(boot_df)
    save_table(boot_df, tab_dir / "bootstrap_ci_ari.csv")

    # ---- 4. Significance summary figure (ALL datasets) ----
    section_header("4. Significance Summary Figures")
    if len(paired_df):
        datasets_with_sig = sorted(paired_df["dataset"].unique())
        for ds in datasets_with_sig:
            sig = paired_df[paired_df["dataset"] == ds].copy()
            if sig.empty:
                continue
            sig["comparison"] = sig["model_a"].str.split(":").str[0] + " vs " + sig["model_b"].str.split(":").str[0]
            sig = sig.head(15)  # top 15 comparisons by p-value
            sig["log_p"] = -np.log10(sig["p_value"].clip(lower=1e-20))

            plt.figure(figsize=(10, max(4, len(sig) * 0.35)))
            colors = ["#2E933C" if s else "#D90429" for s in sig["significant_005"]]
            plt.barh(sig["comparison"], sig["log_p"], color=colors)
            plt.axvline(x=-np.log10(0.05), color="black", linestyle="--", linewidth=1, label="p=0.05")
            plt.xlabel("-log10(p-value)")
            plt.title(f"Statistical Significance — {dataset_label(ds)}")
            plt.legend(loc="lower right")
            save_fig(fig_dir / f"significance_{ds}.png")

    # ---- 5. Bootstrap CI forest plot ----
    section_header("5. Bootstrap CI Forest Plot")
    if len(boot_df):
        for ds in sorted(boot_df["dataset"].unique()):
            ds_boot = boot_df[boot_df["dataset"] == ds].sort_values("mean_ari", ascending=True).copy()
            if ds_boot.empty:
                continue
            plt.figure(figsize=(8, max(3, len(ds_boot) * 0.4)))
            y_pos = range(len(ds_boot))
            plt.errorbar(
                ds_boot["mean_ari"], y_pos,
                xerr=[ds_boot["mean_ari"] - ds_boot["ci_2.5"],
                      ds_boot["ci_97.5"] - ds_boot["mean_ari"]],
                fmt="o", color="#2274A5", ecolor="#999999", capsize=3, markersize=6,
            )
            plt.yticks(y_pos, ds_boot["model_loss"])
            plt.xlabel("Mean ARI (with 95% Bootstrap CI)")
            plt.title(f"Model Performance — {dataset_label(ds)}")
            plt.axvline(x=ds_boot["mean_ari"].max(), color="#E85D04", linestyle="--", alpha=0.4)
            save_fig(fig_dir / f"bootstrap_ci_{ds}.png")

    # ---- 6. Significance count summary table ----
    section_header("6. Significance Count Summary")
    if len(paired_df):
        sig_summary = paired_df.groupby("dataset", as_index=False).agg(
            total_comparisons=("significant_005", "count"),
            significant_at_005=("significant_005", "sum"),
        )
        sig_summary["pct_significant"] = (
            100 * sig_summary["significant_at_005"] / sig_summary["total_comparisons"]
        ).round(1)
        display(sig_summary)
        save_table(sig_summary, tab_dir / "significance_count_summary.csv")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
