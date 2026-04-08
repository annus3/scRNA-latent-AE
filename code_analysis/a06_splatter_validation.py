#!/usr/bin/env python3
"""
06 — Splatter K-Sweep Validation

Dedicated analysis of synthetic data (K=4, 8, 12).
Validates d=f(K) trends under controlled conditions and compares
with real-data findings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    display, find_repo_root, resolve_output_root,
    load_splatter_results, success_only, save_fig, save_table,
    section_header, get_model_palette, MODEL_ORDER, FORMULA_DISPLAY,
)


def _f_linear(K, a, b): return a * K + b
def _f_power(K, a, b): return a * np.power(K, b)
def _f_sqrt(K, a, b): return a * np.sqrt(K) + b
def _f_logk(K, a, b): return a * np.log(K) + b

FORMULAS = {"linear": _f_linear, "power": _f_power, "sqrt": _f_sqrt, "logk": _f_logk}


def main() -> None:
    parser = argparse.ArgumentParser(description="06 — Splatter Validation")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "06_splatter_validation", \
        out_root / "06_splatter_validation" / "figures", \
        out_root / "06_splatter_validation" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("06 — Splatter K-Sweep Validation Analysis")

    df = success_only(load_splatter_results(repo))
    df["model_loss"] = df["model_type"].astype(str) + ":" + df["loss_type"].astype(str)
    palette = get_model_palette(df)

    print(f"  Total rows: {len(df)}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Model/loss: {sorted(df['model_loss'].unique())}")

    # ---- 1. Score vs d per K ----
    section_header("1. Score vs Latent Dim Faceted by K")
    df["score"] = np.where(df["ari"].notna(), df["ari"], df["silhouette_kmeans"])

    agg = df.groupby(["dataset", "K", "model_loss", "latent_dim"], as_index=False).agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        mean_sil=("silhouette_kmeans", "mean"),
        mean_recon=("reconstruction_loss", "mean"),
        n=("seed", "count"),
    )

    K_vals = sorted(df["K"].unique())
    fig, axes = plt.subplots(1, len(K_vals), figsize=(6 * len(K_vals), 5), squeeze=False)
    for i, k in enumerate(K_vals):
        ax = axes[0][i]
        sub = agg[agg["K"] == k]
        for ml in sorted(sub["model_loss"].unique(), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99):
            ml_sub = sub[sub["model_loss"] == ml].sort_values("latent_dim")
            color = palette.get(ml, "#999999")
            ax.errorbar(ml_sub["latent_dim"], ml_sub["mean_score"], yerr=ml_sub["std_score"],
                       fmt="o-", label=ml, color=color, markersize=4, capsize=2)
        ax.set_title(f"K = {k}", fontsize=12)
        ax.set_xlabel("Latent Dim (d)")
        ax.set_ylabel("Score")
        ax.legend(fontsize=6, ncol=2)
    plt.suptitle("Splatter: Score vs Latent Dimension", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "splatter_score_vs_d_by_K.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] splatter_score_vs_d_by_K.png")

    save_table(agg, tab_dir / "splatter_score_by_d.csv")

    # ---- 2. Recommended d per K per model ----
    section_header("2. Recommended d per K (Splatter)")
    rec_rows = []
    for (k, ml), sub in df.groupby(["K", "model_loss"]):
        by_d = sub.groupby("latent_dim", as_index=False)["score"].mean()
        by_d = by_d.sort_values("score", ascending=False)
        if by_d.empty:
            continue
        best = by_d.iloc[0]
        rec_rows.append({"K": k, "model_loss": ml,
                         "best_d": int(best["latent_dim"]),
                         "best_score": round(best["score"], 4)})
    rec_df = pd.DataFrame(rec_rows).sort_values(["model_loss", "K"])
    display(rec_df)
    save_table(rec_df, tab_dir / "splatter_recommended_d.csv")

    # ---- 3. d vs K trend per model (the Splatter validation of d=f(K)) ----
    section_header("3. d vs K Trend (Splatter Validation)")
    fig, ax = plt.subplots(figsize=(8, 5))
    for ml in sorted(rec_df["model_loss"].unique(),
                     key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99):
        sub = rec_df[rec_df["model_loss"] == ml].sort_values("K")
        color = palette.get(ml, "#999999")
        ax.plot(sub["K"], sub["best_d"], "o-", label=ml, color=color,
                markersize=8, linewidth=2)
    ax.set_xlabel("K (Number of Cell Types)")
    ax.set_ylabel("Recommended d")
    ax.set_title("Splatter: Optimal d vs K (Controlled Synthetic Data)")
    ax.legend()
    save_fig(fig_dir / "splatter_d_vs_K.png")

    # ---- 4. Formula fit on Splatter data ----
    section_header("4. Formula Fit (Splatter scvi:nb)")
    scvi_rec = rec_df[rec_df["model_loss"] == "scvi:nb"].copy()
    if len(scvi_rec) >= 3:
        K_arr = scvi_rec["K"].to_numpy(dtype=float)
        d_arr = scvi_rec["best_d"].to_numpy(dtype=float)
        fit_rows = []
        for name, func in FORMULAS.items():
            try:
                params, _ = curve_fit(func, K_arr, d_arr, p0=[1.0, 1.0], maxfev=5000)
                pred = func(K_arr, *params)
                ss_res = np.sum((d_arr - pred) ** 2)
                ss_tot = np.sum((d_arr - np.mean(d_arr)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                rmse = float(np.sqrt(np.mean((d_arr - pred) ** 2)))
                fit_rows.append({"formula": name, "r2": r2, "rmse": rmse,
                                 "param_a": params[0], "param_b": params[1]})
            except Exception as e:
                print(f"  Formula {name} failed: {e}")
        fit_df = pd.DataFrame(fit_rows).sort_values("rmse")
        display(fit_df)
        save_table(fit_df, tab_dir / "splatter_formula_fit.csv")

    # ---- 5. Model ranking by K ----
    section_header("5. Model Ranking per K")
    rank_agg = df.groupby(["K", "model_loss"], as_index=False).agg(
        mean_score=("score", "mean"),
        mean_sil=("silhouette_kmeans", "mean"),
        mean_recon=("reconstruction_loss", "mean"),
    )
    save_table(rank_agg.sort_values(["K", "mean_score"], ascending=[True, False]),
               tab_dir / "splatter_model_ranking.csv")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
