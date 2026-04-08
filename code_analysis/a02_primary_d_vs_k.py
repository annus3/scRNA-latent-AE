#!/usr/bin/env python3
"""
02 — PRIMARY d = f(K) Analysis

Core analysis: recommended latent dimension vs biological complexity.
Includes formula fitting with curve overlays, extrapolation, bootstrap CIs,
LOO validation, and 95%-threshold d selection.
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
    load_main_results, load_recommendations, load_formula_fit, load_formula_loo,
    success_only, save_fig, save_table,
    section_header, dataset_label, FORMULA_DISPLAY,
)


# ---------------------------------------------------------------------------
# Formula definitions (must match run_experiment.py)
# ---------------------------------------------------------------------------
def _f_linear(K, a, b):
    return a * K + b

def _f_power(K, a, b):
    return a * np.power(K, b)

def _f_sqrt(K, a, b):
    return a * np.sqrt(K) + b

def _f_logk(K, a, b):
    return a * np.log(K) + b

FORMULAS = {
    "linear": (_f_linear, [0.1, 5.0]),
    "power":  (_f_power,  [1.0, 0.5]),
    "sqrt":   (_f_sqrt,   [1.0, 0.0]),
    "logk":   (_f_logk,   [3.0, -5.0]),
}


def fit_formula(K, d, name):
    """Fit a formula and return (params, r2, rmse)."""
    func, p0 = FORMULAS[name]
    try:
        params, _ = curve_fit(func, K, d, p0=p0, maxfev=5000)
    except Exception:
        params, _ = curve_fit(func, K, d, p0=[1.0, 1.0], maxfev=5000)
    pred = func(K, *params)
    ss_res = np.sum((d - pred) ** 2)
    ss_tot = np.sum((d - np.mean(d)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((d - pred) ** 2)))
    return params, r2, rmse


def main() -> None:
    parser = argparse.ArgumentParser(description="02 — Primary d=f(K) Analysis")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "02_primary_d_vs_k", \
        out_root / "02_primary_d_vs_k" / "figures", \
        out_root / "02_primary_d_vs_k" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("02 — PRIMARY d = f(K) Analysis")

    rec = load_recommendations(repo)
    fit = load_formula_fit(repo)
    loo = load_formula_loo(repo)

    # Focus on scvi:nb as the primary model
    scvi_rec = rec[(rec.model_type == "scvi") & (rec.loss_type == "nb")].copy().sort_values("K")
    scvi_fit = fit[(fit.model_type == "scvi") & (fit.loss_type == "nb")].copy().sort_values("fit_rmse")
    scvi_loo = loo[(loo.model_type == "scvi") & (loo.loss_type == "nb")].copy()

    section_header("1. scvi:nb Recommended d by K")
    display(scvi_rec[["dataset", "K", "recommended_d", "selection_metric", "selection_score"]])
    save_table(scvi_rec, tab_dir / "scvi_nb_recommended_d.csv")

    # ---- 2. Formula fit comparison ----
    section_header("2. Formula Fit Summary")
    display(scvi_fit[["fit_formula_name", "fit_score_r2", "fit_rmse"]])
    save_table(scvi_fit, tab_dir / "scvi_nb_formula_fit.csv")

    # ---- 3. CORE FIGURE: d vs K with fitted curves ----
    section_header("3. Core Figure: d vs K with Fitted Curves")
    K_data = scvi_rec["K"].to_numpy(dtype=float)
    d_data = scvi_rec["recommended_d"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(K_data, d_data, s=120, zorder=5, color="#2274A5", edgecolors="black", linewidths=0.8)

    # Annotate each point with dataset name
    for _, r in scvi_rec.iterrows():
        ax.annotate(
            dataset_label(r["dataset"]),
            (r["K"], r["recommended_d"]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=8, fontstyle="italic",
        )

    # Fit and overlay each formula
    K_smooth = np.linspace(max(1, K_data.min() - 5), K_data.max() * 1.3, 200)
    colors = {"linear": "#E85D04", "power": "#7B2D8E", "sqrt": "#2E933C", "logk": "#D90429"}
    fit_results = []

    for name in ["linear", "power", "sqrt", "logk"]:
        try:
            params, r2, rmse = fit_formula(K_data, d_data, name)
            func = FORMULAS[name][0]
            y_smooth = func(K_smooth, *params)
            label = f"{FORMULA_DISPLAY.get(name, name)}  R²={r2:.3f}  RMSE={rmse:.2f}"
            ax.plot(K_smooth, y_smooth, "--", color=colors[name], linewidth=1.5, label=label, alpha=0.8)
            fit_results.append({
                "formula": name, "param_a": params[0], "param_b": params[1],
                "r2": r2, "rmse": rmse,
            })
        except Exception as e:
            print(f"  WARNING: Formula {name} failed: {e}")

    ax.set_xlabel("K (Number of Cell Types)", fontsize=12)
    ax.set_ylabel("Recommended Latent Dimension (d)", fontsize=12)
    ax.set_title("PRIMARY: Optimal Latent Dimension vs Biological Complexity (scvi:nb)", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_ylim(bottom=0)
    save_fig(fig_dir / "core_d_vs_K_with_formulas.png")

    save_table(pd.DataFrame(fit_results), tab_dir / "formula_fit_results_refitted.csv")

    # ---- 4. Extrapolation table ----
    section_header("4. Formula Extrapolation")
    extrap_Ks = [5, 10, 20, 50, 100, 200, 500]
    extrap_rows = []
    for name in ["linear", "power", "sqrt", "logk"]:
        try:
            params, _, _ = fit_formula(K_data, d_data, name)
            func = FORMULAS[name][0]
            for k in extrap_Ks:
                extrap_rows.append({
                    "formula": name, "K": k,
                    "predicted_d": round(float(func(k, *params)), 2),
                })
        except Exception:
            pass
    extrap_df = pd.DataFrame(extrap_rows)
    extrap_pivot = extrap_df.pivot(index="K", columns="formula", values="predicted_d")
    display(extrap_pivot)
    save_table(extrap_pivot.reset_index(), tab_dir / "formula_extrapolation.csv")

    # ---- 5. Bootstrap CI on formula parameters ----
    section_header("5. Bootstrap Confidence Intervals (n=1000)")
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_rows = []
    for name in ["linear", "power", "sqrt", "logk"]:
        boot_params = []
        for _ in range(n_boot):
            idx = rng.choice(len(K_data), size=len(K_data), replace=True)
            K_b, d_b = K_data[idx], d_data[idx]
            try:
                params, _, _ = fit_formula(K_b, d_b, name)
                boot_params.append(params)
            except Exception:
                pass
        if boot_params:
            arr = np.array(boot_params)
            for i, pname in enumerate(["a", "b"]):
                boot_rows.append({
                    "formula": name, "parameter": pname,
                    "mean": np.mean(arr[:, i]),
                    "std": np.std(arr[:, i]),
                    "ci_2.5": np.percentile(arr[:, i], 2.5),
                    "ci_97.5": np.percentile(arr[:, i], 97.5),
                    "n_successful_fits": len(boot_params),
                })
    boot_df = pd.DataFrame(boot_rows)
    display(boot_df)
    save_table(boot_df, tab_dir / "formula_bootstrap_ci.csv")

    # ---- 6. LOO validation ----
    section_header("6. LOO Validation")
    loo_ok = scvi_loo[scvi_loo["status"] == "ok"].copy()
    if len(loo_ok):
        loo_agg = loo_ok.groupby("fit_formula_name", as_index=False).agg(
            loo_mae=("abs_error", "mean"),
            loo_rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            n=("held_out_dataset", "count"),
        ).sort_values("loo_rmse")
        display(loo_agg)
        save_table(loo_agg, tab_dir / "scvi_nb_loo_aggregate.csv")

        # LOO detail
        save_table(loo_ok, tab_dir / "scvi_nb_loo_detail.csv")

    # ---- 7. 95%-threshold d selection ----
    section_header("7. Minimum-Sufficient d (95% threshold)")
    main_df = success_only(load_main_results(repo))
    scvi_nb = main_df[(main_df.model_type == "scvi") & (main_df.loss_type == "nb")].copy()

    thresh_rows = []
    for ds, sub in scvi_nb.groupby("dataset"):
        score_by_d = sub.groupby("latent_dim", as_index=False)["ari"].mean()
        score_by_d = score_by_d.sort_values("latent_dim")
        max_score = score_by_d["ari"].max()
        threshold = 0.95 * max_score
        argmax_d = int(score_by_d.loc[score_by_d["ari"].idxmax(), "latent_dim"])

        # Find smallest d where score >= 95% of max
        above = score_by_d[score_by_d["ari"] >= threshold]
        min_suff_d = int(above["latent_dim"].min()) if len(above) else argmax_d

        thresh_rows.append({
            "dataset": ds, "K": int(sub["K"].iloc[0]),
            "argmax_d": argmax_d,
            "argmax_score": round(max_score, 4),
            "threshold_95pct": round(threshold, 4),
            "min_sufficient_d": min_suff_d,
        })

    thresh_df = pd.DataFrame(thresh_rows).sort_values("K")
    display(thresh_df)
    save_table(thresh_df, tab_dir / "min_sufficient_d_95pct.csv")

    # ---- 8. All model/loss recommendations side-by-side ----
    section_header("8. Cross-Model Recommended d Comparison")
    all_rec = rec.copy().sort_values(["K", "model_type", "loss_type"])
    all_rec["display_ds"] = all_rec["dataset"].map(dataset_label)
    all_rec["model_loss"] = all_rec["model_type"] + ":" + all_rec["loss_type"]
    save_table(all_rec, tab_dir / "all_model_recommended_d.csv")

    rec_pivot = all_rec.pivot_table(
        index="dataset", columns="model_loss", values="recommended_d", aggfunc="first"
    )
    plt.figure(figsize=(12, 5))
    sns.heatmap(rec_pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Recommended d by Dataset × Model/Loss")
    plt.ylabel("")
    save_fig(fig_dir / "recommended_d_heatmap_all_models.png")

    # ---- Summary ----
    interpretation = []
    if len(fit_results):
        best = min(fit_results, key=lambda x: x["rmse"])
        interpretation.append(f"Best in-sample formula (RMSE): {best['formula']} (R²={best['r2']:.3f}, RMSE={best['rmse']:.2f})")
    if len(loo_ok):
        interpretation.append(f"Best LOO formula: {loo_agg.iloc[0]['fit_formula_name']}")
    for _, r in thresh_df.iterrows():
        interpretation.append(f"  {dataset_label(r['dataset'])}: K={r['K']}, argmax_d={r['argmax_d']}, min_suff_d={r['min_sufficient_d']}")

    summary = "\n".join(interpretation)
    print(f"\n{summary}")
    (out_dir / "primary_interpretation.txt").write_text(summary)

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
