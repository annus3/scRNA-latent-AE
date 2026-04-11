#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    display, find_repo_root, resolve_output_root,
    load_main_results, load_recommendations, load_splatter_results,
    success_only, save_fig, save_table, find_elbow,
    section_header, dataset_label, get_model_palette, PALETTE, MODEL_ORDER,
    FORMULA_DISPLAY,
)


def _f_linear(K, a, b): return a * K + b
def _f_power(K, a, b): return a * np.power(K, b)
def _f_sqrt(K, a, b): return a * np.sqrt(K) + b
def _f_logk(K, a, b): return a * np.log(K) + b


def _save_pub(fig, fig_dir, name):
    """Save figure as both PNG (300 DPI) and PDF."""
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}.png/pdf")


# Metadata for UMAP generation: (dataset_name, label_key, display_name, h5ad_candidates)
_UMAP_DATASETS = [
    ("scvi_pbmc12k",      "cell_type",        "PBMC12k",   ["scvi_pbmc12k.h5ad"]),
    ("paul15",            "paul15_clusters",   "Paul15",    ["paul15.h5ad"]),
    ("aifi_immune_full",  "cell_type",         "AIFI",      ["aifi_immune_full.h5ad"]),
    ("ts2_lung",          "cell_type",         "TS2-Lung",  ["ts2_lung.h5ad"]),
    ("ts1_all_cells",     "cell_type",         "TS1",       ["ts1_all_cells_phase4_ready.h5ad",
                                                              "ts1_all_cells.h5ad"]),
]

_UMAP_MAX_CELLS = 50000  # subsample large datasets for tractability


def _load_for_umap(repo: Path, ds_name: str, label_key: str, candidates: list):
    """Load a dataset h5ad and return (adata_with_umap, labels).

    Searches processed/ directories on both $HOME and $WORK.
    Large datasets are subsampled to _UMAP_MAX_CELLS.
    Returns None if no h5ad is found.
    """
    search_dirs = [
        repo / "data" / "processed",
        Path(os.environ.get("WORK", "/dev/null")) / "sc_autoencoder_project" / "data" / "processed",
    ]
    for cand in candidates:
        for d in search_dirs:
            path = d / cand
            if not path.exists():
                continue
            print(f"    Found: {path}")
            try:
                # Peek at size via backed mode
                adata_peek = sc.read_h5ad(str(path), backed="r")
                n_obs = adata_peek.n_obs
                adata_peek.file.close()

                if n_obs > _UMAP_MAX_CELLS and ds_name == "aifi_immune_full":
                    # AIFI is too large for full read — use h5py direct indexing
                    rng = np.random.default_rng(42)
                    idx = np.sort(rng.choice(n_obs, _UMAP_MAX_CELLS, replace=False))
                    with h5py.File(str(path), "r") as f:
                        ct_grp = f["obs"][label_key]
                        if isinstance(ct_grp, h5py.Group):
                            cats = [x.decode() if isinstance(x, bytes) else x
                                    for x in ct_grp["categories"][:]]
                            codes = ct_grp["codes"][:]
                            labels = np.array([cats[c] for c in codes[idx]])
                        else:
                            labels = np.array([x.decode() if isinstance(x, bytes) else x
                                               for x in ct_grp[idx]])
                        X = f["X"][idx, :].astype(np.float32)
                    import anndata as ad
                    adata = ad.AnnData(X=X)
                    adata.obs[label_key] = pd.Categorical(labels)
                else:
                    adata = sc.read_h5ad(str(path))
                    if adata.n_obs > _UMAP_MAX_CELLS:
                        rng = np.random.default_rng(42)
                        idx = np.sort(rng.choice(adata.n_obs, _UMAP_MAX_CELLS, replace=False))
                        adata = adata[idx].copy()

                # Compute UMAP if missing
                if "X_umap" not in adata.obsm:
                    if "X_pca" not in adata.obsm:
                        sc.pp.pca(adata, n_comps=50)
                    sc.pp.neighbors(adata, n_neighbors=15,
                                    n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
                    sc.tl.umap(adata)
                return adata
            except Exception as e:
                print(f"    WARNING: failed to load {path}: {e}")
    return None


def _generate_umap_strip(repo: Path, rec: pd.DataFrame, fig_dir: Path):
    """Create a 1x5 UMAP strip figure — clean visualization, no legends."""
    n_ds = len(_UMAP_DATASETS)
    fig, axes = plt.subplots(1, n_ds, figsize=(20, 4))
    fig.subplots_adjust(wspace=0.15, left=0.03, right=0.98, top=0.88, bottom=0.08)

    for i, (ds_name, label_key, display_name, candidates) in enumerate(_UMAP_DATASETS):
        ax = axes[i]

        # Get K from recommendations or hardcode fallback
        K_row = rec[rec["dataset"] == ds_name]
        K = int(K_row["K"].iloc[0]) if len(K_row) > 0 else "?"

        print(f"    [{i+1}/{n_ds}] {display_name} (K={K})...")
        adata = _load_for_umap(repo, ds_name, label_key, candidates)

        if adata is None:
            ax.text(0.5, 0.5, f"{display_name}\n(data not available)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{display_name}  (K={K})", fontsize=12, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        labels = adata.obs[label_key].astype(str).values
        unique = sorted(set(labels))
        n_types = len(unique)

        # Generate distinct colors
        if n_types <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_types]
        elif n_types <= 40:
            colors = np.vstack([plt.cm.tab20(np.linspace(0, 1, 20)),
                                plt.cm.tab20b(np.linspace(0, 1, 20))])[:n_types]
        else:
            colors = plt.cm.gist_ncar(np.linspace(0.05, 0.95, n_types))

        label_to_color = {l: colors[j] for j, l in enumerate(unique)}
        umap = adata.obsm["X_umap"]

        for label in unique:
            mask = labels == label
            ax.scatter(umap[mask, 0], umap[mask, 1],
                       c=[label_to_color[label]], s=1.5, alpha=0.5, rasterized=True)

        ax.set_title(f"{display_name}  (K={K})", fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.set_ylabel("UMAP 2", fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=11, fontweight="bold")

    _save_pub(fig, fig_dir, "fig5_umap_strip")


def main() -> None:
    parser = argparse.ArgumentParser(description="14 — Publication Figures")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "14_publication_figures", \
        out_root / "14_publication_figures" / "figures", \
        out_root / "14_publication_figures" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("14 — Publication-Quality Summary Figures")

    df = success_only(load_main_results(repo))
    rec = load_recommendations(repo)
    palette = get_model_palette(df)

    # ====================================================================
    # FIGURE 1: Core d = f(K) with all four formula curves
    # ====================================================================
    section_header("Figure 1: d = f(K) Core Result")
    scvi_rec = rec[(rec.model_type == "scvi") & (rec.loss_type == "nb")].sort_values("K")
    K = scvi_rec["K"].to_numpy(dtype=float)
    d = scvi_rec["recommended_d"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(K, d, s=140, zorder=5, color="#2274A5", edgecolors="black", linewidths=1.0)
    for _, r in scvi_rec.iterrows():
        ax.annotate(dataset_label(r["dataset"]),
                    (r["K"], r["recommended_d"]),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=8, fontstyle="italic", color="#333333")

    K_smooth = np.linspace(1, max(K) * 1.5, 300)
    formulas = [
        ("power", _f_power, [1.0, 0.5], "#7B2D8E", "--"),
        ("sqrt", _f_sqrt, [1.0, 0.0], "#2E933C", "-."),
        ("linear", _f_linear, [0.05, 3.0], "#E85D04", ":"),
        ("logk", _f_logk, [3.0, -5.0], "#D90429", (0, (3, 1, 1, 1))),
    ]
    fit_table = []
    for name, func, p0, color, ls in formulas:
        try:
            params, _ = curve_fit(func, K, d, p0=p0, maxfev=5000)
            pred = func(K, *params)
            rmse = float(np.sqrt(np.mean((d - pred) ** 2)))
            ss_res = np.sum((d - pred) ** 2)
            ss_tot = np.sum((d - np.mean(d)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            label = f"{FORMULA_DISPLAY[name]}  R²={r2:.3f}"
            ax.plot(K_smooth, func(K_smooth, *params), ls, color=color, linewidth=2, label=label, alpha=0.8)
            fit_table.append({"formula": name, "R2": round(r2, 4), "RMSE": round(rmse, 3),
                              "param_a": round(params[0], 4), "param_b": round(params[1], 4)})
        except Exception as e:
            print(f"  WARNING: Formula {name} failed: {e}")

    ax.set_xlabel("K (Number of Cell Types)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Optimal Latent Dimension (d)", fontsize=12, fontweight="bold")
    ax.set_title("Optimal Latent Dimensionality vs Biological Complexity\n(scVI with NB likelihood)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    _save_pub(fig, fig_dir, "fig1_d_vs_K")
    if fit_table:
        save_table(pd.DataFrame(fit_table), tab_dir / "fig1_formula_fits.csv")

    # ====================================================================
    # FIGURE 2: Model comparison violin/box per dataset
    # ====================================================================
    section_header("Figure 2: Model Comparison")
    datasets_with_ari = sorted([ds for ds in df["dataset"].unique()
                                if df[df["dataset"] == ds]["ari"].notna().any()])
    n = len(datasets_with_ari)
    if n > 0:
        cols = min(3, n)
        rows_n = max(1, int(np.ceil(n / cols)))
        fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 4.5 * rows_n), squeeze=False)
        for i, ds in enumerate(datasets_with_ari):
            ax = axes[i // cols][i % cols]
            sub = df[(df["dataset"] == ds) & df["ari"].notna()]
            order = sorted(sub["model_loss"].unique(),
                          key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)
            sns.boxplot(data=sub, x="model_loss", y="ari", ax=ax, order=order,
                       palette=palette, width=0.6)
            ax.set_title(dataset_label(ds), fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("ARI")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        for j in range(n, rows_n * cols):
            axes[j // cols][j % cols].set_visible(False)
        plt.suptitle("ARI Distribution by Model/Loss per Dataset", fontsize=13, y=1.01)
        plt.tight_layout()
        _save_pub(fig, fig_dir, "fig2_model_comparison")

    # ====================================================================
    # FIGURE 3: Score vs d elbow curves (scvi:nb across datasets)
    # ====================================================================
    section_header("Figure 3: Score vs d Elbow Curves")
    scvi_nb = df[(df["model_type"] == "scvi") & (df["loss_type"] == "nb")].copy()
    score_by_d = scvi_nb.groupby(["dataset", "latent_dim"], as_index=False).agg(
        mean_ari=("ari", "mean"),
        std_ari=("ari", "std"),
        mean_sil=("silhouette_kmeans", "mean"),
    )
    n_ds = score_by_d["dataset"].nunique()
    cols = min(3, n_ds)
    rows_n = max(1, int(np.ceil(n_ds / cols)))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4 * rows_n), squeeze=False)
    for i, ds in enumerate(sorted(score_by_d["dataset"].unique())):
        ax = axes[i // cols][i % cols]
        sub = score_by_d[score_by_d["dataset"] == ds].sort_values("latent_dim")
        score = sub["mean_ari"] if sub["mean_ari"].notna().any() else sub["mean_sil"]
        score_std = sub["std_ari"] if sub["mean_ari"].notna().any() else None

        ax.plot(sub["latent_dim"], score, "o-", color="#2274A5", markersize=6, linewidth=2)
        if score_std is not None and score_std.notna().any():
            ax.fill_between(sub["latent_dim"], score - score_std, score + score_std,
                           alpha=0.15, color="#2274A5")

        idx_max = score.idxmax()
        if pd.notna(idx_max):
            best_d = sub.loc[idx_max, "latent_dim"]
            best_s = score.loc[idx_max]
            ax.axvline(x=best_d, color="#E85D04", linestyle="--", alpha=0.5)
            ax.annotate(f"d*={int(best_d)}", (best_d, best_s), fontsize=9,
                       xytext=(10, -15), textcoords="offset points", color="#E85D04", fontweight="bold")

        K_val = scvi_nb[scvi_nb["dataset"] == ds]["K"].iloc[0] if len(scvi_nb[scvi_nb["dataset"] == ds]) else ""
        ax.set_title(f"{dataset_label(ds)} (K={K_val})", fontsize=10)
        ax.set_xlabel("d")
        ax.set_ylabel("ARI")
    for j in range(n_ds, rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)
    plt.suptitle("scVI-NB: ARI vs Latent Dimension per Dataset", fontsize=13, y=1.01)
    plt.tight_layout()
    _save_pub(fig, fig_dir, "fig3_score_elbow")

    # ====================================================================
    # FIGURE S1: Full ARI heatmap
    # ====================================================================
    section_header("Figure S1: Full ARI Heatmap")
    ari_agg = df.groupby(["dataset", "model_loss"], as_index=False)["ari"].mean()
    ari_pivot = ari_agg.pivot(index="dataset", columns="model_loss", values="ari")
    # Reorder columns by MODEL_ORDER
    col_order = [c for c in MODEL_ORDER if c in ari_pivot.columns]
    ari_pivot = ari_pivot[col_order]
    ari_pivot.index = ari_pivot.index.map(dataset_label)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(ari_pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5, ax=ax)
    ax.set_title("Mean ARI: All Datasets × Model/Loss Combinations", fontsize=13)
    ax.set_ylabel("")
    _save_pub(fig, fig_dir, "figS1_ari_heatmap")

    # ====================================================================
    # FIGURE S2: Recommended d comparison across models (heatmap)
    # ====================================================================
    section_header("Figure S2: Recommended d Comparison")
    all_rec = rec.copy()
    all_rec["model_loss"] = all_rec["model_type"] + ":" + all_rec["loss_type"]
    rec_pivot = all_rec.pivot_table(
        index="dataset", columns="model_loss", values="recommended_d", aggfunc="first"
    )
    col_order = [c for c in MODEL_ORDER if c in rec_pivot.columns]
    rec_pivot = rec_pivot[col_order]
    rec_pivot.index = rec_pivot.index.map(dataset_label)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(rec_pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_title("Recommended Latent Dimension by Dataset × Model", fontsize=13)
    ax.set_ylabel("")
    _save_pub(fig, fig_dir, "figS2_recommended_d_heatmap")

    # ====================================================================
    # FIGURE S3: Reconstruction loss comparison (count models only)
    # ====================================================================
    section_header("Figure S3: Reconstruction Loss Comparison")
    count_models = df[df["loss_type"].isin(["nb", "zinb"])].copy()
    if len(count_models):
        recon_agg = count_models.groupby(["dataset", "model_loss"], as_index=False)["reconstruction_loss"].mean()
        recon_pivot = recon_agg.pivot(index="dataset", columns="model_loss", values="reconstruction_loss")
        col_order = [c for c in MODEL_ORDER if c in recon_pivot.columns]
        recon_pivot = recon_pivot[col_order]
        recon_pivot.index = recon_pivot.index.map(dataset_label)

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(recon_pivot, annot=True, fmt=".1f", cmap="magma_r", linewidths=0.5, ax=ax)
        ax.set_title("Mean Reconstruction Loss: Count-Likelihood Models", fontsize=13)
        ax.set_ylabel("")
        _save_pub(fig, fig_dir, "figS3_reconstruction_heatmap")

    # ====================================================================
    # FIGURE 4: Combined summary panel (2x2 layout)
    # ====================================================================
    section_header("Figure 4: Combined Summary Panel")
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: d vs K
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.scatter(K, d, s=100, zorder=5, color="#2274A5", edgecolors="black", linewidths=0.8)
    for _, r in scvi_rec.iterrows():
        ax_a.annotate(dataset_label(r["dataset"]), (r["K"], r["recommended_d"]),
                      textcoords="offset points", xytext=(8, 6), fontsize=7, fontstyle="italic")
    try:
        params, _ = curve_fit(_f_power, K, d, p0=[1.0, 0.5], maxfev=5000)
        ax_a.plot(K_smooth, _f_power(K_smooth, *params), "--", color="#7B2D8E", linewidth=1.5, alpha=0.7)
    except Exception:
        pass
    ax_a.set_xlabel("K")
    ax_a.set_ylabel("d*")
    ax_a.set_title("(A) Optimal d vs K (scVI-NB)", fontsize=11, fontweight="bold")
    ax_a.set_ylim(bottom=0)
    ax_a.grid(True, alpha=0.3)

    # Panel B: ARI by model (overall)
    ax_b = fig.add_subplot(gs[0, 1])
    model_ari = df.groupby("model_loss", as_index=False)["ari"].mean().dropna()
    model_ari = model_ari.sort_values("ari", ascending=True)
    colors = [PALETTE.get(ml, "#999999") for ml in model_ari["model_loss"]]
    ax_b.barh(model_ari["model_loss"], model_ari["ari"], color=colors)
    ax_b.set_xlabel("Mean ARI")
    ax_b.set_title("(B) Overall Model Ranking", fontsize=11, fontweight="bold")

    # Panel C: Score vs d for top 3 datasets by K
    ax_c = fig.add_subplot(gs[1, 0])
    scvi_nb_agg = scvi_nb.groupby(["dataset", "latent_dim", "K"], as_index=False)["ari"].mean()
    top_ds = sorted(scvi_nb_agg["dataset"].unique(), key=lambda x: scvi_nb_agg[scvi_nb_agg["dataset"] == x]["K"].iloc[0])
    ds_colors = ["#2274A5", "#E85D04", "#7B2D8E", "#2E933C", "#D90429"]
    for i, ds in enumerate(top_ds[:5]):
        sub = scvi_nb_agg[scvi_nb_agg["dataset"] == ds].sort_values("latent_dim")
        ax_c.plot(sub["latent_dim"], sub["ari"], "o-", label=f"{dataset_label(ds)}",
                  color=ds_colors[i % len(ds_colors)], markersize=4, linewidth=1.5)
    ax_c.set_xlabel("Latent Dimension (d)")
    ax_c.set_ylabel("ARI")
    ax_c.set_title("(C) ARI vs d by Dataset (scVI-NB)", fontsize=11, fontweight="bold")
    ax_c.legend(fontsize=7, loc="best")
    ax_c.grid(True, alpha=0.3)

    # Panel D: Formula fit quality bar chart
    ax_d = fig.add_subplot(gs[1, 1])
    if fit_table:
        ft_df = pd.DataFrame(fit_table).sort_values("R2", ascending=True)
        formula_colors = {"linear": "#E85D04", "power": "#7B2D8E", "sqrt": "#2E933C", "logk": "#D90429"}
        bar_colors = [formula_colors.get(f, "#999999") for f in ft_df["formula"]]
        bars = ax_d.barh(ft_df["formula"], ft_df["R2"], color=bar_colors)
        ax_d.set_xlabel("R²")
        ax_d.set_title("(D) Formula Fit Quality (in-sample)", fontsize=11, fontweight="bold")
        ax_d.set_xlim(0, 1)
        for bar, rmse in zip(bars, ft_df["RMSE"]):
            ax_d.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                      f"RMSE={rmse}", va="center", fontsize=8)

    _save_pub(fig, fig_dir, "fig4_summary_panel")

    # ====================================================================
    # FIGURE S4: Reconstruction loss elbow detection per model/dataset
    # ====================================================================
    section_header("Figure S4: Reconstruction Loss Elbow Detection")
    recon_data = df[df["reconstruction_loss"].notna()].copy()
    recon_agg = recon_data.groupby(
        ["dataset", "model_loss", "latent_dim"], as_index=False
    ).agg(mean_loss=("reconstruction_loss", "mean"), std_loss=("reconstruction_loss", "std"))

    elbow_rows = []
    ds_list = sorted(recon_agg["dataset"].unique())
    n_ds = len(ds_list)
    if n_ds > 0:
        cols = min(3, n_ds)
        rows_n = max(1, int(np.ceil(n_ds / cols)))
        fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 4.5 * rows_n), squeeze=False)
        for i, ds in enumerate(ds_list):
            ax = axes[i // cols][i % cols]
            ds_sub = recon_agg[recon_agg["dataset"] == ds]
            for ml in sorted(ds_sub["model_loss"].unique(),
                             key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99):
                sub = ds_sub[ds_sub["model_loss"] == ml].sort_values("latent_dim")
                if len(sub) < 3:
                    continue
                x_vals = sub["latent_dim"].values
                y_vals = sub["mean_loss"].values
                elbow_idx = find_elbow(x_vals, y_vals)
                elbow_d = int(x_vals[elbow_idx])
                c = PALETTE.get(ml, "#999999")
                ax.plot(x_vals, y_vals, "o-", label=f"{ml} (elbow={elbow_d})",
                        color=c, markersize=4, linewidth=1.5)
                ax.axvline(x=elbow_d, color=c, linestyle=":", alpha=0.4)
                elbow_rows.append({"dataset": ds, "model_loss": ml, "elbow_d": elbow_d})
            ax.set_title(dataset_label(ds), fontsize=10)
            ax.set_xlabel("Latent Dimension (d)")
            ax.set_ylabel("Reconstruction Loss")
            ax.legend(fontsize=6, loc="best")
            ax.grid(True, alpha=0.3)
        for j in range(n_ds, rows_n * cols):
            axes[j // cols][j % cols].set_visible(False)
        plt.suptitle("Reconstruction Loss Elbow Detection", fontsize=13, y=1.01)
        plt.tight_layout()
        _save_pub(fig, fig_dir, "figS4_recon_elbow")

    if elbow_rows:
        elbow_df = pd.DataFrame(elbow_rows)
        save_table(elbow_df, tab_dir / "reconstruction_elbow_points.csv")

    # ====================================================================
    # FIGURE 5: UMAP strip (1x5) — cell-type structure across datasets
    # ====================================================================
    section_header("Figure 5: UMAP Strip — Cell-Type Structure")
    _generate_umap_strip(repo, rec, fig_dir)

    # ---- Summary interpretation ----
    section_header("Publication Summary")
    summary_lines = [
        "KEY FINDINGS (scvi:nb model):",
        f"  Datasets analyzed: {df['dataset'].nunique()}",
        f"  Total successful experiments: {len(df)}",
        f"  K range: {int(df['K'].min())} to {int(df['K'].max())}",
        "",
        "RECOMMENDED LATENT DIMENSIONS:",
    ]
    for _, r in scvi_rec.iterrows():
        summary_lines.append(
            f"  {dataset_label(r['dataset'])}: K={int(r['K'])}, d*={int(r['recommended_d'])}"
        )
    if fit_table:
        summary_lines.append("")
        summary_lines.append("FORMULA FIT QUALITY:")
        for ft in fit_table:
            summary_lines.append(f"  {ft['formula']}: R²={ft['R2']}, RMSE={ft['RMSE']}")
    summary_lines.append("")
    summary_lines.append("CONCLUSION:")
    summary_lines.append("  No universal d=f(K) law — dataset-dependent narrow bands observed.")
    summary_lines.append("  Low-to-moderate K: d in [2,4]; atlas-scale K: d in [10,12].")

    summary = "\n".join(summary_lines)
    print(summary)
    (out_dir / "publication_summary.txt").write_text(summary)

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
