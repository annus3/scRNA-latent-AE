#!/usr/bin/env python3
"""
08 — Training Dynamics & Convergence

Analyzes training convergence, epochs, runtime, memory usage,
and operational reliability evidence from logs and result tables.
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
    display, find_repo_root, get_workroot, resolve_output_root,
    load_main_results, success_only, save_fig, save_table,
    section_header, dataset_label, get_model_palette,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="08 — Training Dynamics")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    workroot = get_workroot()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir, fig_dir, tab_dir = out_root / "08_training_dynamics", \
        out_root / "08_training_dynamics" / "figures", \
        out_root / "08_training_dynamics" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    section_header("08 — Training Dynamics & Convergence")

    df = success_only(load_main_results(repo))
    palette = get_model_palette(df)

    # ---- 1. Training summary table ----
    section_header("1. Training Summary by Model/Dataset")
    dyn = df.groupby(["dataset", "model_loss"], as_index=False).agg(
        mean_epochs=("total_epochs", "mean"),
        std_epochs=("total_epochs", "std"),
        mean_val_loss=("best_val_loss", "mean"),
        mean_runtime_s=("runtime_seconds", "mean"),
        mean_cpu_mb=("peak_cpu_mem_mb", "mean"),
        mean_gpu_mb=("peak_gpu_mem_mb", "mean"),
        n=("seed", "count"),
    )
    display(dyn.sort_values(["dataset", "model_loss"]))
    save_table(dyn, tab_dir / "training_dynamics_summary.csv")

    # ---- 2. Epoch distribution ----
    section_header("2. Epoch Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="model_loss", y="total_epochs", ax=axes[0], palette=palette)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_title("Epochs to Convergence")

    sns.boxplot(data=df, x="model_loss", y="runtime_seconds", ax=axes[1], palette=palette)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_title("Runtime (seconds)")
    plt.suptitle("Training Efficiency", fontsize=13)
    plt.tight_layout()
    save_fig(fig_dir / "epoch_runtime_distribution.png")

    # ---- 3. Memory usage ----
    section_header("3. Memory Usage")
    mem = df.groupby(["dataset", "model_loss"], as_index=False).agg(
        cpu_mb_mean=("peak_cpu_mem_mb", "mean"),
        cpu_mb_max=("peak_cpu_mem_mb", "max"),
        gpu_mb_mean=("peak_gpu_mem_mb", "mean"),
        gpu_mb_max=("peak_gpu_mem_mb", "max"),
    )
    save_table(mem, tab_dir / "memory_usage.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mem_plot = mem.dropna(subset=["cpu_mb_mean"])
    if len(mem_plot):
        sns.barplot(data=mem_plot, x="model_loss", y="cpu_mb_mean", hue="dataset", ax=axes[0])
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
        axes[0].set_title("Peak CPU Memory (MB)")
        axes[0].legend(fontsize=7)

    mem_plot_gpu = mem.dropna(subset=["gpu_mb_mean"])
    if len(mem_plot_gpu):
        sns.barplot(data=mem_plot_gpu, x="model_loss", y="gpu_mb_mean", hue="dataset", ax=axes[1])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
        axes[1].set_title("Peak GPU Memory (MB)")
        axes[1].legend(fontsize=7)
    plt.tight_layout()
    save_fig(fig_dir / "memory_usage.png")

    # ---- 4. Runtime vs dataset scale ----
    section_header("4. Runtime vs Dataset Scale")
    rt = df.groupby(["dataset", "model_loss"], as_index=False).agg(
        n_cells=("n_cells", "first"),
        mean_runtime=("runtime_seconds", "mean"),
    )
    plt.figure(figsize=(10, 6))
    for ml in sorted(rt["model_loss"].unique()):
        sub = rt[rt["model_loss"] == ml].sort_values("n_cells")
        color = palette.get(ml, "#999999")
        plt.plot(sub["n_cells"], sub["mean_runtime"], "o-", label=ml, color=color)
    plt.xlabel("Number of Cells")
    plt.ylabel("Mean Runtime (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Computational Scaling: Runtime vs Dataset Size")
    plt.legend(fontsize=8)
    save_fig(fig_dir / "runtime_vs_scale.png")

    # ---- 5. Convergence: val_loss vs latent_dim ----
    section_header("5. Validation Loss vs Latent Dimension")
    val_agg = df.groupby(["dataset", "model_loss", "latent_dim"], as_index=False).agg(
        mean_val=("best_val_loss", "mean"),
        std_val=("best_val_loss", "std"),
    )
    save_table(val_agg, tab_dir / "val_loss_vs_latent.csv")

    # Convergence figure: epochs to convergence vs latent_dim per dataset
    datasets = sorted(df["dataset"].unique())
    n_ds = len(datasets)
    cols = min(3, n_ds)
    rows_n = max(1, int(np.ceil(n_ds / cols)))
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 4 * rows_n), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // cols][i % cols]
        sub = df[df["dataset"] == ds]
        epoch_agg = sub.groupby(["model_loss", "latent_dim"], as_index=False)["total_epochs"].mean()
        for ml in sorted(epoch_agg["model_loss"].unique()):
            ml_sub = epoch_agg[epoch_agg["model_loss"] == ml].sort_values("latent_dim")
            color = palette.get(ml, "#999999")
            ax.plot(ml_sub["latent_dim"], ml_sub["total_epochs"], "o-", label=ml,
                    color=color, markersize=3, linewidth=1.2)
        ax.set_title(dataset_label(ds), fontsize=10)
        ax.set_xlabel("Latent Dim (d)")
        ax.set_ylabel("Epochs")
        ax.legend(fontsize=6, ncol=2)
    for j in range(n_ds, rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)
    plt.suptitle("Convergence Speed: Epochs vs Latent Dimension", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(fig_dir / "convergence_epochs_vs_d.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "convergence_epochs_vs_d.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] convergence_epochs_vs_d.png")

    # Early stopping analysis
    section_header("5b. Early Stopping Analysis")
    max_epochs_default = 100
    es_summary = df.groupby(["dataset", "model_loss"], as_index=False).agg(
        mean_epochs=("total_epochs", "mean"),
        max_epochs=("total_epochs", "max"),
        min_epochs=("total_epochs", "min"),
        pct_early_stop=("total_epochs", lambda x: 100 * (x < max_epochs_default).mean()),
    )
    save_table(es_summary, tab_dir / "early_stopping_summary.csv")

    # ---- 6. SLURM log evidence scan ----
    section_header("6. SLURM Log Evidence Scan")
    patterns = ("oom_kill", "Out Of Memory", "Killed", "Traceback", "ERROR", "FAILED")
    log_dir = repo / "logs"
    log_rows = []
    if log_dir.exists():
        for lf in sorted(log_dir.glob("slurm_*")):
            if lf.stat().st_size > 50 * 1024 * 1024:  # skip > 50MB
                continue
            try:
                with open(lf, "r", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if any(tok in line for tok in patterns):
                            log_rows.append({
                                "log_file": lf.name, "line": i,
                                "match": line.strip()[:200],
                            })
            except Exception:
                pass
    log_df = pd.DataFrame(log_rows)
    print(f"  Log evidence hits: {len(log_df)}")
    save_table(log_df, tab_dir / "slurm_log_evidence.csv")

    # ---- 7. Failure analysis ----
    section_header("7. Failure Analysis")
    all_df = load_main_results(repo)
    failed = all_df[all_df["status"] == "failed"].copy()
    if len(failed):
        fail_summary = failed.groupby(["dataset", "model_loss", "error_type"]).size().reset_index(name="count")
        display(fail_summary)
        save_table(fail_summary, tab_dir / "failure_summary.csv")
        save_table(failed[["dataset", "model_loss", "latent_dim", "seed", "error_type", "error"]],
                   tab_dir / "failure_detail.csv")
    else:
        print("  No failures in main evidence table.")
        pd.DataFrame(columns=["dataset", "model_loss", "error_type", "count"]).to_csv(
            tab_dir / "failure_summary.csv", index=False)

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
