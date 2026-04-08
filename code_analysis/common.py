"""
Shared utilities for the analysis pipeline.

Provides: path resolution, plotting defaults, data loading, display helpers.
All analysis scripts import from this module.
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Publication-quality plotting defaults
# ---------------------------------------------------------------------------
PALETTE = {
    "scvi:nb": "#2274A5",
    "scvi:zinb": "#6BA3BE",
    "ae:mse": "#E85D04",
    "ae:nb": "#F48C06",
    "ae:zinb": "#FAA307",
    "vae:mse": "#7B2D8E",
    "vae:nb": "#9D4EDD",
    "vae:zinb": "#C77DFF",
}

MODEL_ORDER = ["scvi:nb", "scvi:zinb", "ae:nb", "ae:zinb", "ae:mse", "vae:nb", "vae:zinb", "vae:mse"]

DATASET_DISPLAY = {
    "paul15": "Paul15",
    "pbmc3k": "PBMC3k",
    "scvi_pbmc12k": "PBMC12k",
    "ts1_all_cells": "TS1 (Tabula Sapiens)",
    "ts2_lung": "TS2 Lung",
    "aifi_immune_full": "AIFI Immune",
    "splatter_k04": "Splatter K=4",
    "splatter_k08": "Splatter K=8",
    "splatter_k12": "Splatter K=12",
}

FORMULA_DISPLAY = {
    "linear": r"$d = a \cdot K + b$",
    "power": r"$d = a \cdot K^b$",
    "sqrt": r"$d = a \cdot \sqrt{K} + b$",
    "logk": r"$d = a \cdot \log(K) + b$",
}


def setup_plotting(context: str = "paper", font_scale: float = 1.2) -> None:
    """Configure publication-quality plot aesthetics."""
    sns.set_theme(style="whitegrid", context=context, font_scale=font_scale)
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
    })


setup_plotting()


# ---------------------------------------------------------------------------
# Display helper (works in both IPython and plain terminal)
# ---------------------------------------------------------------------------
try:
    from IPython.display import display as _ipython_display

    def display(obj):
        _ipython_display(obj)
except Exception:
    def display(obj):
        if hasattr(obj, "to_string"):
            try:
                print(obj.to_string())
                return
            except Exception:
                pass
        print(obj)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up from start to find the repo root by sentinel files."""
    start = Path.cwd() if start is None else start
    for p in [start, *start.parents]:
        if (p / "config" / "default.yaml").exists() and (p / "scripts" / "run_experiment.py").exists():
            return p
    raise RuntimeError("Could not locate repository root")


def get_workroot() -> Optional[Path]:
    """Return $WORK/sc_autoencoder_project or None."""
    work = os.environ.get("WORK")
    if not work:
        return None
    return Path(work) / "sc_autoencoder_project"


def resolve_output_root(repo_root: Path, output_root: Optional[str] = None) -> Path:
    """Create and return a timestamped output directory."""
    if output_root:
        out = Path(output_root).expanduser()
    else:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out = repo_root / "results" / "code_analysis_runs" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_dirs(base: Path, slug: str) -> tuple[Path, Path, Path]:
    """Create and return (out_dir, fig_dir, tab_dir) for a script."""
    out = base / slug
    fig = out / "figures"
    tab = out / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    return out, fig, tab


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_main_results(repo_root: Path) -> pd.DataFrame:
    """Load the combined curated evidence CSV (all phases, enriched)."""
    p = repo_root / "results" / "global" / "tables" / "combined_curated_real_plus_phase4_enriched.csv"
    if not p.exists():
        raise FileNotFoundError(f"Main results CSV not found: {p}")
    df = pd.read_csv(p)
    df["model_loss"] = df["model_type"].astype(str) + ":" + df["loss_type"].astype(str)
    return df


def load_splatter_results(repo_root: Path) -> pd.DataFrame:
    """Load Phase 3 Splatter K-sweep results."""
    p = repo_root / "results" / "global" / "tables" / "phase3_splatter_full_completed.csv"
    if not p.exists():
        raise FileNotFoundError(f"Splatter results CSV not found: {p}")
    df = pd.read_csv(p)
    df["model_loss"] = df["model_type"].astype(str) + ":" + df["loss_type"].astype(str)
    return df


def load_recommendations(repo_root: Path) -> pd.DataFrame:
    p = repo_root / "results" / "global" / "tables" / "recommended_d_by_K_primary.csv"
    return pd.read_csv(p)


def load_formula_fit(repo_root: Path) -> pd.DataFrame:
    p = repo_root / "results" / "global" / "tables" / "formula_fit_summary_primary.csv"
    return pd.read_csv(p)


def load_formula_loo(repo_root: Path) -> pd.DataFrame:
    p = repo_root / "results" / "global" / "tables" / "formula_fit_loo_primary.csv"
    return pd.read_csv(p)


def load_phase_results(repo_root: Path, phase: str) -> Optional[pd.DataFrame]:
    """Load a phase-specific CSV from global tables. Returns None if missing."""
    mapping = {
        "phase1": "phase1_pbmc3k_quick_results.csv",
        "phase2": "phase2_scvi_pbmc12k_results.csv",
        "phase3": "phase3_splatter_full_completed.csv",
    }
    name = mapping.get(phase)
    if not name:
        return None
    p = repo_root / "results" / "global" / "tables" / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def success_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to successful runs only."""
    return df[df["status"] == "success"].copy()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def save_fig(path: Path, dpi: int = 300, tight: bool = True, also_pdf: bool = True) -> None:
    """Save current figure (PNG + optional PDF for publication) and close."""
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        pdf_path = path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {path.name}")


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save DataFrame to CSV and optionally LaTeX."""
    df.to_csv(path, index=index)
    # Also save LaTeX version for thesis
    latex_path = path.with_suffix(".tex")
    try:
        df.to_latex(latex_path, index=index, float_format="%.4f", escape=True)
    except Exception:
        pass
    print(f"  [tab] {path.name} ({len(df)} rows)")


def get_model_palette(df: pd.DataFrame) -> dict:
    """Return a palette dict filtered to model_loss values present in df."""
    present = set(df["model_loss"].unique())
    return {k: v for k, v in PALETTE.items() if k in present}


def dataset_label(name: str) -> str:
    """Human-friendly dataset name."""
    return DATASET_DISPLAY.get(name, name)


def model_loss_order(values):
    """Sort model_loss values by canonical order, unknown at end."""
    return sorted(values, key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)


def load_all_experiment_results(repo_root: Path) -> pd.DataFrame:
    """Load experiment_results_all.csv (includes exploratory datasets)."""
    p = repo_root / "results" / "global" / "tables" / "experiment_results_all.csv"
    if not p.exists():
        raise FileNotFoundError(f"experiment_results_all.csv not found: {p}")
    df = pd.read_csv(p)
    df["model_loss"] = df["model_type"].astype(str) + ":" + df["loss_type"].astype(str)
    return df


def find_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """Find the elbow point using max perpendicular distance to chord.

    Args:
        x: 1-D array of x values (e.g. latent dimensions), must be sorted.
        y: 1-D array of corresponding y values (e.g. reconstruction loss).

    Returns:
        Index into x/y of the detected elbow point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return int(np.argmin(y))
    # Normalize to [0,1]
    x_n = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    y_n = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    # Chord from first to last point
    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return int(np.argmin(y))
    line_unit = line_vec / line_len
    # Perpendicular distance for each point
    dists = np.abs(np.cross(line_unit, p1 - np.column_stack([x_n, y_n])))
    return int(np.argmax(dists))


def section_header(title: str) -> None:
    """Print a visible section header for HPC log readability."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
