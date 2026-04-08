#!/usr/bin/env python3
"""
00 — Data Inventory & Readiness Audit

Verifies all required input files exist, profiles datasets,
and flags missing components before any analysis runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    display, find_repo_root, get_workroot, resolve_output_root,
    save_table, section_header,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="00 — Inventory & Readiness")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    repo = find_repo_root()
    workroot = get_workroot()
    out_root = resolve_output_root(repo, args.output_root)
    out_dir = out_root / "00_inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    section_header("00 — Data Inventory & Readiness Audit")
    print(f"  REPO      = {repo}")
    print(f"  WORKROOT  = {workroot}")
    print(f"  OUTPUT    = {out_root}")

    # ---- 1. Key global CSV inventory ----
    section_header("1. Key Global CSV Inventory")
    key_files = [
        "combined_curated_real_plus_phase4_enriched.csv",
        "phase1_pbmc3k_quick_results.csv",
        "phase2_scvi_pbmc12k_results.csv",
        "phase3_splatter_full_completed.csv",
        "recommended_d_by_K_primary.csv",
        "formula_fit_summary_primary.csv",
        "formula_fit_loo_primary.csv",
        "recommended_d_by_K.csv",
        "formula_fit_summary.csv",
        "recommended_d_by_K_exploratory.csv",
        "formula_fit_summary_exploratory.csv",
    ]
    rows = []
    for name in key_files:
        p = repo / "results" / "global" / "tables" / name
        if p.exists():
            df = pd.read_csv(p)
            status_str = ""
            if "status" in df.columns:
                status_str = json.dumps(
                    {str(k): int(v) for k, v in df["status"].value_counts(dropna=False).to_dict().items()}
                )
            rows.append({
                "file": name, "exists": True, "rows": len(df),
                "columns": len(df.columns), "size_kb": round(p.stat().st_size / 1024, 1),
                "status_counts": status_str,
            })
        else:
            rows.append({"file": name, "exists": False, "rows": np.nan, "columns": np.nan,
                         "size_kb": np.nan, "status_counts": ""})
    inv = pd.DataFrame(rows)
    display(inv)
    save_table(inv, out_dir / "global_csv_inventory.csv")

    # ---- 2. Main evidence table profile ----
    section_header("2. Main Evidence Table Profile")
    merged_path = repo / "results" / "global" / "tables" / "combined_curated_real_plus_phase4_enriched.csv"
    if merged_path.exists():
        df = pd.read_csv(merged_path)
        profile = {
            "total_rows": len(df),
            "status": df["status"].value_counts(dropna=False).to_dict(),
            "datasets": sorted(df["dataset"].unique().tolist()),
            "n_datasets": df["dataset"].nunique(),
            "model_types": sorted(df["model_type"].unique().tolist()),
            "loss_types": sorted(df["loss_type"].unique().tolist()),
            "latent_dims": sorted(df["latent_dim"].unique().tolist()),
            "seeds": sorted(df["seed"].unique().tolist()),
            "columns": list(df.columns),
        }
        print(json.dumps({k: v for k, v in profile.items() if k != "columns"}, indent=2, default=str))
        (out_dir / "main_evidence_profile.json").write_text(json.dumps(profile, indent=2, default=str))

        # Trust & external metrics by dataset
        trust = df.groupby("dataset")["label_trust"].value_counts(dropna=False).rename("count").reset_index()
        ext = df.groupby("dataset")["external_metrics_enabled"].value_counts(dropna=False).rename("count").reset_index()
        save_table(trust, out_dir / "trust_by_dataset.csv")
        save_table(ext, out_dir / "external_metrics_by_dataset.csv")

        # Coverage matrix
        coverage = df.groupby(["dataset", "model_type", "loss_type"]).agg(
            n_runs=("status", "count"),
            n_success=("status", lambda x: (x == "success").sum()),
            n_failed=("status", lambda x: (x == "failed").sum()),
            latent_dims_tested=("latent_dim", lambda x: sorted(x.unique().tolist())),
            n_seeds=("seed", "nunique"),
        ).reset_index()
        coverage["latent_dims_tested"] = coverage["latent_dims_tested"].apply(
            lambda x: ",".join(map(str, x))
        )
        save_table(coverage, out_dir / "experiment_coverage_matrix.csv")
    else:
        print(f"  WARNING: {merged_path} not found")

    # ---- 3. Run manifest index ----
    section_header("3. Run Manifest Index")
    run_dirs = sorted((repo / "results" / "runs").glob("run_*/manifest.json"))
    manifest_rows = []
    for p in run_dirs:
        d = json.loads(p.read_text())
        manifest_rows.append({
            "run_id": d.get("run_id", ""),
            "status": d.get("status", ""),
            "datasets": ",".join(map(str, d.get("datasets", []))),
            "total": d.get("total_experiments", d.get("total_experiments_partial", np.nan)),
            "success": d.get("successful", d.get("successful_partial", np.nan)),
            "failed": d.get("failed", d.get("failed_partial", np.nan)),
        })
    manifests = pd.DataFrame(manifest_rows)
    print(f"  Runs found: {len(manifests)}")
    display(manifests.tail(10))
    save_table(manifests, out_dir / "run_manifest_index.csv")

    # ---- 4. Phase 4 runtime artifacts ----
    section_header("4. Phase 4 Runtime Artifacts (WORK)")
    p4_rows = []
    if workroot is not None:
        patterns = {
            "ts1_stageA": str(workroot / "phase4/stageA/job_*/results/phase4_ts1_stageA_pilot.csv"),
            "ts1_stageB": str(workroot / "phase4/stageB/job_*/results/phase4_ts1_stageB_reduced.csv"),
            "aifi_reduced": str(workroot / "phase4/aifi_reduced/job_*/results/phase4_aifi_scvi_nb_reduced.csv"),
            "ts2_reduced": str(workroot / "phase4/ts2_reduced/job_*/results/phase4_ts2_lung_scvi_nb_reduced.csv"),
        }
        for name, pat in patterns.items():
            matches = sorted(glob.glob(pat), key=os.path.getmtime)
            p4_rows.append({"artifact": name, "count": len(matches),
                            "latest": matches[-1] if matches else "MISSING"})
    else:
        p4_rows.append({"artifact": "ALL", "count": 0, "latest": "WORK env var not set"})
    save_table(pd.DataFrame(p4_rows), out_dir / "phase4_runtime_index.csv")

    # ---- 5. Missing/attention items ----
    section_header("5. Missing or Attention Items")
    missing = []
    if not merged_path.exists():
        missing.append({"area": "primary_evidence", "status": "MISSING", "note": "Main CSV not found"})
    if workroot is None:
        missing.append({"area": "phase4_runtime", "status": "UNKNOWN", "note": "WORK not set"})

    # Check for latent export files
    latent_count = 0
    for root in [repo / "results" / "runs", repo / "results" / "datasets"]:
        if root.exists():
            for p in root.rglob("*latent*"):
                if p.is_file():
                    latent_count += 1
    if latent_count == 0:
        missing.append({"area": "latent_exports", "status": "MISSING",
                         "note": "No latent embedding files found. Proxy UMAP only."})
    save_table(pd.DataFrame(missing), out_dir / "attention_items.csv")

    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
