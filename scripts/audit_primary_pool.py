#!/usr/bin/env python3
"""Audit PRIMARY inclusion from an existing experiment results CSV.

This script is read-only and meant to make dataset inclusion/exclusion explicit
under current trust semantics and primary filtering settings.
"""

import argparse
import fnmatch
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.loader import get_label_trust


def _matches_any_pattern(value: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    return any(fnmatch.fnmatch(str(value), pat) for pat in patterns)


def _derive_trust(row: pd.Series) -> str:
    trust = str(row.get("label_trust", "")).strip().lower()
    if trust:
        return trust

    dataset = str(row.get("dataset", ""))
    mapped = str(get_label_trust(dataset) or "unknown").lower()
    if mapped in {"ground_truth", "untrusted"}:
        return mapped

    source = str(row.get("label_source", ""))
    if dataset.lower().startswith("splatter_"):
        return "untrusted"
    if source.startswith("provided:cell_ontology_class"):
        return "ground_truth"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PRIMARY inclusion from results CSV")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--csv", type=str, required=True, help="Path to results CSV")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of table")
    args = parser.parse_args()

    config = load_config(args.config)
    df = pd.read_csv(args.csv)

    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()

    if df.empty:
        print("No successful rows found.")
        return

    df["label_trust_eff"] = df.apply(_derive_trust, axis=1)
    if "is_ground_truth_labels" in df.columns:
        gt_mask = df["is_ground_truth_labels"].fillna(False).astype(bool)
    else:
        gt_mask = df["label_trust_eff"].eq("ground_truth")

    include_source_patterns = list(config.experiment.auto_d_primary_include_label_sources or [])
    if "label_source" in df.columns and include_source_patterns:
        src_mask = df["label_source"].astype(str).apply(
            lambda x: _matches_any_pattern(x, include_source_patterns)
        )
    else:
        src_mask = pd.Series(True, index=df.index)

    exclude_patterns = list(config.experiment.auto_d_primary_exclude_dataset_patterns or [])
    ds_excl = df["dataset"].astype(str).apply(lambda x: _matches_any_pattern(x, exclude_patterns))

    if config.experiment.auto_d_primary_require_curated_labels:
        primary_mask = gt_mask & src_mask & (~ds_excl)
    else:
        primary_mask = src_mask & (~ds_excl)

    df["primary_included"] = primary_mask

    rows = []
    for dataset, sub in df.groupby("dataset"):
        label_keys = sorted({str(v) for v in sub.get("label_key", pd.Series(dtype=object)).dropna().unique() if str(v)})
        sources = sorted({str(v) for v in sub.get("label_source", pd.Series(dtype=object)).dropna().unique() if str(v)})
        trusts = sorted(set(sub["label_trust_eff"].astype(str).tolist()))
        k_vals = sorted({int(v) for v in sub.get("K", pd.Series(dtype=float)).dropna().tolist()})

        ext = "n/a"
        if "external_metrics_enabled" in sub.columns:
            ext = str(sub["external_metrics_enabled"].value_counts(dropna=False).to_dict())

        included = bool(sub["primary_included"].any())
        reason = "included"
        if not included:
            if config.experiment.auto_d_primary_require_curated_labels and not bool(gt_mask[sub.index].any()):
                reason = "excluded_no_ground_truth"
            elif bool(ds_excl[sub.index].any()):
                reason = "excluded_by_dataset_pattern"
            elif include_source_patterns and not bool(src_mask[sub.index].any()):
                reason = "excluded_by_label_source_pattern"
            else:
                reason = "excluded"

        rows.append(
            {
                "dataset": dataset,
                "rows_success": int(len(sub)),
                "label_key": ";".join(label_keys),
                "label_source": ";".join(sources),
                "label_trust": ";".join(trusts),
                "K_values": k_vals,
                "external_metrics_enabled": ext,
                "primary_included": included,
                "why": reason,
            }
        )

    out = pd.DataFrame(rows).sort_values("dataset")
    if args.json:
        print(out.to_json(orient="records", indent=2))
    else:
        print(out.to_string(index=False))

    n_ds = int(out["primary_included"].sum())
    k_primary = sorted(
        {
            int(v)
            for _, sub in df[df["primary_included"]].groupby("dataset")
            for v in sub.get("K", pd.Series(dtype=float)).dropna().tolist()
        }
    )
    print("\nSummary:")
    print(f"  primary_datasets={n_ds}")
    print(f"  primary_unique_K={k_primary}")


if __name__ == "__main__":
    main()
