#!/usr/bin/env python3
import argparse

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def split_list(s):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def pick_col(obs_cols, primary, fallbacks, kind):
    if primary in obs_cols:
        return primary
    for c in fallbacks:
        if c in obs_cols:
            return c
    raise SystemExit(f"Missing required {kind} column. Tried: {[primary] + fallbacks}")


def _to_csr_float32(x):
    # Handle AnnData backed sparse datasets (_CSRDataset / _CSCDataset).
    # They expose .to_memory() but are not scipy sparse instances directly.
    if hasattr(x, "to_memory"):
        x = x.to_memory()
    if sp.issparse(x):
        return x.tocsr().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _build_counts_from_x(x, transform: str):
    if sp.issparse(x):
        counts = x.tocsr(copy=True)
        if counts.data.size:
            np.clip(counts.data, 0, None, out=counts.data)
            if transform == "expm1_round":
                counts.data = np.expm1(counts.data)
            counts.data = np.rint(counts.data).astype(np.float32, copy=False)
        return counts

    arr = np.asarray(x)
    arr = np.clip(arr, 0, None)
    if transform == "expm1_round":
        arr = np.expm1(arr)
    return np.rint(arr).astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-primary", required=True)
    ap.add_argument("--label-fallback", default="")
    ap.add_argument("--batch-primary", required=True)
    ap.add_argument("--batch-fallback", default="")
    ap.add_argument("--batch-canonical", required=True)
    ap.add_argument("--min-genes-per-cell", type=int, default=200)
    ap.add_argument("--min-cells-per-gene", type=int, default=3)
    ap.add_argument("--n-top-genes", type=int, default=2000)
    ap.add_argument("--target-sum", type=float, default=10000.0)
    ap.add_argument(
        "--counts-source-policy",
        choices=["x_only", "prefer_raw"],
        default="x_only",
        help="Use X only (safe for large atlases) or prefer raw.X when available.",
    )
    ap.add_argument(
        "--counts-transform",
        choices=["round", "expm1_round"],
        default="round",
        help="How to derive counts from selected matrix.",
    )
    ap.add_argument(
        "--skip-filter-hvg",
        action="store_true",
        help="Skip filter_cells/filter_genes/hvg steps.",
    )
    ap.add_argument(
        "--skip-normalize-log1p",
        action="store_true",
        help="Preserve X and skip normalize_total/log1p.",
    )
    args = ap.parse_args()

    # Backed read avoids materializing massive raw count matrices by default.
    src = ad.read_h5ad(args.source, backed="r")
    obs_cols = set(src.obs.columns)

    label_col = pick_col(obs_cols, args.label_primary, split_list(args.label_fallback), "label")
    batch_col = pick_col(obs_cols, args.batch_primary, split_list(args.batch_fallback), "batch")
    source_label_ontology_key = ""
    source_label_ontology_nonnull_fraction = float("nan")
    if label_col == "cell_type" and "cell_type_ontology_term_id" in obs_cols:
        source_label_ontology_key = "cell_type_ontology_term_id"
        try:
            term_series = src.obs[source_label_ontology_key]
            source_label_ontology_nonnull_fraction = float(term_series.notna().mean())
        except Exception:
            source_label_ontology_nonnull_fraction = float("nan")

    use_raw = args.counts_source_policy == "prefer_raw" and src.raw is not None
    if use_raw:
        x_source = src.raw.X
        var = src.raw.var.copy()
        counts_source = "raw.X"
    else:
        x_source = src.X
        var = src.var.copy()
        counts_source = "X"

    x_mat = _to_csr_float32(x_source)
    adata = ad.AnnData(X=x_mat, obs=src.obs.copy(), var=var)

    adata.obs["cell_type"] = adata.obs[label_col].astype(str).astype("category")
    adata.obs[args.batch_canonical] = adata.obs[batch_col].astype(str).astype("category")
    adata.obs["batch"] = adata.obs[args.batch_canonical].copy()

    if not args.skip_filter_hvg:
        sc.pp.filter_cells(adata, min_genes=args.min_genes_per_cell)
        sc.pp.filter_genes(adata, min_cells=args.min_cells_per_gene)
        if adata.n_vars > args.n_top_genes:
            try:
                sc.pp.highly_variable_genes(
                    adata, n_top_genes=args.n_top_genes, flavor="seurat_v3", subset=True
                )
            except Exception:
                sc.pp.highly_variable_genes(
                    adata, n_top_genes=args.n_top_genes, flavor="seurat", subset=True
                )

    adata.layers["counts"] = _build_counts_from_x(adata.X, args.counts_transform)

    if not args.skip_normalize_log1p:
        sc.pp.normalize_total(adata, target_sum=args.target_sum)
        sc.pp.log1p(adata)

    if adata.layers["counts"].shape != adata.X.shape:
        raise SystemExit(
            f"counts/X shape mismatch: {adata.layers['counts'].shape} vs {adata.X.shape}"
        )

    adata.uns["cell_type_source"] = f"provided:{label_col}"
    adata.uns["batch_source"] = f"provided:{batch_col}"
    adata.uns["onboard_contract"] = {
        "label_key": "cell_type",
        "batch_key": args.batch_canonical,
        "source_label_key": label_col,
        "source_batch_key": batch_col,
        "source_label_ontology_key": source_label_ontology_key,
        "source_label_ontology_nonnull_fraction": source_label_ontology_nonnull_fraction,
        "counts_layer": "counts",
        "counts_source": counts_source,
        "counts_source_policy": args.counts_source_policy,
        "counts_transform": args.counts_transform,
        "steps": (
            ["filter_cells", "filter_genes", "hvg_subset"] if not args.skip_filter_hvg else []
        ) + (
            ["normalize_total", "log1p"]
            if not args.skip_normalize_log1p
            else ["preserve_x"]
        ),
    }

    adata.write_h5ad(args.out, compression="lzf")

    chk = ad.read_h5ad(args.out, backed="r")
    need_obs = ["cell_type", args.batch_canonical]
    miss = [k for k in need_obs if k not in chk.obs.columns]
    if miss:
        raise SystemExit(f"Missing required obs keys in output: {miss}")
    if "counts" not in chk.layers:
        raise SystemExit("Missing layers['counts'] in output")
    if chk.layers["counts"].shape != chk.X.shape:
        raise SystemExit("Output counts/X shape mismatch")

    print("OK")
    print("shape=", chk.shape)
    print("label_source=", f"provided:{label_col}")
    print("batch_source=", f"provided:{batch_col}")


if __name__ == "__main__":
    main()
