"""
Preprocessing pipeline for scRNA-seq data.

Implements: QC -> normalize -> log1p -> HVG -> optional scaling/PCA -> neighbors -> labels.
Stores raw counts in a layer for NB/ZINB models.
"""

import os
import json
import hashlib
import logging
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple

import scanpy as sc
import anndata as ad

from src.config import PreprocessingConfig

logger = logging.getLogger(__name__)

_PREPROCESS_FINGERPRINT_VERSION = "preprocess_v1"


def _preprocess_fingerprint_payload(
    config: PreprocessingConfig,
    cell_type_key: Optional[str],
    batch_key: Optional[str],
) -> Dict[str, Any]:
    """Build a stable payload for preprocessing-cache fingerprinting."""
    payload = asdict(config)
    payload.update(
        {
            "cell_type_key": cell_type_key or "<none>",
            "batch_key": batch_key or "<none>",
            "fingerprint_version": _PREPROCESS_FINGERPRINT_VERSION,
        }
    )
    return payload


def _fingerprint_from_payload(payload: Dict[str, Any]) -> str:
    """Generate a compact SHA256 fingerprint from a payload dictionary."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_preprocess_fingerprint(
    config: PreprocessingConfig,
    cell_type_key: Optional[str] = None,
    batch_key: Optional[str] = None,
) -> str:
    """Public helper to compute preprocessing fingerprint from config + label context."""
    payload = _preprocess_fingerprint_payload(config, cell_type_key, batch_key)
    return _fingerprint_from_payload(payload)


def check_cached_preprocess_fingerprint(
    adata: ad.AnnData,
    config: PreprocessingConfig,
    cell_type_key: Optional[str] = None,
    batch_key: Optional[str] = None,
) -> Tuple[bool, str]:
    """Check whether cached processed data matches current preprocessing settings.

    Returns:
        (is_match, message)
    """
    expected_payload = _preprocess_fingerprint_payload(config, cell_type_key, batch_key)
    expected = _fingerprint_from_payload(expected_payload)

    observed = adata.uns.get("preprocess_fingerprint")
    if not observed:
        return False, (
            f"missing fingerprint in cached file (expected={expected}). "
            "This likely came from an older preprocessing run."
        )

    observed_str = str(observed)
    if observed_str != expected:
        return False, (
            f"fingerprint mismatch (cached={observed_str}, expected={expected}). "
            "Cached preprocessing may be stale for current config."
        )

    return True, f"fingerprint match ({expected})"


def _apply_optional_integration(adata: ad.AnnData, integration_mode: str) -> ad.AnnData:
    """Apply optional batch integration placeholders safely."""
    if integration_mode == "none":
        return adata

    if integration_mode == "harmony":
        logger.warning(
            "integration_mode='harmony' requested but not enabled in this baseline. "
            "Proceeding without integration."
        )
        return adata

    if integration_mode == "bbknn":
        logger.warning(
            "integration_mode='bbknn' requested but not enabled in this baseline. "
            "Proceeding with standard neighbors."
        )
        return adata

    raise ValueError(f"Unknown integration_mode: {integration_mode}")


def preprocess(
    adata: ad.AnnData,
    config: PreprocessingConfig,
    cell_type_key: Optional[str] = None,
    batch_key: Optional[str] = None,
) -> ad.AnnData:
    """Run full preprocessing pipeline for scRNA-seq data."""
    adata = adata.copy()
    logger.info("Starting preprocessing pipeline...")

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=config.min_genes_per_cell)
    n_cells_removed = n_before - adata.n_obs

    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=config.min_cells_per_gene)
    n_genes_removed = n_genes_before - adata.n_vars

    logger.info(
        f"  QC: removed {n_cells_removed} cells, {n_genes_removed} genes. "
        f"Remaining: {adata.n_obs} cells × {adata.n_vars} genes"
    )

    # Keep raw counts for count likelihood losses
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=config.target_sum)
    logger.info(f"  Normalized to {config.target_sum} counts per cell")

    sc.pp.log1p(adata)
    logger.info("  Applied log1p transformation")

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=config.n_top_genes,
            flavor=config.hvg_flavor,
            layer="counts",
        )
    except (ImportError, ValueError) as exc:
        # seurat_v3 depends on skmisc.loess; on minimal HPC envs this package is
        # sometimes absent. Fallback keeps pipeline runnable.
        logger.warning(
            "HVG selection with flavor='%s' on raw counts failed (%s). "
            "Falling back to flavor='seurat' on log-normalized X.",
            config.hvg_flavor,
            exc,
        )
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=config.n_top_genes,
            flavor="seurat",
        )
    n_hvg = int(adata.var["highly_variable"].sum())
    logger.info(f"  Selected {n_hvg} highly variable genes ({config.hvg_flavor})")

    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info(f"  Subsetted to HVGs: {adata.n_obs} × {adata.n_vars}")

    if config.scale_policy == "full":
        sc.pp.scale(adata, max_value=10)
        logger.info("  Applied full scaling on adata.X")

    if config.run_pca:
        adata_pca = adata.copy()
        if config.scale_policy in {"none", "pca_only"}:
            sc.pp.scale(adata_pca, max_value=10)
        sc.tl.pca(adata_pca, n_comps=max(2, min(config.n_pcs, adata.n_vars - 1)))
        adata.obsm["X_pca"] = adata_pca.obsm["X_pca"]
        adata.varm["PCs"] = adata_pca.varm["PCs"]
        adata.uns["pca"] = adata_pca.uns["pca"]
        logger.info(f"  Computed PCA ({adata.obsm['X_pca'].shape[1]} components)")

    adata = _apply_optional_integration(adata, config.integration_mode)

    if config.run_pca and "X_pca" in adata.obsm:
        sc.pp.neighbors(
            adata,
            n_neighbors=config.n_neighbors,
            n_pcs=min(40, adata.obsm["X_pca"].shape[1]),
        )
    else:
        sc.pp.neighbors(adata, n_neighbors=config.n_neighbors)

    if cell_type_key and cell_type_key in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[cell_type_key].astype(str)
        adata.uns["cell_type_source"] = f"provided:{cell_type_key}"
        K = int(adata.obs["cell_type"].nunique())
        logger.info(f"  Using labels from '{cell_type_key}': K={K}")
    else:
        try:
            sc.tl.leiden(adata, resolution=config.leiden_resolution)
            adata.obs["cell_type"] = adata.obs["leiden"].astype(str)
            adata.uns["cell_type_source"] = "inferred:leiden"
        except ImportError:
            logger.warning("leidenalg not installed, falling back to Louvain")
            sc.tl.louvain(adata, resolution=config.leiden_resolution)
            adata.obs["cell_type"] = adata.obs["louvain"].astype(str)
            adata.uns["cell_type_source"] = "inferred:louvain"
        K = int(adata.obs["cell_type"].nunique())
        logger.info(f"  Inferred clusters: K={K}")

    if batch_key:
        if batch_key not in adata.obs.columns:
            raise ValueError(
                f"Batch key '{batch_key}' requested but not present in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        adata.obs["batch"] = adata.obs[batch_key].astype(str)
        logger.info(f"  Preserved batch metadata from '{batch_key}'")

    fingerprint_payload = _preprocess_fingerprint_payload(config, cell_type_key, batch_key)
    adata.uns["preprocess_fingerprint_meta"] = fingerprint_payload
    adata.uns["preprocess_fingerprint"] = _fingerprint_from_payload(fingerprint_payload)

    logger.info(
        f"Preprocessing complete: {adata.n_obs} cells × {adata.n_vars} genes, K={K}, "
        f"label_source={adata.uns.get('cell_type_source', 'unknown')}"
    )
    return adata


def save_processed(adata: ad.AnnData, path: str) -> None:
    """Save processed AnnData to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    adata.write_h5ad(path)
    logger.info(f"  Saved processed data to {path}")


def load_processed(path: str, backed: Optional[str] = None) -> ad.AnnData:
    """
    Load processed AnnData from disk.

    Args:
        path: Path to .h5ad file.
        backed: Optional anndata backed mode ("r" or "r+").
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")

    adata = ad.read_h5ad(path, backed=backed)
    mode = "backed" if backed else "in_memory"
    logger.info(f"  Loaded processed data ({mode}): {adata.n_obs} × {adata.n_vars}")
    return adata
