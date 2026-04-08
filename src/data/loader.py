"""
Dataset loading utilities for scRNA-seq data.

Supports built-in scanpy datasets and provides cell-type subsampling
for controlled experiments varying K.
"""

import logging
import os
from typing import Optional, List

import numpy as np
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


def _load_scvi_builtin(dataset_loader_name: str, data_dir: Optional[str] = None) -> ad.AnnData:
    """Load a scvi-tools built-in dataset by loader function name."""
    try:
        from scvi import data as scvi_data
    except ImportError as exc:
        raise ImportError(
            f"Dataset '{dataset_loader_name}' requires scvi-tools. "
            "Install/activate an environment with scvi-tools to use this loader."
        ) from exc

    if not hasattr(scvi_data, dataset_loader_name):
        raise ValueError(
            f"scvi.data has no loader '{dataset_loader_name}'. "
            "Check your scvi-tools version."
        )

    loader = getattr(scvi_data, dataset_loader_name)
    save_path = data_dir or "data"
    os.makedirs(save_path, exist_ok=True)
    return loader(save_path=save_path)




def _load_local_processed_only(dataset_name: str, data_dir: Optional[str] = None) -> ad.AnnData:
    """Load a dataset only from an existing processed file under <data_dir>/processed."""
    base_dir = data_dir or "data"
    path = os.path.join(base_dir, "processed", f"{dataset_name}.h5ad")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' expects a pre-onboarded processed file at {path}. "
            "Create/stage the file first (for large atlases, prefer WORK/TMPDIR workflow)."
        )
    return ad.read_h5ad(path)

def _load_scvi_pbmc_dataset(data_dir: Optional[str] = None) -> ad.AnnData:
    """Load scvi-tools PBMC benchmark dataset (12,039 cells)."""
    try:
        from scvi.data import pbmc_dataset
    except ImportError as exc:
        raise ImportError(
            "Dataset 'scvi_pbmc12k' requires scvi-tools. "
            "Install/activate an environment with scvi-tools to use this loader."
        ) from exc

    # Keep cache/download location explicit and HPC-friendly.
    save_path = data_dir or "data"
    os.makedirs(save_path, exist_ok=True)
    return pbmc_dataset(save_path=save_path)


def _load_scvi_purified_pbmc_dataset(data_dir: Optional[str] = None) -> ad.AnnData:
    """Load scvi-tools Purified PBMC dataset (curated labels + batch)."""
    return _load_scvi_builtin("purified_pbmc_dataset", data_dir=data_dir)


def _load_scvi_heart_cell_atlas_subsampled(data_dir: Optional[str] = None) -> ad.AnnData:
    """Load scvi-tools Heart Cell Atlas 20k subsampled dataset."""
    return _load_scvi_builtin("heart_cell_atlas_subsampled", data_dir=data_dir)


def _load_scvi_cortex(data_dir: Optional[str] = None) -> ad.AnnData:
    """Load scvi-tools cortex dataset (gold-standard labels)."""
    return _load_scvi_builtin("cortex", data_dir=data_dir)


# Registry of supported datasets and their metadata
DATASET_REGISTRY = {
    "pbmc3k": {
        "loader": lambda _data_dir=None: sc.datasets.pbmc3k(),
        "cell_type_key": None,  # default to clustering if no explicit label key
        "batch_key": None,
        "label_trust": "unknown",
        "tier": "tier_b",
        "description": "2,700 PBMCs from 10x Genomics",
    },
    "paul15": {
        "loader": lambda _data_dir=None: sc.datasets.paul15(),
        "cell_type_key": "paul15_clusters",
        "batch_key": None,
        "label_trust": "ground_truth",
        "tier": "tier_b",
        "description": "2,730 myeloid progenitor cells (Paul et al. 2015)",
    },
    "pbmc68k_reduced": {
        "loader": lambda _data_dir=None: sc.datasets.pbmc68k_reduced(),
        "cell_type_key": "louvain",
        "batch_key": "bulk_labels",  # coarse pseudo-batch/category in this small benchmark
        "label_trust": "untrusted",
        "tier": "tier_a",
        "description": "724 PBMCs (reduced 68k dataset)",
    },



    "ts2_lung": {
        "loader": lambda _data_dir=None: _load_local_processed_only("ts2_lung", data_dir=_data_dir),
        "cell_type_key": "cell_type",
        "batch_key": "donor_id",
        "label_trust": "unknown",
        "tier": "tier_a",
        "description": "Tabula Sapiens v2 Lung atlas (expects pre-onboarded processed file)",
    },
    "aifi_immune_full": {
        "loader": lambda _data_dir=None: _load_local_processed_only("aifi_immune_full", data_dir=_data_dir),
        "cell_type_key": "cell_type",
        "batch_key": "batch_id",
        "label_trust": "unknown",
        "tier": "tier_a",
        "description": "Allen Immune Health Atlas full (~1.8M cells, expects pre-onboarded processed file)",
    },
    "ts2_all_cells": {
        "loader": lambda _data_dir=None: _load_local_processed_only("ts2_all_cells", data_dir=_data_dir),
        "cell_type_key": "cell_type",
        "batch_key": "donor_id",
        "label_trust": "unknown",
        "tier": "tier_a",
        "description": "Tabula Sapiens v2 all-cells atlas (expects pre-onboarded processed file)",
    },
    "hca_kidney_healthy": {
        "loader": lambda _data_dir=None: _load_local_processed_only("hca_kidney_healthy", data_dir=_data_dir),
        "cell_type_key": "cell_type",
        "batch_key": "donor_id",
        "label_trust": "ground_truth",
        "tier": "tier_a",
        "description": (
            "HCA/KPMP healthy kidney atlas subset (expects pre-onboarded processed file "
            "with curated cell_type labels and donor_id batch key)"
        ),
    },
    "scvi_pbmc12k": {
        "loader": _load_scvi_pbmc_dataset,
        "cell_type_key": "labels",
        "batch_key": "batch",
        "label_trust": "ground_truth",
        "tier": "tier_b",
        "description": "scvi-tools PBMC benchmark (12,039 cells) with labels and batch metadata",
    },
    "scvi_purified_pbmc": {
        "loader": _load_scvi_purified_pbmc_dataset,
        "cell_type_key": "cell_types",
        "batch_key": "batch",
        "label_trust": "ground_truth",
        "tier": "tier_b",
        "description": "scvi-tools Purified PBMC dataset (~106k cells), curated labels + batch",
    },
    "scvi_heart_atlas_20k": {
        "loader": _load_scvi_heart_cell_atlas_subsampled,
        "cell_type_key": "cell_type",
        "batch_key": "donor",
        "label_trust": "ground_truth",
        "tier": "tier_b",
        "description": "scvi-tools Heart Cell Atlas subsampled (~18.6k cells), curated labels + donor",
    },
    "scvi_cortex": {
        "loader": _load_scvi_cortex,
        "cell_type_key": "labels",
        "batch_key": None,
        "label_trust": "ground_truth",
        "tier": "tier_b",
        "description": "scvi-tools cortex dataset (3,005 cells), curated gold-standard labels",
    },
}


def list_datasets_by_tiers(requested_tiers: List[str]) -> List[str]:
    """Return dataset names that belong to the requested tiers."""
    tier_set = set(requested_tiers)
    return [
        name for name, info in DATASET_REGISTRY.items()
        if info.get("tier") in tier_set
    ]


def load_dataset(name: str, data_dir: Optional[str] = None) -> ad.AnnData:
    """Load a scRNA-seq dataset by name."""
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")

    entry = DATASET_REGISTRY[name]
    logger.info(f"Loading dataset '{name}': {entry['description']}")
    adata = entry["loader"](data_dir)
    logger.info(f"  Shape: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def get_cell_type_key(dataset_name: str) -> Optional[str]:
    """Get the cell-type annotation key for a dataset, if available."""
    return DATASET_REGISTRY.get(dataset_name, {}).get("cell_type_key")


def get_batch_key(dataset_name: str) -> Optional[str]:
    """Get preferred batch key for a dataset, if available."""
    return DATASET_REGISTRY.get(dataset_name, {}).get("batch_key")


def get_label_trust(dataset_name: str) -> str:
    """Get label trust category for a dataset.

    Values:
      - ground_truth: trusted curated labels
      - untrusted: pseudo-labels or otherwise not suitable as primary ground truth
      - unknown: no explicit trust assignment
    """
    return str(DATASET_REGISTRY.get(dataset_name, {}).get("label_trust", "unknown"))


def resolve_label_key(
    adata: ad.AnnData,
    explicit_label_key: Optional[str],
    dataset_name: str,
) -> Optional[str]:
    """Resolve label key with priority explicit > dataset default > None."""
    if explicit_label_key:
        if explicit_label_key not in adata.obs.columns:
            raise ValueError(
                f"Requested label key '{explicit_label_key}' not present in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        return explicit_label_key

    default_key = get_cell_type_key(dataset_name)
    if default_key and default_key in adata.obs.columns:
        return default_key
    return None


def resolve_batch_key(
    adata: ad.AnnData,
    explicit_batch_key: Optional[str],
    dataset_name: str,
) -> Optional[str]:
    """Resolve batch key with priority explicit > dataset default > None."""
    if explicit_batch_key:
        if explicit_batch_key not in adata.obs.columns:
            raise ValueError(
                f"Requested batch key '{explicit_batch_key}' not present in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        return explicit_batch_key

    default_key = get_batch_key(dataset_name)
    if default_key and default_key in adata.obs.columns:
        return default_key
    return None


def subsample_cell_types(
    adata: ad.AnnData,
    cell_type_key: str,
    k: int,
    min_cells_per_type: int = 30,
    seed: int = 42,
) -> ad.AnnData:
    """Subsample an AnnData to keep only the K largest cell types."""
    rng = np.random.default_rng(seed)

    type_counts = adata.obs[cell_type_key].value_counts()
    eligible_types = type_counts[type_counts >= min_cells_per_type].index.tolist()

    if len(eligible_types) < k:
        raise ValueError(
            f"Dataset has only {len(eligible_types)} cell types with >= "
            f"{min_cells_per_type} cells, but K={k} requested."
        )

    # Keep K largest types, deterministic ordering by count
    selected_types = type_counts.loc[eligible_types].nlargest(k).index.tolist()

    # Optional deterministic downsample placeholder for future extension
    _ = rng  # keep deterministic rng hook for future per-type downsampling

    mask = adata.obs[cell_type_key].isin(selected_types)
    adata_sub = adata[mask].copy()

    logger.info(
        f"  Subsampled to K={k}: {adata_sub.n_obs} cells, "
        f"types: {selected_types}"
    )
    return adata_sub
