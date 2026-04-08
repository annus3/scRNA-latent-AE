"""
PyTorch Dataset and DataLoader utilities for scRNA-seq data.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import anndata as ad

logger = logging.getLogger(__name__)


def _to_dense_array(X, dense_policy: str = "auto") -> np.ndarray:
    """Convert matrix to numpy array based on dense conversion policy."""
    is_sparse = hasattr(X, "toarray")
    if dense_policy == "always" and is_sparse:
        return np.asarray(X.toarray(), dtype=np.float32)
    if dense_policy == "never":
        if is_sparse:
            raise ValueError(
                "dense_conversion_policy='never' is incompatible with eager dense conversion. "
                "Use preload=False in SingleCellDataset/make_dataloaders for lazy row loading."
            )
        return np.asarray(X, dtype=np.float32)

    # auto
    if is_sparse:
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


class SingleCellDataset(Dataset):
    """Torch dataset wrapper over AnnData matrix/layer."""

    def __init__(
        self,
        adata: ad.AnnData,
        layer: Optional[str] = None,
        dense_policy: str = "auto",
        preload: bool = True,
    ):
        X = adata.layers[layer] if layer else adata.X
        self._preload = preload

        if preload:
            self.X = torch.from_numpy(_to_dense_array(X, dense_policy=dense_policy))
            self.n_cells, self.n_genes = self.X.shape
        else:
            # Keep backing storage (sparse/backed) and materialize rows on demand.
            self._X = X
            self.n_cells, self.n_genes = adata.n_obs, adata.n_vars

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._preload:
            return self.X[idx]

        row = self._X[idx]
        if hasattr(row, "toarray"):
            row = row.toarray()
        row = np.asarray(row, dtype=np.float32)
        if row.ndim > 1:
            row = row.reshape(-1)
        return torch.from_numpy(row)


def split_adata(
    adata: ad.AnnData,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Split AnnData into train, validation, and test sets."""
    rng = np.random.default_rng(seed)
    n = adata.n_obs
    indices = np.arange(n)
    rng.shuffle(indices)

    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    logger.info(
        f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return (
        adata[train_idx].copy(),
        adata[val_idx].copy(),
        adata[test_idx].copy(),
    )


def make_dataloaders(
    adata_train: ad.AnnData,
    adata_val: ad.AnnData,
    adata_test: ad.AnnData,
    batch_size: int = 256,
    layer: Optional[str] = None,
    dense_policy: str = "auto",
    preload: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from split AnnData objects.

    Args:
        num_workers: Number of subprocesses for data loading. Use >0 on HPC
            nodes with many CPU cores for better throughput.
        pin_memory: If True, DataLoader will copy tensors to pinned memory
            before returning them. Improves CPU->GPU transfer speed.
    """
    train_ds = SingleCellDataset(adata_train, layer=layer, dense_policy=dense_policy, preload=preload)
    val_ds = SingleCellDataset(adata_val, layer=layer, dense_policy=dense_policy, preload=preload)
    test_ds = SingleCellDataset(adata_test, layer=layer, dense_policy=dense_policy, preload=preload)

    # Persistent workers avoids re-spawning worker processes every epoch
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    logger.info(
        f"  DataLoaders: train={len(train_ds)}, val={len(val_ds)}, "
        f"test={len(test_ds)}, batch_size={batch_size}, layer={layer or 'X'}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}, preload={preload}"
    )
    return train_loader, val_loader, test_loader
