"""
Evaluation metrics for latent space quality.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

from src.models.losses import reconstruction_loss, vae_loss
from src.models.vae import VAE
from src.utils.conversion_utils import to_numpy

logger = logging.getLogger(__name__)


def _safe_silhouette(latent: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette safely, returning NaN for degenerate partitions."""
    n_unique = len(np.unique(labels))
    if n_unique <= 1 or n_unique >= len(labels):
        return float("nan")
    return float(silhouette_score(latent, labels))


def _move_batch_to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move batch to device; non-blocking only for pinned CPU -> CUDA transfers."""
    use_non_blocking = (
        device.type == "cuda"
        and getattr(batch, "device", None) is not None
        and batch.device.type == "cpu"
        and hasattr(batch, "is_pinned")
        and batch.is_pinned()
    )
    if use_non_blocking:
        return batch.to(device, non_blocking=True)
    return batch.to(device)


def _compute_batch_knn_entropy(latent: np.ndarray, batch_labels: np.ndarray, knn_k: int) -> float:
    """Compute normalized neighborhood batch entropy in [0, 1] (higher = better batch mixing)."""
    from sklearn.neighbors import NearestNeighbors

    latent = to_numpy(latent)
    batch_labels = to_numpy(batch_labels)
    n = int(latent.shape[0])
    if n < 3:
        return float("nan")

    unique_batches = np.unique(batch_labels)
    if unique_batches.size <= 1:
        return float("nan")

    k_eff = max(1, min(int(knn_k), n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1)
    nn.fit(latent)
    neigh_idx = nn.kneighbors(return_distance=False)[:, 1:]

    log_base = np.log(float(unique_batches.size))
    if not np.isfinite(log_base) or log_base <= 0:
        return float("nan")

    entropies = []
    for row in neigh_idx:
        vals = batch_labels[row]
        _, counts = np.unique(vals, return_counts=True)
        p = counts.astype(np.float64) / max(1.0, float(counts.sum()))
        e = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0))) / log_base
        entropies.append(e)

    if not entropies:
        return float("nan")
    return float(np.mean(entropies))


def _safe_calinski_harabasz(latent: np.ndarray, labels: np.ndarray) -> float:
    """Compute Calinski-Harabasz safely, returning NaN for degenerate partitions."""
    n_unique = len(np.unique(labels))
    if n_unique <= 1 or n_unique >= len(labels):
        return float("nan")
    try:
        return float(calinski_harabasz_score(latent, labels))
    except Exception:
        return float("nan")


def _safe_davies_bouldin(latent: np.ndarray, labels: np.ndarray) -> float:
    """Compute Davies-Bouldin safely, returning NaN for degenerate partitions."""
    n_unique = len(np.unique(labels))
    if n_unique <= 1 or n_unique >= len(labels):
        return float("nan")
    try:
        return float(davies_bouldin_score(latent, labels))
    except Exception:
        return float("nan")


def _compute_continuity(
    X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 5
) -> float:
    """Compute continuity: how well the embedding preserves original neighborhoods.

    Measures whether points that are neighbors in the original space remain
    neighbors in the latent space.  Complement of trustworthiness.
    Based on Venna & Kaski (2006).
    """
    from sklearn.metrics import pairwise_distances

    n = X_high.shape[0]
    k = min(n_neighbors, n - 1)
    if k < 1:
        return float("nan")

    dist_high = pairwise_distances(X_high)
    dist_low = pairwise_distances(X_low)

    rank_high = np.argsort(np.argsort(dist_high, axis=1), axis=1)
    rank_low = np.argsort(np.argsort(dist_low, axis=1), axis=1)

    nn_high = np.argsort(dist_high, axis=1)[:, 1 : k + 1]

    penalty = 0.0
    for i in range(n):
        for j in nn_high[i]:
            r = rank_low[i, j]
            if r > k:
                penalty += r - k

    norm = n * k * (2 * n - 3 * k - 1)
    if norm == 0:
        return 1.0
    return float(1.0 - 2.0 * penalty / norm)


def evaluate_latent(
    latent: np.ndarray,
    n_clusters: int,
    true_labels: Optional[np.ndarray] = None,
    n_init: int = 10,
    seed: int = 42,
    silhouette_max_cells: int = 0,
    batch_labels: Optional[np.ndarray] = None,
    batch_metrics_enabled: bool = True,
    batch_metrics_max_cells: int = 0,
    batch_metrics_knn_k: int = 15,
    original_data: Optional[np.ndarray] = None,
    neighborhood_max_cells: int = 5000,
) -> Dict[str, float]:
    """Evaluate latent space with unsupervised, external-label, and optional batch metrics.

    - Always computes KMeans labels and silhouette on KMeans partitions.
    - Computes Calinski-Harabasz and Davies-Bouldin indices on KMeans partitions.
    - Computes ARI/AMI/silhouette_true_labels only when `true_labels` are provided.
    - Computes trustworthiness/continuity when `original_data` is provided.
    - Optionally samples evaluation cells for scalability on large datasets.
    - Batch-aware metrics are optional and fail-safe (NaN + metadata on failure).
    """
    latent = to_numpy(latent)
    if latent.ndim != 2:
        raise ValueError(f"Expected latent to be 2D, got shape={latent.shape}")

    n_total = int(latent.shape[0])
    n_used = n_total
    sampled = False
    sample_indices = None

    if silhouette_max_cells > 0 and n_total > silhouette_max_cells:
        rng = np.random.default_rng(seed)
        n_used = int(silhouette_max_cells)
        sample_indices = rng.choice(n_total, size=n_used, replace=False)
        latent_eval = latent[sample_indices]
        sampled = True
        logger.info(
            "  Latent evaluation sampled for scalability: n_used=%d / n_total=%d (seed=%d)",
            n_used,
            n_total,
            seed,
        )
    else:
        latent_eval = latent

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    predicted = kmeans.fit_predict(latent_eval)

    sil_kmeans = _safe_silhouette(latent_eval, predicted)
    ch_score = _safe_calinski_harabasz(latent_eval, predicted)
    db_score = _safe_davies_bouldin(latent_eval, predicted)

    # Trustworthiness & continuity (require original high-dim data)
    trust_score = float("nan")
    cont_score = float("nan")
    if original_data is not None:
        try:
            orig = to_numpy(original_data)
            if sample_indices is not None:
                orig_eval = orig[sample_indices]
            else:
                orig_eval = orig
            # Subsample for O(n^2) neighborhood metrics
            n_neigh = min(len(orig_eval), neighborhood_max_cells)
            if n_neigh < len(orig_eval):
                rng_neigh = np.random.default_rng(seed + 7)
                idx_neigh = rng_neigh.choice(len(orig_eval), size=n_neigh, replace=False)
                orig_sub = orig_eval[idx_neigh]
                lat_sub = latent_eval[idx_neigh]
            else:
                orig_sub = orig_eval
                lat_sub = latent_eval
            n_k = min(5, n_neigh - 1)
            if n_k >= 1:
                trust_score = float(sklearn_trustworthiness(orig_sub, lat_sub, n_neighbors=n_k))
                cont_score = _compute_continuity(orig_sub, lat_sub, n_neighbors=n_k)
        except Exception as exc:
            logger.warning("Trustworthiness/continuity computation failed: %s", exc)

    if true_labels is not None:
        true_labels = to_numpy(true_labels)
        if sample_indices is not None:
            true_eval = true_labels[sample_indices]
        else:
            true_eval = true_labels
        ari = float(adjusted_rand_score(true_eval, predicted))
        ami = float(adjusted_mutual_info_score(true_eval, predicted))
        sil_true = _safe_silhouette(latent_eval, true_eval)
    else:
        ari = float("nan")
        ami = float("nan")
        sil_true = float("nan")

    # Batch metrics defaults
    batch_sil = float("nan")
    batch_entropy = float("nan")
    batch_mode = "not_available"
    batch_n_total = n_total
    batch_n_used = 0

    if not batch_metrics_enabled:
        batch_mode = "disabled"
    elif batch_labels is not None:
        try:
            batch_all = to_numpy(batch_labels)
            if sample_indices is not None:
                batch_eval = batch_all[sample_indices]
                latent_batch = latent_eval
                batch_mode = "sampled_shared"
            else:
                batch_eval = batch_all
                latent_batch = latent
                batch_mode = "full"

            if batch_metrics_max_cells > 0 and len(batch_eval) > batch_metrics_max_cells:
                rng = np.random.default_rng(seed + 17)
                idx = rng.choice(len(batch_eval), size=int(batch_metrics_max_cells), replace=False)
                batch_eval = batch_eval[idx]
                latent_batch = latent_batch[idx]
                batch_mode = "sampled"

            batch_n_used = int(len(batch_eval))

            unique_batches = np.unique(batch_eval)
            if unique_batches.size <= 1:
                batch_mode = "single_batch"
            elif unique_batches.size >= batch_n_used:
                batch_mode = "degenerate_partition"
            else:
                batch_sil = _safe_silhouette(latent_batch, batch_eval)
                batch_entropy = _compute_batch_knn_entropy(
                    latent_batch,
                    batch_eval,
                    knn_k=batch_metrics_knn_k,
                )
        except Exception as exc:
            batch_mode = "failed"
            logger.warning("Batch metric computation failed: %s", exc)

    # Note on directionality:
    # - batch_silhouette: higher usually means stronger batch separation (worse mixing)
    # - batch_silhouette_mixing: oriented so higher is better mixing (=-batch_silhouette)
    batch_sil_mixing = -batch_sil if np.isfinite(batch_sil) else float("nan")

    metrics = {
        "ari": ari,
        "ami": ami,
        # Keep legacy column name mapped to unsupervised clustering silhouette.
        "silhouette": sil_kmeans,
        "silhouette_kmeans": sil_kmeans,
        "silhouette_true_labels": sil_true,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_score,
        "trustworthiness": trust_score,
        "continuity": cont_score,
        "silhouette_sampled": sampled,
        "silhouette_n_used": n_used,
        "silhouette_n_total": n_total,
        "batch_silhouette": batch_sil,
        "batch_silhouette_mixing": batch_sil_mixing,
        "batch_knn_entropy": batch_entropy,
        "batch_metrics_mode": batch_mode,
        "batch_metrics_n_used": int(batch_n_used),
        "batch_metrics_n_total": int(batch_n_total),
    }

    def _fmt(v: float) -> str:
        return f"{v:.4f}" if np.isfinite(v) else "nan"

    logger.info(
        "  Metrics: ARI=%s, AMI=%s, Sil(km)=%s, CH=%s, DB=%s, Trust=%s, Cont=%s [n=%d/%d]",
        _fmt(ari), _fmt(ami), _fmt(sil_kmeans),
        _fmt(ch_score), _fmt(db_score), _fmt(trust_score), _fmt(cont_score),
        n_used, n_total,
    )
    return metrics


def extract_original_data(dataloader: DataLoader) -> np.ndarray:
    """Extract the original input data from a dataloader (for neighborhood metrics)."""
    data_list = []
    for batch in dataloader:
        data_list.append(to_numpy(batch))
    return np.concatenate(data_list, axis=0)


@torch.no_grad()
def compute_reconstruction_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_type: str = "mse",
    normalize_by_genes: bool = True,
) -> float:
    """Compute average reconstruction loss on a dataset for selected loss type.

    For NB/ZINB, the training objective sums over genes and averages over cells.
    Setting `normalize_by_genes=True` reports a per-gene comparable NLL scale.
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    is_vae = isinstance(model, VAE)

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)

        if is_vae:
            x_out, _, _ = model(batch, deterministic=True)
        else:
            x_out = model(batch)

        loss = reconstruction_loss(batch, x_out, loss_type=loss_type)
        if normalize_by_genes and loss_type in {"nb", "zinb"}:
            loss = loss / max(1, batch.size(1))

        total_loss += loss.item() * batch.size(0)
        n_samples += batch.size(0)

    avg_loss = total_loss / max(1, n_samples)
    logger.info(f"  Reconstruction loss ({loss_type}): {avg_loss:.6f}")
    return avg_loss


@torch.no_grad()
def compute_vae_elbo(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float = 1.0,
    loss_type: str = "mse",
    normalize_by_genes: bool = False,
) -> float:
    """Compute average ELBO for a VAE model on a dataset.

    Returns ELBO (higher is better), i.e. the negative of the optimization
    objective `reconstruction + beta * KL` used during training.
    """
    if not isinstance(model, VAE):
        raise TypeError("compute_vae_elbo expects a VAE model instance")

    model.eval()
    total_elbo = 0.0
    n_samples = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        x_out, mu, logvar = model(batch, deterministic=True)
        total_loss, _, _ = vae_loss(
            batch,
            x_out,
            mu,
            logvar,
            beta=beta,
            loss_type=loss_type,
        )

        if normalize_by_genes and loss_type in {"nb", "zinb"}:
            total_loss = total_loss / max(1, batch.size(1))

        total_elbo += (-total_loss.item()) * batch.size(0)
        n_samples += batch.size(0)

    avg_elbo = total_elbo / max(1, n_samples)
    logger.info(f"  VAE ELBO ({loss_type}): {avg_elbo:.6f}")
    return avg_elbo


@torch.no_grad()
def extract_latent(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """Extract latent representations for all data."""
    model.eval()
    latent_list = []

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        z = model.get_latent(batch)
        latent_list.append(to_numpy(z))

    return np.concatenate(latent_list, axis=0)


def _pack_centrality_return(
    variance: float,
    mode: str,
    n_used: int,
    n_total: int,
    return_metadata: bool,
) -> Union[float, Tuple[float, Dict[str, Union[str, int]]]]:
    metadata = {
        "centrality_mode": mode,
        "centrality_n_used": int(n_used),
        "centrality_n_total": int(n_total),
    }
    if return_metadata:
        return variance, metadata
    return variance


def compute_centrality_variance(
    latent: np.ndarray,
    n_neighbors: int = 15,
    policy: str = "full",
    threshold_cells: int = 0,
    sample_size: int = 0,
    seed: int = 42,
    return_metadata: bool = False,
) -> Union[float, Tuple[float, Dict[str, Union[str, int]]]]:
    """Compute variance of eigenvector centrality on the kNN graph.

    Safety behavior:
    - Returns NaN for tiny/invalid latent matrices.
    - Clamps neighbors to a valid range to avoid Scanpy runtime errors.
    - Supports policy-driven scale guardrails for large datasets.
    """
    if policy not in {"full", "sample", "skip"}:
        raise ValueError("centrality policy must be one of: full, sample, skip")

    try:
        import networkx as nx
        import scanpy as sc
        import anndata as ad
    except ImportError:
        logger.warning("networkx or scanpy not available; centrality set to NaN.")
        return _pack_centrality_return(
            float("nan"),
            "missing_dependencies",
            0,
            0,
            return_metadata,
        )

    latent = to_numpy(latent)
    if latent.ndim != 2:
        logger.warning("Centrality expects 2D latent matrix, got shape=%s; returning NaN.", latent.shape)
        return _pack_centrality_return(
            float("nan"),
            "invalid_shape",
            0,
            int(latent.shape[0]) if latent.ndim > 0 else 0,
            return_metadata,
        )

    n_total = int(latent.shape[0])
    if n_total < 3:
        logger.warning(
            "Centrality skipped: too few cells for stable kNN graph (n_obs=%d). Returning NaN.",
            n_total,
        )
        return _pack_centrality_return(
            float("nan"),
            "too_small",
            n_total,
            n_total,
            return_metadata,
        )

    latent_eval = latent
    mode = "full"

    if threshold_cells > 0 and n_total > threshold_cells:
        if policy == "skip":
            logger.info(
                "  Centrality skipped by policy: n_obs=%d exceeds threshold=%d.",
                n_total,
                threshold_cells,
            )
            return _pack_centrality_return(
                float("nan"),
                "skipped_threshold",
                0,
                n_total,
                return_metadata,
            )

        if policy == "sample":
            target = sample_size if sample_size > 0 else threshold_cells
            n_used = max(3, min(int(target), n_total))
            rng = np.random.default_rng(seed)
            indices = rng.choice(n_total, size=n_used, replace=False)
            latent_eval = latent[indices]
            mode = "sampled"
            logger.info(
                "  Centrality sampled for scalability: n_used=%d / n_total=%d (seed=%d)",
                n_used,
                n_total,
                seed,
            )

    n_used = int(latent_eval.shape[0])
    safe_neighbors = max(1, min(int(n_neighbors), n_used - 1))

    try:
        adata_tmp = ad.AnnData(X=latent_eval)
        sc.pp.neighbors(adata_tmp, n_neighbors=safe_neighbors, use_rep="X")
        adj_matrix = adata_tmp.obsp["connectivities"]
        G = nx.from_scipy_sparse_array(adj_matrix)

        centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
        variance = float(np.var(list(centrality.values())))
    except Exception as exc:
        logger.warning(
            "Centrality computation failed (mode=%s, n_obs=%d, n_neighbors=%d): %s",
            mode,
            n_used,
            safe_neighbors,
            exc,
        )
        return _pack_centrality_return(
            float("nan"),
            "failed",
            n_used,
            n_total,
            return_metadata,
        )

    logger.info(
        "  Centrality variance: %.6f [mode=%s, n_used=%d/%d]",
        variance,
        mode,
        n_used,
        n_total,
    )
    return _pack_centrality_return(variance, mode, n_used, n_total, return_metadata)
