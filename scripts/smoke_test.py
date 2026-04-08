#!/usr/bin/env python3
"""
Smoke test: verify the full pipeline on a tiny synthetic dataset.

This tests AE, VAE, and (optionally) scVI on ~200 synthetic cells
with known cluster labels. Runs in <2 minutes on CPU.

Usage:
    python scripts/smoke_test.py              # AE + VAE only (no scvi-tools needed)
    python scripts/smoke_test.py --with_scvi  # AE + VAE + scVI
"""

import os
import sys
import argparse
import logging
import json
import numpy as np

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import anndata as ad
from scipy.sparse import csr_matrix

from src.models.autoencoder import Autoencoder
from src.models.vae import VAE
from src.models.losses import reconstruction_loss, vae_loss
from src.data.dataset import SingleCellDataset, split_adata, make_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_latent, compute_centrality_variance
from src.utils.conversion_utils import to_numpy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_synthetic_adata(n_cells=200, n_genes=500, n_clusters=4, seed=42):
    """Create a tiny synthetic dataset with clear cluster structure."""
    rng = np.random.default_rng(seed)

    # Generate count-like data with cluster structure
    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    labels = []
    cells_per_cluster = n_cells // n_clusters

    for k in range(n_clusters):
        start = k * cells_per_cluster
        end = start + cells_per_cluster
        # Each cluster has elevated expression in a different gene subset
        base_rate = rng.poisson(lam=2.0, size=(cells_per_cluster, n_genes)).astype(np.float32)
        marker_start = k * (n_genes // n_clusters)
        marker_end = marker_start + (n_genes // n_clusters)
        base_rate[:, marker_start:marker_end] += rng.poisson(
            lam=10.0, size=(cells_per_cluster, marker_end - marker_start)
        ).astype(np.float32)
        counts[start:end] = base_rate
        labels.extend([f"cluster_{k}"] * cells_per_cluster)

    # Normalized version (log1p)
    lib_sizes = counts.sum(axis=1, keepdims=True)
    lib_sizes = np.maximum(lib_sizes, 1.0)
    normalized = np.log1p((counts / lib_sizes) * 10000)

    adata = ad.AnnData(
        X=normalized,
        layers={"counts": csr_matrix(counts)},
    )
    adata.obs["cell_type"] = labels
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    logger.info(f"  Synthetic data: {n_cells} cells, {n_genes} genes, {n_clusters} clusters")
    return adata


def test_ae(adata, device, tb_dir=None, ckpt_dir="checkpoints/smoke_test"):
    """Test standard Autoencoder."""
    logger.info("=" * 60)
    logger.info("TEST: Autoencoder (AE) with MSE loss")
    logger.info("=" * 60)

    latent_dim = 8
    K = adata.obs["cell_type"].nunique()

    adata_train, adata_val, adata_test = split_adata(adata, seed=42)
    train_loader, val_loader, test_loader = make_dataloaders(
        adata_train, adata_val, adata_test,
        batch_size=64, layer=None,
    )

    model = Autoencoder(
        input_dim=adata.n_vars,
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        activation="relu",
        loss_type="mse",
    )

    logger.info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Architecture:\n{model}")

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        early_stopping_patience=5,
        checkpoint_dir=ckpt_dir,
        loss_type="mse",
        profile_memory=True,
        max_grad_norm=5.0,
        tensorboard_dir=tb_dir,
        run_tag="smoke_ae_mse_d8",
    )

    history = trainer.train(
        train_loader, val_loader,
        max_epochs=20,
        checkpoint_name="smoke_ae",
    )

    # Extract latent and evaluate
    full_loader = make_dataloaders(adata, adata[:0], adata[:0], batch_size=64)[0]
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in full_loader:
            z = model.get_latent(batch.to(device))
            latents.append(to_numpy(z))
    latent = np.concatenate(latents, axis=0)

    true_labels = adata.obs["cell_type"].values
    metrics = evaluate_latent(latent, n_clusters=K, true_labels=true_labels, seed=42)
    centrality_var = compute_centrality_variance(latent)

    trainer.close()

    logger.info(f"  ✅ AE Results:")
    logger.info(f"     Best val loss: {history['best_val_loss']:.6f} (epoch {history['best_epoch']})")
    logger.info(f"     Total epochs:  {history['total_epochs']}")
    logger.info(f"     ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
    logger.info(f"     Silhouette: {metrics['silhouette']:.4f}")
    logger.info(f"     Centrality variance: {centrality_var:.4f}")
    logger.info(f"     Peak CPU mem: {history['peak_cpu_mem_mb']:.1f} MB")
    logger.info(f"     Latent shape: {latent.shape}")

    return {"status": "PASS", "ari": metrics["ari"], "val_loss": history["best_val_loss"]}


def test_vae(adata, device, tb_dir=None, ckpt_dir="checkpoints/smoke_test"):
    """Test Variational Autoencoder."""
    logger.info("=" * 60)
    logger.info("TEST: Variational Autoencoder (VAE) with MSE loss")
    logger.info("=" * 60)

    latent_dim = 8
    K = adata.obs["cell_type"].nunique()

    adata_train, adata_val, adata_test = split_adata(adata, seed=42)
    train_loader, val_loader, test_loader = make_dataloaders(
        adata_train, adata_val, adata_test,
        batch_size=64, layer=None,
    )

    model = VAE(
        input_dim=adata.n_vars,
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        activation="relu",
        loss_type="mse",
    )

    logger.info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        early_stopping_patience=5,
        vae_beta=1.0,
        checkpoint_dir=ckpt_dir,
        loss_type="mse",
        profile_memory=True,
        max_grad_norm=5.0,
        tensorboard_dir=tb_dir,
        run_tag="smoke_vae_mse_d8",
    )

    history = trainer.train(
        train_loader, val_loader,
        max_epochs=20,
        checkpoint_name="smoke_vae",
    )

    # Extract latent
    full_loader = make_dataloaders(adata, adata[:0], adata[:0], batch_size=64)[0]
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in full_loader:
            z = model.get_latent(batch.to(device))
            latents.append(to_numpy(z))
    latent = np.concatenate(latents, axis=0)

    true_labels = adata.obs["cell_type"].values
    metrics = evaluate_latent(latent, n_clusters=K, true_labels=true_labels, seed=42)

    trainer.close()

    logger.info(f"  ✅ VAE Results:")
    logger.info(f"     Best val loss: {history['best_val_loss']:.6f} (epoch {history['best_epoch']})")
    logger.info(f"     Total epochs:  {history['total_epochs']}")
    logger.info(f"     ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
    logger.info(f"     Silhouette: {metrics['silhouette']:.4f}")
    logger.info(f"     Peak CPU mem: {history['peak_cpu_mem_mb']:.1f} MB")

    return {"status": "PASS", "ari": metrics["ari"], "val_loss": history["best_val_loss"]}


def test_nb_loss(adata, device, ckpt_dir="checkpoints/smoke_test"):
    """Test AE with NB loss on count data."""
    logger.info("=" * 60)
    logger.info("TEST: AE with Negative Binomial (NB) loss")
    logger.info("=" * 60)

    latent_dim = 8
    adata_train, adata_val, adata_test = split_adata(adata, seed=42)
    train_loader, val_loader, _ = make_dataloaders(
        adata_train, adata_val, adata_test,
        batch_size=64, layer="counts",  # raw counts for NB
    )

    model = Autoencoder(
        input_dim=adata.n_vars,
        latent_dim=latent_dim,
        hidden_dims=[64, 32],
        activation="relu",
        loss_type="nb",
    )

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        early_stopping_patience=5,
        checkpoint_dir=ckpt_dir,
        loss_type="nb",
        profile_memory=False,
        max_grad_norm=5.0,
    )

    history = trainer.train(
        train_loader, val_loader,
        max_epochs=10,
        checkpoint_name="smoke_ae_nb",
    )
    trainer.close()

    logger.info(f"  ✅ NB Loss Results:")
    logger.info(f"     Best val loss: {history['best_val_loss']:.6f}")
    logger.info(f"     Gradients stable: {'yes' if not np.isnan(history['best_val_loss']) else 'NO — NaN detected!'}")

    return {"status": "PASS" if not np.isnan(history["best_val_loss"]) else "FAIL"}


def test_scvi(adata):
    """Test scVI wrapper."""
    logger.info("=" * 60)
    logger.info("TEST: scVI (via scvi-tools)")
    logger.info("=" * 60)

    try:
        from src.models.scvi_wrapper import ScVIWrapper

        wrapper = ScVIWrapper(
            latent_dim=8,
            n_hidden=64,
            n_layers=1,
            gene_likelihood="nb",
            max_epochs=10,
            learning_rate=0.001,
            seed=42,
        )

        train_info = wrapper.setup_and_train(adata)
        latent = wrapper.get_latent(adata)
        recon_loss = wrapper.get_reconstruction_loss(adata)

        K = adata.obs["cell_type"].nunique()
        true_labels = adata.obs["cell_type"].values
        metrics = evaluate_latent(latent, n_clusters=K, true_labels=true_labels, seed=42)

        logger.info(f"  ✅ scVI Results:")
        logger.info(f"     Actual epochs: {wrapper.actual_epochs}")
        logger.info(f"     Training time: {wrapper.training_time:.1f}s")
        logger.info(f"     Recon loss (neg ELBO): {recon_loss:.4f}")
        logger.info(f"     ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
        logger.info(f"     Latent shape: {latent.shape}")

        return {"status": "PASS", "ari": metrics["ari"]}

    except ImportError as e:
        logger.warning(f"  ⚠️ scVI SKIPPED: {e}")
        return {"status": "SKIPPED", "reason": str(e)}
    except Exception as e:
        logger.error(f"  ❌ scVI FAILED: {e}", exc_info=True)
        return {"status": "FAIL", "error": str(e)}


def test_symmetric_architecture():
    """Verify encoder-decoder symmetry."""
    logger.info("=" * 60)
    logger.info("TEST: Architecture symmetry check")
    logger.info("=" * 60)

    for hidden_dims in [[128, 64], [256, 128, 64], [64]]:
        ae = Autoencoder(input_dim=500, latent_dim=10, hidden_dims=hidden_dims, loss_type="mse")
        vae = VAE(input_dim=500, latent_dim=10, hidden_dims=hidden_dims, loss_type="mse")

        # Count layers in encoder and decoder
        enc_linears = sum(1 for m in ae.encoder.modules() if isinstance(m, torch.nn.Linear))
        dec_linears = sum(1 for m in ae.decoder_hidden.modules() if isinstance(m, torch.nn.Linear))
        # decoder_hidden has len(hidden_dims) linears, plus recon_head adds 1 more = enc_linears total

        logger.info(
            f"  hidden_dims={hidden_dims}: "
            f"encoder_linears={enc_linears}, "
            f"decoder_trunk_linears={dec_linears}, "
            f"+ recon_head=1 → "
            f"total_decoder={dec_linears + 1}"
        )

        assert enc_linears == dec_linears + 1, (
            f"Asymmetry! encoder has {enc_linears} linear layers, "
            f"decoder trunk has {dec_linears} + 1 (head) = {dec_linears + 1}"
        )

    logger.info("  ✅ All architectures are symmetric")
    return {"status": "PASS"}


def main():
    parser = argparse.ArgumentParser(description="Smoke test for the pipeline")
    parser.add_argument("--with_scvi", action="store_true", help="Also test scVI")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Use project-local dirs (not /tmp which is node-local and ephemeral)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tb_dir = os.path.join(project_root, "logs", "tensorboard", "smoke_test") if args.tensorboard else None
    ckpt_dir = os.path.join(project_root, "checkpoints", "smoke_test")

    # Create synthetic data
    adata = make_synthetic_adata(n_cells=200, n_genes=500, n_clusters=4, seed=42)

    results = {}

    # 1. Architecture symmetry
    results["symmetry"] = test_symmetric_architecture()

    # 2. AE with MSE
    results["ae_mse"] = test_ae(adata, device, tb_dir=tb_dir, ckpt_dir=ckpt_dir)

    # 3. VAE with MSE
    results["vae_mse"] = test_vae(adata, device, tb_dir=tb_dir, ckpt_dir=ckpt_dir)

    # 4. AE with NB loss
    results["ae_nb"] = test_nb_loss(adata, device, ckpt_dir=ckpt_dir)

    # 5. scVI (optional)
    if args.with_scvi:
        results["scvi"] = test_scvi(adata)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)
    all_pass = True
    for name, result in results.items():
        status = result["status"]
        icon = "✅" if status == "PASS" else ("⚠️" if status == "SKIPPED" else "❌")
        logger.info(f"  {icon} {name}: {status}")
        if status == "FAIL":
            all_pass = False

    if tb_dir:
        logger.info(f"\n  TensorBoard logs at: {tb_dir}")
        logger.info(f"  View with: tensorboard --logdir {tb_dir}")
        # List TB event files
        for root, dirs, files in os.walk(tb_dir):
            for f in files:
                if "events.out.tfevents" in f:
                    fpath = os.path.join(root, f)
                    size_kb = os.path.getsize(fpath) / 1024
                    logger.info(f"  📊 TB event: {fpath} ({size_kb:.1f} KB)")

    if all_pass:
        logger.info("\n🎉 All tests passed! Pipeline is ready.")
    else:
        logger.info("\n💥 Some tests failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
