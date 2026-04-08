"""
scVI model wrapper for scRNA-seq autoencoder experiments.

Wraps scvi.model.SCVI with a consistent experiment interface for
latent extraction, reconstruction metrics, training history, and TensorBoard.
"""

import os
import logging
import time
from typing import Optional, Dict, Any

import numpy as np
import anndata as ad

from src.utils.conversion_utils import to_float, to_numpy

logger = logging.getLogger(__name__)


class ScVIWrapper:
    """Wrapper around scvi.model.SCVI for consistent experiment interface.

    Follows the scvi-tools v1.3.x API:
        - setup_anndata with layer="counts" for raw count input
        - gene_likelihood: "nb" (default) or "zinb"
        - Optional batch_key for batch-aware modeling
        - TensorBoard via PyTorch Lightning's TensorBoardLogger
    """

    def __init__(
        self,
        latent_dim: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        gene_likelihood: str = "nb",
        max_epochs: int = 100,
        learning_rate: float = 0.001,
        seed: int = 42,
        batch_key: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        run_tag: Optional[str] = None,
        log_every_n_epochs: int = 1,
    ):
        self.latent_dim = latent_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.gene_likelihood = gene_likelihood
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.batch_key = batch_key
        self.tensorboard_dir = tensorboard_dir
        self.run_tag = run_tag
        self.log_every_n_epochs = max(1, int(log_every_n_epochs))
        self.model = None
        self._is_trained = False
        self._training_time = 0.0
        self._actual_epochs = 0

    @staticmethod
    def _metric_to_float(value) -> Optional[float]:
        """Best-effort conversion of trainer callback metrics to float."""
        try:
            return to_float(value)
        except Exception:
            return None

    def setup_and_train(self, adata: ad.AnnData) -> Dict[str, Any]:
        """Setup scVI model, train it, and return training info.

        Returns:
            Dict with keys: history, actual_epochs, training_time_seconds
        """
        try:
            import scvi
        except ImportError as exc:
            raise ImportError(
                "scvi-tools is required for the scVI model. "
                "Install with: pip install scvi-tools"
            ) from exc

        scvi.settings.seed = self.seed

        adata_scvi = adata.copy()

        # Verify counts layer exists (required for scVI)
        if "counts" not in adata_scvi.layers:
            raise ValueError(
                "scVI requires raw count data in adata.layers['counts']. "
                "Ensure preprocessing stores counts before normalization."
            )

        if self.batch_key:
            if self.batch_key not in adata_scvi.obs.columns:
                raise ValueError(
                    f"batch_key='{self.batch_key}' not found in adata.obs. "
                    f"Available: {list(adata_scvi.obs.columns)}"
                )
            scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key=self.batch_key)
        else:
            scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts")

        logger.info(
            f"  scVI setup: d={self.latent_dim}, likelihood={self.gene_likelihood}, "
            f"hidden={self.n_hidden}, layers={self.n_layers}, batch_key={self.batch_key}"
        )

        self.model = scvi.model.SCVI(
            adata_scvi,
            n_latent=self.latent_dim,
            gene_likelihood=self.gene_likelihood,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
        )

        # Configure trainer kwargs passed into scvi-tools / Lightning.
        train_kwargs = {
            "max_epochs": self.max_epochs,
            "early_stopping": True,
            "plan_kwargs": {"lr": self.learning_rate},
            # Progress bars render with carriage returns and are hard to follow in `tail -f`.
            "enable_progress_bar": False,
        }

        # Add line-based epoch progress logs for tail-friendly monitoring.
        callback_base = None
        try:
            from lightning.pytorch.callbacks import Callback as callback_base
        except ImportError:
            try:
                from pytorch_lightning.callbacks import Callback as callback_base
            except ImportError:
                callback_base = None

        if callback_base is not None:
            wrapper = self

            class EpochProgressLogger(callback_base):
                def on_train_epoch_end(self, trainer, pl_module):
                    epoch = int(getattr(trainer, "current_epoch", 0)) + 1
                    max_ep = int(getattr(trainer, "max_epochs", 0) or 0)
                    if (
                        epoch % wrapper.log_every_n_epochs != 0
                        and epoch != 1
                        and (max_ep == 0 or epoch != max_ep)
                    ):
                        return

                    metrics = getattr(trainer, "callback_metrics", {}) or {}

                    train_val = None
                    for key in ("train_loss_epoch", "train_loss", "elbo_train"):
                        if key in metrics:
                            train_val = wrapper._metric_to_float(metrics[key])
                            if train_val is not None:
                                break

                    val_val = None
                    for key in (
                        "validation_loss",
                        "val_loss",
                        "elbo_validation",
                        "reconstruction_loss_validation",
                    ):
                        if key in metrics:
                            val_val = wrapper._metric_to_float(metrics[key])
                            if val_val is not None:
                                break

                    if train_val is not None and val_val is not None:
                        logger.info(
                            "    scVI Epoch %3d/%s: train=%.6f, val=%.6f",
                            epoch,
                            str(max_ep) if max_ep else "?",
                            train_val,
                            val_val,
                        )
                    elif train_val is not None:
                        logger.info(
                            "    scVI Epoch %3d/%s: train=%.6f",
                            epoch,
                            str(max_ep) if max_ep else "?",
                            train_val,
                        )
                    else:
                        logger.info(
                            "    scVI Epoch %3d/%s",
                            epoch,
                            str(max_ep) if max_ep else "?",
                        )

            train_kwargs["callbacks"] = train_kwargs.get("callbacks", []) + [EpochProgressLogger()]

        if self.tensorboard_dir:
            try:
                from pytorch_lightning.loggers import TensorBoardLogger
                # Use full path as save_dir with empty name/version to avoid
                # Lightning creating extra subdirectories (name/version_N/)
                tb_log_path = os.path.join(self.tensorboard_dir, self.run_tag or "scvi")
                os.makedirs(tb_log_path, exist_ok=True)
                tb_logger = TensorBoardLogger(
                    save_dir=tb_log_path,
                    name="",
                    version="",
                )
                train_kwargs["logger"] = tb_logger
                logger.info(f"  scVI TensorBoard logging to: {tb_log_path}")
            except ImportError:
                try:
                    from lightning.pytorch.loggers import TensorBoardLogger
                    tb_log_path = os.path.join(self.tensorboard_dir, self.run_tag or "scvi")
                    os.makedirs(tb_log_path, exist_ok=True)
                    tb_logger = TensorBoardLogger(
                        save_dir=tb_log_path,
                        name="",
                        version="",
                    )
                    train_kwargs["logger"] = tb_logger
                    logger.info(f"  scVI TensorBoard logging to: {tb_log_path}")
                except ImportError:
                    logger.warning(
                        "  pytorch_lightning/lightning not found; "
                        "scVI TensorBoard logging disabled."
                    )

        start_time = time.time()
        self.model.train(**train_kwargs)
        self._training_time = time.time() - start_time

        self._is_trained = True

        # Extract actual epoch count from training history
        history = self.model.history
        if "train_loss_epoch" in history:
            self._actual_epochs = len(history["train_loss_epoch"])
        elif "elbo_train" in history:
            self._actual_epochs = len(history["elbo_train"])
        else:
            # Fallback: try to count any history key
            for key in history:
                if hasattr(history[key], "__len__"):
                    self._actual_epochs = len(history[key])
                    break
            else:
                self._actual_epochs = self.max_epochs

        logger.info(
            f"  scVI training complete: {self._actual_epochs} epochs "
            f"in {self._training_time:.1f}s"
        )

        return {
            "history": history,
            "actual_epochs": self._actual_epochs,
            "training_time_seconds": self._training_time,
        }

    def _to_float(self, value) -> float:
        """Convert scvi-tools metric outputs to float via shared conversion utility."""
        return to_float(value)

    def get_latent(self, adata: ad.AnnData) -> np.ndarray:
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model must be trained before extracting latent.")
        return to_numpy(self.model.get_latent_representation(adata))

    def get_reconstruction_loss(self, adata: ad.AnnData) -> float:
        """Return reconstruction error (negative log-likelihood scale)."""
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model must be trained before evaluation.")

        try:
            recon = self.model.get_reconstruction_error(adata)
            return self._to_float(recon)
        except Exception as exc:
            logger.warning(
                "scVI get_reconstruction_error failed (%s); falling back to -ELBO.",
                exc,
            )
            elbo = self.model.get_elbo(adata)
            return -self._to_float(elbo)

    def get_elbo(self, adata: ad.AnnData) -> float:
        """Return ELBO as reported by scvi-tools."""
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model must be trained before evaluation.")
        return self._to_float(self.model.get_elbo(adata))

    @property
    def actual_epochs(self) -> int:
        return self._actual_epochs

    @property
    def training_time(self) -> float:
        return self._training_time
