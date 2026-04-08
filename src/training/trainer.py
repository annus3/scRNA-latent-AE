"""
Training loop for AE and VAE models.

Supports early stopping, loss matrix (MSE/NB/ZINB), logging, checkpointing,
memory profiling, gradient clipping, and TensorBoard monitoring.
"""

import os
import time
import logging
import tracemalloc
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.losses import reconstruction_loss, vae_loss, kl_divergence
from src.models.vae import VAE

logger = logging.getLogger(__name__)


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


class Trainer:
    """Training manager for AE and VAE models.

    TensorBoard Usage:
        Pass `tensorboard_dir` to enable logging. Then view via:
            tensorboard --logdir=<tensorboard_dir>
        On HPC, use SSH port forwarding:
            ssh -L 6006:localhost:6006 <user>@<cluster>
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        early_stopping_patience: int = 10,
        vae_beta: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        loss_type: str = "mse",
        profile_memory: bool = True,
        max_grad_norm: float = 5.0,
        tensorboard_dir: Optional[str] = None,
        run_tag: Optional[str] = None,
        log_every_n_epochs: int = 1,
    ):
        if loss_type not in {"mse", "nb", "zinb"}:
            raise ValueError("loss_type must be one of: mse, nb, zinb")

        self.model = model.to(device)
        self.device = device
        self.vae_beta = vae_beta
        self.patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.loss_type = loss_type
        self.profile_memory = profile_memory
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.log_every_n_epochs = max(1, int(log_every_n_epochs))

        self.is_vae = isinstance(model, VAE)
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        os.makedirs(checkpoint_dir, exist_ok=True)

        # TensorBoard setup
        self.writer = None
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_subdir = run_tag or "default"
                tb_path = os.path.join(tensorboard_dir, log_subdir)
                os.makedirs(tb_path, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tb_path)
                logger.info(f"  TensorBoard logging to: {tb_path}")
            except ImportError:
                logger.warning(
                    "  tensorboard not installed; skipping TensorBoard logging. "
                    "Install with: pip install tensorboard"
                )

    def _compute_grad_norm(self) -> float:
        """Compute total L2 gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch. Returns dict with loss components."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_samples = 0
        grad_norms = []

        for batch in dataloader:
            batch = _move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad()

            if self.is_vae:
                x_out, mu, logvar = self.model(batch)
                loss, recon, kl = vae_loss(
                    batch, x_out, mu, logvar,
                    beta=self.vae_beta,
                    loss_type=self.loss_type,
                )
                total_recon += recon.item() * batch.size(0)
                total_kl += kl.item() * batch.size(0)
            else:
                x_out = self.model(batch)
                loss = reconstruction_loss(batch, x_out, loss_type=self.loss_type)

            loss.backward()

            # Track gradient norm before clipping
            if self.writer:
                grad_norms.append(self._compute_grad_norm())

            # Gradient clipping for NB/ZINB stability
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item() * batch.size(0)
            n_samples += batch.size(0)

        n = max(1, n_samples)
        metrics = {
            "loss": total_loss / n,
        }
        if self.is_vae:
            metrics["recon_loss"] = total_recon / n
            metrics["kl_loss"] = total_kl / n
        if grad_norms:
            metrics["grad_norm_mean"] = sum(grad_norms) / len(grad_norms)
            metrics["grad_norm_max"] = max(grad_norms)

        return metrics

    @torch.no_grad()
    def _eval_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate one epoch. Returns dict with loss components."""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_samples = 0

        for batch in dataloader:
            batch = _move_batch_to_device(batch, self.device)

            if self.is_vae:
                x_out, mu, logvar = self.model(batch, deterministic=True)
                loss, recon, kl = vae_loss(
                    batch, x_out, mu, logvar,
                    beta=self.vae_beta,
                    loss_type=self.loss_type,
                )
                total_recon += recon.item() * batch.size(0)
                total_kl += kl.item() * batch.size(0)
            else:
                x_out = self.model(batch)
                loss = reconstruction_loss(batch, x_out, loss_type=self.loss_type)

            total_loss += loss.item() * batch.size(0)
            n_samples += batch.size(0)

        n = max(1, n_samples)
        metrics = {
            "loss": total_loss / n,
        }
        if self.is_vae:
            metrics["recon_loss"] = total_recon / n
            metrics["kl_loss"] = total_kl / n

        return metrics

    def _log_to_tensorboard(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Write all tracked metrics to TensorBoard.

        Uses only add_scalar() — NOT add_scalars() which creates subdirectories
        that fragment data across multiple event files and confuse TB's UI.
        """
        if not self.writer:
            return

        # Total loss (train and val as separate scalars, same tag prefix for grouping)
        self.writer.add_scalar("Loss/train_total", train_metrics["loss"], epoch)
        self.writer.add_scalar("Loss/val_total", val_metrics["loss"], epoch)

        # VAE-specific decomposed losses
        if self.is_vae:
            self.writer.add_scalar("Loss/train_recon", train_metrics["recon_loss"], epoch)
            self.writer.add_scalar("Loss/val_recon", val_metrics["recon_loss"], epoch)
            self.writer.add_scalar("Loss/train_kl", train_metrics["kl_loss"], epoch)
            self.writer.add_scalar("Loss/val_kl", val_metrics["kl_loss"], epoch)

        # Gradient norms
        if "grad_norm_mean" in train_metrics:
            self.writer.add_scalar("Gradients/norm_mean", train_metrics["grad_norm_mean"], epoch)
            self.writer.add_scalar("Gradients/norm_max", train_metrics["grad_norm_max"], epoch)

        # Learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("Hyperparams/learning_rate", current_lr, epoch)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        checkpoint_name: str = "best_model",
    ) -> Dict[str, float]:
        model_type = "VAE" if self.is_vae else "AE"
        latent_dim = self.model.latent_dim
        logger.info(
            f"  Training {model_type} (d={latent_dim}, loss={self.loss_type}) "
            f"on {self.device} for up to {max_epochs} epochs"
        )

        # Log model architecture summary to TensorBoard
        if self.writer:
            n_params = sum(p.numel() for p in self.model.parameters())
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.writer.add_text("Model/summary", (
                f"**Type**: {model_type}  \n"
                f"**Latent dim**: {latent_dim}  \n"
                f"**Loss**: {self.loss_type}  \n"
                f"**Total params**: {n_params:,}  \n"
                f"**Trainable params**: {n_trainable:,}  \n"
                f"**Beta (VAE)**: {self.vae_beta}  \n"
                f"**Grad clipping**: {self.max_grad_norm}  \n"
            ))
            self.writer.add_text("Model/architecture", str(self.model))

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0
        start_time = time.time()

        # Memory profiling: track per-experiment CPU memory via tracemalloc
        if self.profile_memory:
            tracemalloc.start()

        if self.profile_memory and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(1, max_epochs + 1):
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._eval_epoch(val_loader)

            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # TensorBoard logging
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                ckpt_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            if epoch % self.log_every_n_epochs == 0 or epoch == 1 or epoch == max_epochs:
                if self.is_vae:
                    logger.info(
                        f"    Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f} "
                        f"(recon={train_metrics['recon_loss']:.4f}, kl={train_metrics['kl_loss']:.4f})"
                    )
                else:
                    logger.info(f"    Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

            if patience_counter >= self.patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(best: epoch {best_epoch}, val={best_val_loss:.6f})"
                )
                break

        elapsed = time.time() - start_time

        ckpt_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device, weights_only=True)
            )

        # Collect memory metrics
        if self.profile_memory:
            _, peak_cpu_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_cpu_mem_mb = peak_cpu_bytes / (1024 * 1024)
        else:
            peak_cpu_mem_mb = float("nan")

        if self.profile_memory and self.device.type == "cuda":
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            peak_gpu_mem_mb = float("nan")

        logger.info(
            f"  Training complete in {elapsed:.1f}s. "
            f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch})"
        )

        # Log final metrics to TensorBoard (using add_text, NOT add_hparams
        # which creates a subdirectory that fragments the run data)
        if self.writer:
            summary_text = (
                f"**Best val loss**: {best_val_loss:.6f}  \n"
                f"**Best epoch**: {best_epoch}  \n"
                f"**Total epochs**: {len(history['train_loss'])}  \n"
                f"**Training time**: {elapsed:.1f}s  \n"
                f"**Latent dim**: {latent_dim}  \n"
                f"**Loss type**: {self.loss_type}  \n"
                f"**Model type**: {model_type}  \n"
                f"**Learning rate**: {self.learning_rate}  \n"
                f"**VAE beta**: {self.vae_beta}  \n"
            )
            self.writer.add_text("Results/summary", summary_text)
            if self.profile_memory:
                self.writer.add_scalar("Memory/peak_cpu_mb", peak_cpu_mem_mb, 0)
                if self.device.type == "cuda":
                    self.writer.add_scalar("Memory/peak_gpu_mb", peak_gpu_mem_mb, 0)
            self.writer.flush()

        # Auto-close writer to prevent file descriptor leaks across experiments
        self.close()

        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        history["elapsed_seconds"] = elapsed
        history["total_epochs"] = len(history["train_loss"])
        history["peak_cpu_mem_mb"] = peak_cpu_mem_mb
        history["peak_gpu_mem_mb"] = peak_gpu_mem_mb

        return history

    def close(self):
        """Close TensorBoard writer. Call when done with this trainer."""
        if self.writer:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass  # Best-effort cleanup
            self.writer = None

    def __del__(self):
        """Ensure writer is closed on garbage collection."""
        self.close()
