"""
Logging utilities for the scRNA-seq autoencoder project.

Provides consistent logging to both console and file across all scripts,
and shared helpers for TensorBoard run organization.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, Optional


def _safe_tb_component(value: Any) -> str:
    """Convert a value into a filesystem-safe TensorBoard path component."""
    text = str(value).strip()
    if not text:
        return "unknown"

    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")

    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars)
    return safe or "unknown"


def make_tensorboard_run_subdir(
    dataset: str,
    model_type: str,
    loss_type: str,
    latent_dim: int,
    seed: Optional[int] = None,
) -> str:
    """Return hierarchical TensorBoard subdir: dataset/model/loss/d_X/seed_Y."""
    parts = [
        _safe_tb_component(dataset),
        _safe_tb_component(model_type),
        _safe_tb_component(loss_type),
        f"d_{int(latent_dim)}",
    ]
    if seed is not None:
        parts.append(f"seed_{int(seed)}")
    return os.path.join(*parts)


def write_tensorboard_run_metadata(
    tensorboard_dir: Optional[str],
    run_subdir: str,
    metadata: Dict[str, Any],
    filename: str = "run_metadata.json",
) -> Optional[str]:
    """Best-effort metadata write inside a TensorBoard run directory."""
    if not tensorboard_dir:
        return None

    try:
        run_dir = os.path.join(tensorboard_dir, run_subdir)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        return path
    except Exception:
        return None


def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger with console and optional file output.

    Also configures the root logger with the same handlers so module-level
    loggers (e.g., src.training.trainer, src.models.scvi_wrapper) are visible
    in the same output stream/file.

    Args:
        name: Logger name (typically the script or module name).
        log_dir: Directory for log files.
        log_file: Specific log file name. If None, uses '{name}.log'.
        level: Logging level (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured.
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    file_handler = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        if log_file is None:
            log_file = f"{name}.log"
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent duplicate handling through root for this named logger.
    logger.propagate = False

    # Ensure module loggers are captured consistently.
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        if file_handler is not None:
            root_logger.addHandler(file_handler)

    return logger
