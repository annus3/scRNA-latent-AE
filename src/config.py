"""
Configuration management for the scRNA-seq autoencoder project.

Loads YAML config and provides structured access via dataclasses.
"""

import os
import yaml
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PathsConfig:
    """File system paths."""
    data_dir: str = "data"
    results_dir: str = "results"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    """Data loading and representation options."""
    batch_key: Optional[str] = None
    label_key: Optional[str] = None
    use_backed_mode: bool = False
    backed_threshold_cells: int = 100000
    dense_conversion_policy: str = "auto"  # auto|never|always
    large_run_mode: bool = False
    keep_backed_until_split: bool = False
    backed_probe_cells: int = 512


@dataclass
class PreprocessingConfig:
    """Data preprocessing parameters."""
    n_top_genes: int = 2000
    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    target_sum: float = 10000
    hvg_flavor: str = "seurat_v3"
    leiden_resolution: float = 1.0
    n_pcs: int = 50
    n_neighbors: int = 10
    scale_policy: str = "pca_only"  # none|pca_only|full
    run_pca: bool = True
    integration_mode: str = "none"  # none|harmony|bbknn


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 256
    max_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 10
    train_frac: float = 0.8
    val_frac: float = 0.1
    device: str = "auto"
    loss_type: str = "mse"  # mse|nb|zinb
    profile_memory: bool = True
    num_workers: int = 0      # DataLoader workers; set >0 on HPC
    pin_memory: bool = True   # pin_memory for GPU; auto-disabled on CPU
    max_grad_norm: float = 5.0  # gradient clipping; 0 to disable
    tensorboard: bool = True    # enable TensorBoard logging
    log_every_n_epochs: int = 1  # epoch log cadence for tail-friendly monitoring


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"  # relu|sigmoid|tanh|leaky_relu
    dropout: float = 0.0
    vae_beta: float = 1.0


@dataclass
class ExperimentConfig:
    """Experiment sweep parameters."""
    latent_dims: List[int] = field(
        default_factory=lambda: [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
    )
    model_types: List[str] = field(default_factory=lambda: ["ae", "vae", "scvi"])
    datasets: List[str] = field(default_factory=lambda: ["pbmc68k_reduced", "pbmc3k", "paul15"])
    dataset_tiers: List[str] = field(default_factory=lambda: ["tier_a", "tier_b"])
    n_seeds: int = 3
    scvi_max_epochs: int = 100
    scvi_gene_likelihood: str = "nb"
    loss_matrix: List[str] = field(default_factory=list)
    kmeans_n_init: int = 10
    auto_d_report: bool = True
    formula_candidates: List[str] = field(default_factory=lambda: ["linear", "power", "sqrt", "logk"])
    min_k_for_fit: int = 3
    require_ground_truth_for_external_metrics: bool = True
    silhouette_max_cells: int = 0
    centrality_policy: str = "full"  # full|sample|skip
    centrality_threshold_cells: int = 0
    centrality_sample_size: int = 0
    centrality_n_neighbors: int = 15
    # Batch-aware evaluation guardrails
    batch_metrics_enabled: bool = True
    batch_metrics_max_cells: int = 0
    batch_metrics_knn_k: int = 15
    # Primary vs exploratory auto-d reporting controls
    auto_d_primary_metric: str = "ari"
    auto_d_primary_require_curated_labels: bool = True
    auto_d_primary_include_label_sources: List[str] = field(default_factory=lambda: ["provided:*"])
    auto_d_primary_exclude_dataset_patterns: List[str] = field(default_factory=list)
    auto_d_generate_exploratory_report: bool = True
    auto_d_loo_validation: bool = True
    auto_d_secondary_metric_mode: str = "current_fallback"  # current_fallback|ari|ami|silhouette_kmeans|silhouette


@dataclass
class ProjectConfig:
    """Top-level project configuration."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    seed: int = 42

    def validate(self) -> None:
        """Validate selected option values."""
        allowed_loss = {"mse", "nb", "zinb"}
        if self.training.loss_type not in allowed_loss:
            raise ValueError(f"training.loss_type must be one of {sorted(allowed_loss)}")

        allowed_scale = {"none", "pca_only", "full"}
        if self.preprocessing.scale_policy not in allowed_scale:
            raise ValueError(f"preprocessing.scale_policy must be one of {sorted(allowed_scale)}")

        allowed_integration = {"none", "harmony", "bbknn"}
        if self.preprocessing.integration_mode not in allowed_integration:
            raise ValueError(
                f"preprocessing.integration_mode must be one of {sorted(allowed_integration)}"
            )

        allowed_dense = {"auto", "never", "always"}
        if self.data.dense_conversion_policy not in allowed_dense:
            raise ValueError(f"data.dense_conversion_policy must be one of {sorted(allowed_dense)}")
        if self.data.backed_probe_cells < 16:
            raise ValueError("data.backed_probe_cells must be >= 16")

        if any(not isinstance(item, str) for item in self.experiment.loss_matrix):
            raise ValueError("experiment.loss_matrix must contain only strings of form model:loss")
        if any(":" not in item for item in self.experiment.loss_matrix):
            raise ValueError("experiment.loss_matrix entries must be in form model:loss")
        if self.experiment.silhouette_max_cells < 0:
            raise ValueError("experiment.silhouette_max_cells must be >= 0")
        allowed_centrality = {"full", "sample", "skip"}
        if self.experiment.centrality_policy not in allowed_centrality:
            raise ValueError(
                "experiment.centrality_policy must be one of "
                f"{sorted(allowed_centrality)}"
            )
        if self.experiment.centrality_threshold_cells < 0:
            raise ValueError("experiment.centrality_threshold_cells must be >= 0")
        if self.experiment.centrality_sample_size < 0:
            raise ValueError("experiment.centrality_sample_size must be >= 0")
        if self.experiment.centrality_n_neighbors < 1:
            raise ValueError("experiment.centrality_n_neighbors must be >= 1")
        if self.experiment.batch_metrics_max_cells < 0:
            raise ValueError("experiment.batch_metrics_max_cells must be >= 0")
        if self.experiment.batch_metrics_knn_k < 1:
            raise ValueError("experiment.batch_metrics_knn_k must be >= 1")
        allowed_primary_metric = {"ari", "ami", "silhouette_kmeans", "silhouette"}
        if self.experiment.auto_d_primary_metric not in allowed_primary_metric:
            raise ValueError(
                "experiment.auto_d_primary_metric must be one of "
                f"{sorted(allowed_primary_metric)}"
            )
        allowed_secondary_mode = {"current_fallback", "ari", "ami", "silhouette_kmeans", "silhouette"}
        if self.experiment.auto_d_secondary_metric_mode not in allowed_secondary_mode:
            raise ValueError(
                "experiment.auto_d_secondary_metric_mode must be one of "
                f"{sorted(allowed_secondary_mode)}"
            )
        if any(not isinstance(item, str) for item in self.experiment.auto_d_primary_include_label_sources):
            raise ValueError("experiment.auto_d_primary_include_label_sources must contain only strings")
        if any(not isinstance(item, str) for item in self.experiment.auto_d_primary_exclude_dataset_patterns):
            raise ValueError("experiment.auto_d_primary_exclude_dataset_patterns must contain only strings")

        if self.training.log_every_n_epochs < 1:
            raise ValueError("training.log_every_n_epochs must be >= 1")

    def resolve_device(self):
        """Resolve the device string to a torch.device."""
        import torch
        if self.training.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.training.device)

    def ensure_dirs(self) -> None:
        """Create all top-level output directories if they don't exist."""
        for dir_path in [
            self.paths.data_dir,
            os.path.join(self.paths.data_dir, "raw"),
            os.path.join(self.paths.data_dir, "processed"),
            self.paths.results_dir,
            os.path.join(self.paths.results_dir, "datasets"),
            os.path.join(self.paths.results_dir, "global"),
            os.path.join(self.paths.results_dir, "runs"),
            self.paths.log_dir,
            self.paths.checkpoint_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    def ensure_dataset_dirs(self, dataset_name: str) -> dict:
        """
        Create per-dataset output subdirectories and return their paths.

        Directory structure:
            results/datasets/<dataset>/tables/   -- dataset result tables
            results/datasets/<dataset>/figures/  -- dataset plots
            results/datasets/<dataset>/latents/  -- latent exports
            checkpoints/<dataset>/               -- dataset checkpoints
            logs/<dataset>/                      -- dataset logs
        """
        dirs = {
            "dataset_root": os.path.join(self.paths.results_dir, "datasets", dataset_name),
            "tables": os.path.join(self.paths.results_dir, "datasets", dataset_name, "tables"),
            "figures": os.path.join(self.paths.results_dir, "datasets", dataset_name, "figures"),
            "latents": os.path.join(self.paths.results_dir, "datasets", dataset_name, "latents"),
            "checkpoints": os.path.join(self.paths.checkpoint_dir, dataset_name),
            "logs": os.path.join(self.paths.log_dir, dataset_name),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        return dirs

    def ensure_global_result_dirs(self) -> dict:
        """
        Create global result/report/run directories and return their paths.
        """
        dirs = {
            "global_root": os.path.join(self.paths.results_dir, "global"),
            "tables": os.path.join(self.paths.results_dir, "global", "tables"),
            "reports": os.path.join(self.paths.results_dir, "global", "reports"),
            "runs": os.path.join(self.paths.results_dir, "runs"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        return dirs

    def set_seed(self, seed: Optional[int] = None) -> int:
        """Set all random seeds for reproducibility and return the active seed."""
        import torch

        active_seed = self.seed if seed is None else int(seed)
        random.seed(active_seed)
        np.random.seed(active_seed)
        torch.manual_seed(active_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(active_seed)
        return active_seed


def _build_nested(dataclass_type, raw_dict: dict):
    """Recursively build a dataclass from a raw dictionary."""
    if raw_dict is None:
        return dataclass_type()
    filtered = {}
    for k, v in raw_dict.items():
        if hasattr(dataclass_type, "__dataclass_fields__") and k in dataclass_type.__dataclass_fields__:
            filtered[k] = v
    return dataclass_type(**filtered)


def _normalize_path(value: str, field_name: str) -> str:
    """Expand env vars/user and normalize to an absolute path."""
    raw = str(value)
    expanded = os.path.expandvars(raw)
    expanded = os.path.expanduser(expanded)
    if "$" in expanded:
        raise ValueError(
            f"Unresolved environment variable in paths.{field_name}: {raw}. "
            "Set the environment variable(s) before loading this config."
        )
    return os.path.abspath(expanded)


def _normalize_paths_config(paths: PathsConfig) -> PathsConfig:
    """Normalize all configured paths to explicit absolute paths."""
    paths.data_dir = _normalize_path(paths.data_dir, "data_dir")
    paths.results_dir = _normalize_path(paths.results_dir, "results_dir")
    paths.log_dir = _normalize_path(paths.log_dir, "log_dir")
    paths.checkpoint_dir = _normalize_path(paths.checkpoint_dir, "checkpoint_dir")
    return paths


def load_config(config_path: str) -> ProjectConfig:
    """
    Load project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ProjectConfig dataclass with all settings.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    config = ProjectConfig(
        paths=_normalize_paths_config(_build_nested(PathsConfig, raw.get("paths"))),
        data=_build_nested(DataConfig, raw.get("data")),
        preprocessing=_build_nested(PreprocessingConfig, raw.get("preprocessing")),
        training=_build_nested(TrainingConfig, raw.get("training")),
        model=_build_nested(ModelConfig, raw.get("model")),
        experiment=_build_nested(ExperimentConfig, raw.get("experiment")),
        seed=raw.get("seed", 42),
    )
    config.validate()
    return config
