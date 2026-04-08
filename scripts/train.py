#!/usr/bin/env python3
"""CLI: Train a single autoencoder model."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.dataset import make_dataloaders, split_adata
from src.data.preprocessor import load_processed
from src.models.autoencoder import Autoencoder
from src.models.scvi_wrapper import ScVIWrapper
from src.models.vae import VAE
from src.training.trainer import Trainer
from src.utils.logging_utils import (
    make_tensorboard_run_subdir,
    setup_logger,
    write_tensorboard_run_metadata,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _should_preload(adata, config) -> bool:
    if config.data.dense_conversion_policy == "never":
        return False
    if config.data.use_backed_mode and adata.n_obs >= config.data.backed_threshold_cells:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder model")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["ae", "vae", "scvi"])
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--processed_path", type=str, default=None)
    parser.add_argument("--loss_type", type=str, default=None, choices=["mse", "nb", "zinb"])
    parser.add_argument("--batch_key", type=str, default=None)
    parser.add_argument("--profile_memory", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.ensure_dirs()
    dataset_dirs = config.ensure_dataset_dirs(args.dataset)

    if args.device:
        config.training.device = args.device
    if args.max_epochs:
        config.training.max_epochs = args.max_epochs
        config.experiment.scvi_max_epochs = args.max_epochs
    if args.seed is not None:
        config.seed = args.seed
    if args.loss_type:
        config.training.loss_type = args.loss_type
    if args.batch_key:
        config.data.batch_key = args.batch_key
    if args.profile_memory:
        config.training.profile_memory = True

    active_seed = config.set_seed(config.seed)
    device = config.resolve_device()

    # Optimize A100 Tensor Core performance
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    log_file = f"train_{args.dataset}_{args.model}_d{args.latent_dim}.log"
    logger = setup_logger("train", dataset_dirs["logs"], log_file=log_file)

    logger.info(
        f"=== Training: {args.model} | d={args.latent_dim} | {args.dataset} "
        f"| device={device} | loss={config.training.loss_type} | seed={active_seed} ==="
    )

    data_path = args.processed_path or os.path.join(config.paths.data_dir, "processed", f"{args.dataset}.h5ad")
    adata = load_processed(data_path)
    n_genes = adata.n_vars

    if args.model == "scvi":
        if config.training.loss_type != "mse":
            config.experiment.scvi_gene_likelihood = config.training.loss_type

        if config.experiment.scvi_gene_likelihood not in {"nb", "zinb"}:
            raise ValueError(
                "scVI only supports gene likelihood 'nb' or 'zinb'. "
                f"Got: {config.experiment.scvi_gene_likelihood}"
            )

        tb_dir = None
        run_tag = make_tensorboard_run_subdir(
            dataset=args.dataset,
            model_type="scvi",
            loss_type=config.experiment.scvi_gene_likelihood,
            latent_dim=args.latent_dim,
            seed=active_seed,
        )
        if config.training.tensorboard:
            run_stamp = _utc_now().strftime("train_%Y%m%dT%H%M%SZ")
            tb_dir = os.path.join(config.paths.log_dir, "tensorboard", run_stamp)
            write_tensorboard_run_metadata(
                tb_dir,
                run_tag,
                {
                    "dataset": args.dataset,
                    "model_type": "scvi",
                    "loss_type": config.experiment.scvi_gene_likelihood,
                    "latent_dim": int(args.latent_dim),
                    "seed": int(active_seed),
                },
            )

        scvi_wrapper = ScVIWrapper(
            latent_dim=args.latent_dim,
            n_hidden=config.model.hidden_dims[0] if config.model.hidden_dims else 128,
            n_layers=len(config.model.hidden_dims),
            gene_likelihood=config.experiment.scvi_gene_likelihood,
            max_epochs=config.experiment.scvi_max_epochs,
            learning_rate=config.training.learning_rate,
            seed=active_seed,
            batch_key=config.data.batch_key,
            tensorboard_dir=tb_dir,
            run_tag=run_tag,
            log_every_n_epochs=config.training.log_every_n_epochs,
        )
        _ = scvi_wrapper.setup_and_train(adata)
        logger.info("=== scVI training complete ===")
    else:
        if config.training.loss_type in {"nb", "zinb"} and "counts" not in adata.layers:
            raise ValueError("NB/ZINB loss requested but adata.layers['counts'] is missing.")

        adata_train, adata_val, adata_test = split_adata(
            adata, config.training.train_frac, config.training.val_frac, seed=active_seed
        )

        data_layer = "counts" if config.training.loss_type in {"nb", "zinb"} else None
        use_pin = config.training.pin_memory and (device.type == "cuda")
        preload = _should_preload(adata, config)

        train_loader, val_loader, _ = make_dataloaders(
            adata_train,
            adata_val,
            adata_test,
            batch_size=config.training.batch_size,
            layer=data_layer,
            dense_policy=config.data.dense_conversion_policy,
            preload=preload,
            num_workers=config.training.num_workers,
            pin_memory=use_pin,
        )

        if args.model == "ae":
            model = Autoencoder(
                input_dim=n_genes,
                latent_dim=args.latent_dim,
                hidden_dims=config.model.hidden_dims,
                activation=config.model.activation,
                dropout=config.model.dropout,
                loss_type=config.training.loss_type,
            )
        else:
            model = VAE(
                input_dim=n_genes,
                latent_dim=args.latent_dim,
                hidden_dims=config.model.hidden_dims,
                activation=config.model.activation,
                dropout=config.model.dropout,
                loss_type=config.training.loss_type,
            )

        ckpt_name = f"{args.dataset}_{args.model}_d{args.latent_dim}_{config.training.loss_type}"
        run_tag = make_tensorboard_run_subdir(
            dataset=args.dataset,
            model_type=args.model,
            loss_type=config.training.loss_type,
            latent_dim=args.latent_dim,
            seed=active_seed,
        )
        tb_dir = None
        if config.training.tensorboard:
            run_stamp = _utc_now().strftime("train_%Y%m%dT%H%M%SZ")
            tb_dir = os.path.join(config.paths.log_dir, "tensorboard", run_stamp)
            write_tensorboard_run_metadata(
                tb_dir,
                run_tag,
                {
                    "dataset": args.dataset,
                    "model_type": args.model,
                    "loss_type": config.training.loss_type,
                    "latent_dim": int(args.latent_dim),
                    "seed": int(active_seed),
                },
            )

        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            early_stopping_patience=config.training.early_stopping_patience,
            vae_beta=config.model.vae_beta,
            checkpoint_dir=dataset_dirs["checkpoints"],
            loss_type=config.training.loss_type,
            profile_memory=config.training.profile_memory,
            max_grad_norm=config.training.max_grad_norm,
            tensorboard_dir=tb_dir,
            run_tag=run_tag,
            log_every_n_epochs=config.training.log_every_n_epochs,
        )
        history = trainer.train(
            train_loader,
            val_loader,
            max_epochs=config.training.max_epochs,
            checkpoint_name=ckpt_name,
        )
        trainer.close()

        history_path = os.path.join(dataset_dirs["tables"], f"history_{ckpt_name}.json")
        serializable = {k: v for k, v in history.items() if isinstance(v, (int, float, str, list))}
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"  Saved history to {history_path}")

    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()
