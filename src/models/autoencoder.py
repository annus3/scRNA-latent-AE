"""
Standard Autoencoder for scRNA-seq data.

Symmetric encoder-decoder with configurable layers and activation.
Supports count-likelihood heads for NB/ZINB experiments.
"""

from typing import List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    activations = {
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(0.1),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: '{name}'. Available: {list(activations)}")
    return activations[name]


def _build_encoder(
    input_dim: int,
    hidden_dims: List[int],
    latent_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build encoder MLP: input_dim -> hidden_dims -> latent_dim."""
    layers = []
    prev_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(_get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim

    layers.append(nn.Linear(prev_dim, latent_dim))
    return nn.Sequential(*layers)


def _build_decoder_trunk(
    latent_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build symmetric decoder trunk: latent_dim -> reversed(hidden_dims).

    The trunk outputs features of size hidden_dims[0] (the widest hidden layer),
    which are then fed to output heads. This ensures the decoder is symmetric
    with the encoder.
    """
    reversed_dims = list(reversed(hidden_dims))
    layers = []
    prev_dim = latent_dim

    for h_dim in reversed_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(_get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim

    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Standard fully-connected autoencoder with optional NB/ZINB heads.

    Architecture (for hidden_dims=[128, 64]):
        Encoder: input -> 128 -> 64 -> latent
        Decoder: latent -> 64 -> 128 -> input  (symmetric)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        loss_type: str = "mse",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        if loss_type not in {"mse", "nb", "zinb"}:
            raise ValueError("loss_type must be one of: mse, nb, zinb")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self._hidden_dims = hidden_dims

        # Encoder: input -> hidden_dims -> latent
        self.encoder = _build_encoder(
            input_dim, hidden_dims, latent_dim,
            activation=activation, dropout=dropout
        )

        # Decoder trunk: latent -> reversed(hidden_dims)
        # Output dimension = hidden_dims[0] (widest layer)
        self.decoder_hidden = _build_decoder_trunk(
            latent_dim, hidden_dims,
            activation=activation, dropout=dropout
        )

        # Output dimension of decoder trunk
        trunk_out_dim = hidden_dims[0] if hidden_dims else latent_dim

        # Output heads
        self.recon_head = nn.Linear(trunk_out_dim, input_dim)
        self.pi_head = nn.Linear(trunk_out_dim, input_dim)
        self.log_theta = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _decode_hidden(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_hidden(z)

    def decode(self, z: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        h = self._decode_hidden(z)

        if self.loss_type == "mse":
            return self.recon_head(h)

        mu = F.softplus(self.recon_head(h)) + 1e-4
        theta = F.softplus(self.log_theta).unsqueeze(0).expand_as(mu) + 1e-4

        if self.loss_type == "nb":
            return {"mu": mu, "theta": theta}

        # zinb
        pi = torch.sigmoid(self.pi_head(h))
        return {"mu": mu, "theta": theta, "pi": pi}

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z)

    @torch.no_grad()
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
