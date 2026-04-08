"""
Variational Autoencoder for scRNA-seq data.

Supports MSE and count-likelihood heads (NB/ZINB).
"""

from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.autoencoder import _build_encoder, _build_decoder_trunk, _get_activation


class VAE(nn.Module):
    """Variational Autoencoder with optional NB/ZINB reconstruction heads.

    Architecture (for hidden_dims=[128, 64]):
        Encoder: input -> 128 -> 64 -> (mu, logvar) of size latent_dim
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

        # Encoder outputs 2*latent_dim for (mu, logvar)
        self.encoder = _build_encoder(
            input_dim, hidden_dims, 2 * latent_dim,
            activation=activation, dropout=dropout
        )

        # Symmetric decoder trunk: latent -> reversed(hidden_dims)
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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.decoder_hidden(z)

        if self.loss_type == "mse":
            return self.recon_head(h)

        mu = F.softplus(self.recon_head(h)) + 1e-4
        theta = F.softplus(self.log_theta).unsqueeze(0).expand_as(mu) + 1e-4

        if self.loss_type == "nb":
            return {"mu": mu, "theta": theta}

        pi = torch.sigmoid(self.pi_head(h))
        return {"mu": mu, "theta": theta, "pi": pi}

    def forward(self, x: torch.Tensor, deterministic: bool = False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        x_out = self.decode(z)
        return x_out, mu, logvar

    @torch.no_grad()
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu
