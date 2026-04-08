"""
Loss functions for autoencoder models.

Provides MSE, KL divergence, and count-based likelihood losses (NB/ZINB).
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error reconstruction loss."""
    return F.mse_loss(x_hat, x, reduction="mean")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between N(mu, sigma^2) and N(0, I)."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def _nb_negative_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative binomial negative log-likelihood averaged over batch."""
    x = torch.clamp(x, min=0.0)
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)

    log_theta_mu = torch.log(theta + mu + eps)
    nll = (
        torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )
    # nll here is log prob; negate and mean
    return -torch.mean(torch.sum(nll, dim=1))


def nb_loss(x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """NB loss from model outputs dict containing mu/theta."""
    if "mu" not in outputs or "theta" not in outputs:
        raise ValueError("NB loss requires outputs with keys: 'mu', 'theta'")
    return _nb_negative_log_likelihood(x, outputs["mu"], outputs["theta"])


def zinb_loss(
    x: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """ZINB loss with dropout logits/probabilities."""
    if "mu" not in outputs or "theta" not in outputs or "pi" not in outputs:
        raise ValueError("ZINB loss requires outputs with keys: 'mu', 'theta', 'pi'")

    x = torch.clamp(x, min=0.0)
    mu = torch.clamp(outputs["mu"], min=eps)
    theta = torch.clamp(outputs["theta"], min=eps)
    pi = torch.clamp(outputs["pi"], min=eps, max=1.0 - eps)

    # NB log-probability
    log_theta_mu = torch.log(theta + mu + eps)
    log_nb = (
        torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )

    # P(x=0) under NB
    zero_nb = torch.pow(theta / (theta + mu + eps), theta)

    # ZINB mixture
    is_zero = (x < eps).float()
    log_prob_zero = torch.log(pi + (1.0 - pi) * zero_nb + eps)
    log_prob_nonzero = torch.log(1.0 - pi + eps) + log_nb
    log_prob = is_zero * log_prob_zero + (1.0 - is_zero) * log_prob_nonzero

    return -torch.mean(torch.sum(log_prob, dim=1))


def reconstruction_loss(
    x: torch.Tensor,
    model_output,
    loss_type: str,
) -> torch.Tensor:
    """Dispatch reconstruction loss by type."""
    if loss_type == "mse":
        if isinstance(model_output, dict):
            raise ValueError("MSE loss expects tensor reconstruction output, got dict")
        return mse_loss(x, model_output)
    if loss_type == "nb":
        if not isinstance(model_output, dict):
            raise ValueError("NB loss expects dict outputs")
        return nb_loss(x, model_output)
    if loss_type == "zinb":
        if not isinstance(model_output, dict):
            raise ValueError("ZINB loss expects dict outputs")
        return zinb_loss(x, model_output)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def vae_loss(
    x: torch.Tensor,
    model_output,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    loss_type: str = "mse",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined VAE loss: reconstruction + beta * KL."""
    recon = reconstruction_loss(x, model_output, loss_type=loss_type)
    kl = kl_divergence(mu, logvar)
    total = recon + beta * kl
    return total, recon, kl
