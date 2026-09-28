from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossTerms:
    similarity: torch.Tensor
    latent_kl: torch.Tensor
    smoothness: torch.Tensor
    displacement_kl: torch.Tensor
    total: torch.Tensor


def squared_difference(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Squared first-order finite differences along ``dim``."""
    if dim < 0:
        dim += tensor.ndim
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError("dim is out of range.")
    if tensor.shape[dim] < 2:
        raise ValueError("Selected dimension must contain at least two elements.")
    return torch.diff(tensor, dim=dim).square()


def dragnet_loss_terms(
    target: torch.Tensor,
    prediction: torch.Tensor,
    z_mu: torch.Tensor,
    z_logvar: torch.Tensor,
    z_prior_mu: torch.Tensor,
    z_prior_logvar: torch.Tensor,
    displacement: torch.Tensor,
    displacement_mu: torch.Tensor,
    displacement_covariance: torch.Tensor,
    *,
    latent_kl_weight: float = 2e-4,
    smoothness_weight: float = 0.03,
    displacement_kl_weight: float = 1e-4,
    epsilon: float = 1e-6,
) -> LossTerms:
    """Compute the four DragNet loss components used by the published implementation."""
    similarity = torch.mean((target - prediction).square())

    latent_element = (
        z_prior_logvar
        - z_logvar
        - 1.0
        + (z_logvar.exp() + (z_mu - z_prior_mu).square())
        / (z_prior_logvar.exp() + epsilon)
    )
    latent_kl = 0.5 * torch.mean(latent_element)

    diff_x = squared_difference(displacement, 2)
    diff_y = squared_difference(displacement, 3)
    smoothness = 0.5 * (diff_x.mean() + diff_y.mean())

    cov = displacement_covariance
    determinant = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] * cov[..., 1, 0]
    determinant = determinant.clamp_min(epsilon)
    mu = displacement_mu.permute(0, 2, 3, 1).unsqueeze(-1)
    quadratic = torch.matmul(mu.transpose(-1, -2), mu).squeeze(-1).squeeze(-1)
    displacement_element = (
        -torch.log(determinant)
        - 2.0
        + cov[..., 0, 0]
        + cov[..., 1, 1]
        + quadratic
    )
    displacement_kl = 0.5 * torch.mean(displacement_element)

    weighted_latent = latent_kl * latent_kl_weight
    weighted_smoothness = smoothness * smoothness_weight
    weighted_displacement = displacement_kl * displacement_kl_weight
    total = similarity + weighted_latent + weighted_smoothness + weighted_displacement
    return LossTerms(
        similarity=similarity,
        latent_kl=weighted_latent,
        smoothness=weighted_smoothness,
        displacement_kl=weighted_displacement,
        total=total,
    )
