import torch

from dragnet_cmr.losses import dragnet_loss_terms, squared_difference


def test_squared_difference_shape():
    tensor = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    assert squared_difference(tensor, 2).shape == (1, 2, 2, 4)
    assert squared_difference(tensor, 3).shape == (1, 2, 3, 3)


def test_loss_terms_are_finite():
    target = torch.zeros(1, 1, 8, 8)
    prediction = torch.zeros_like(target)
    z_mu = torch.zeros(1, 4)
    z_logvar = torch.zeros(1, 4)
    displacement = torch.zeros(1, 2, 8, 8)
    displacement_mu = torch.zeros_like(displacement)
    covariance = torch.eye(2).reshape(1, 1, 1, 2, 2).expand(1, 8, 8, 2, 2).clone()
    terms = dragnet_loss_terms(
        target,
        prediction,
        z_mu,
        z_logvar,
        z_mu,
        z_logvar,
        displacement,
        displacement_mu,
        covariance,
    )
    assert torch.isfinite(terms.total)
    assert abs(terms.total.item()) < 1e-8
