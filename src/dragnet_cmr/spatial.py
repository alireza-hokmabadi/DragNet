from __future__ import annotations

import torch
import torch.nn.functional as F


def identity_grid_2d(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an identity sampling grid for ``grid_sample`` with align_corners=False."""
    theta = torch.zeros((batch_size, 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    return F.affine_grid(theta, size=[batch_size, 1, height, width], align_corners=False)


def warp_image_2d(image: torch.Tensor, displacement_xy: torch.Tensor) -> torch.Tensor:
    """Warp a 2D image using a displacement field expressed in pixel units.

    Parameters
    ----------
    image:
        Tensor shaped ``(B, C, H, W)``.
    displacement_xy:
        Tensor shaped ``(B, 2, H, W)`` where channel 0 is x/column displacement
        and channel 1 is y/row displacement, both in pixels.
    """
    if image.ndim != 4:
        raise ValueError("image must have shape (B, C, H, W).")
    if displacement_xy.ndim != 4 or displacement_xy.shape[1] != 2:
        raise ValueError("displacement_xy must have shape (B, 2, H, W).")
    if image.shape[0] != displacement_xy.shape[0] or image.shape[2:] != displacement_xy.shape[2:]:
        raise ValueError("image and displacement_xy must have matching batch/spatial dimensions.")

    batch_size, _, height, width = image.shape
    grid = identity_grid_2d(
        batch_size,
        height,
        width,
        device=image.device,
        dtype=image.dtype,
    )
    scale = torch.tensor(
        [2.0 / width, 2.0 / height],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 1, 2)
    deformation = grid + displacement_xy.permute(0, 2, 3, 1) * scale
    return F.grid_sample(
        image,
        deformation,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
