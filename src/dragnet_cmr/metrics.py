from __future__ import annotations

import numpy as np


def rmse(target: np.ndarray, prediction: np.ndarray, *, border: int = 0) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have identical shapes.")
    if border < 0:
        raise ValueError("border must be non-negative.")
    if border:
        if min(target.shape[-2:]) <= 2 * border:
            raise ValueError("border is too large for the image size.")
        target = target[..., border:-border, border:-border]
        prediction = prediction[..., border:-border, border:-border]
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def jacobian_determinant_2d(displacement_xy: np.ndarray) -> np.ndarray:
    """Jacobian determinant of identity + 2D displacement in pixel coordinates."""
    displacement = np.asarray(displacement_xy, dtype=np.float64)
    if displacement.ndim != 3 or displacement.shape[0] != 2:
        raise ValueError("displacement_xy must have shape (2, H, W).")
    disp_x, disp_y = displacement
    dx_dy, dx_dx = np.gradient(disp_x)
    dy_dy, dy_dx = np.gradient(disp_y)
    return (1.0 + dx_dx) * (1.0 + dy_dy) - dy_dx * dx_dy


def nonpositive_jacobian_fraction(displacement_xy: np.ndarray) -> float:
    determinant = jacobian_determinant_2d(displacement_xy)
    return float(np.mean(determinant <= 0.0))
