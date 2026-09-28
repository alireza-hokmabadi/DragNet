from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import jacobian_determinant_2d, rmse


def flow_to_rgb(displacement_xy: np.ndarray) -> np.ndarray:
    flow = np.asarray(displacement_xy, dtype=np.float64)
    if flow.shape[0] != 2:
        raise ValueError("displacement_xy must have shape (2, H, W).")
    x, y = flow
    magnitude = np.hypot(x, y)
    angle = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    saturation = np.ones_like(magnitude)
    scale = float(np.percentile(magnitude, 99))
    value = np.ones_like(magnitude) if scale <= 0 else np.clip(magnitude / scale, 0.0, 1.0)
    hsv = np.stack((angle, saturation, value), axis=-1)
    return plt.matplotlib.colors.hsv_to_rgb(hsv)


def save_sequence_preview(sequence: np.ndarray, path: str | Path) -> Path:
    data = np.asarray(sequence)
    if data.ndim != 3:
        raise ValueError("sequence must have shape (T, H, W).")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, data.shape[0], figsize=(2.2 * data.shape[0], 2.4), squeeze=False)
    for index, ax in enumerate(axes[0]):
        ax.imshow(data[index], cmap="gray", vmin=float(data.min()), vmax=float(data.max()))
        ax.set_title(f"phase {index}")
        ax.axis("off")
    fig.suptitle("Programmatically generated synthetic phantom")
    fig.tight_layout()
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target


def save_registration_figure(
    target_sequence: np.ndarray,
    prediction_sequence: np.ndarray,
    displacement: np.ndarray,
    path: str | Path,
    *,
    border: int = 2,
) -> Path:
    target = np.asarray(target_sequence)
    prediction = np.asarray(prediction_sequence)
    flow = np.asarray(displacement)
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("target_sequence and prediction_sequence must match with shape (T,H,W).")
    if flow.shape != (target.shape[0], 2, target.shape[1], target.shape[2]):
        raise ValueError("displacement must have shape (T,2,H,W).")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = target.shape[0]
    fig, axes = plt.subplots(4, frame_count, figsize=(2.3 * frame_count, 8.5), squeeze=False)
    for index in range(frame_count):
        axes[0, index].imshow(target[index], cmap="gray", vmin=0, vmax=1)
        axes[0, index].set_title(f"phase {index}")
        axes[1, index].imshow(prediction[index], cmap="gray", vmin=0, vmax=1)
        error = rmse(target[index], prediction[index], border=border)
        axes[1, index].set_title(f"RMSE {error:.3f}")
        axes[2, index].imshow(flow_to_rgb(flow[index]))
        jacobian = jacobian_determinant_2d(flow[index][[1, 0]])
        axes[3, index].imshow(jacobian, cmap="coolwarm", vmin=-2, vmax=2)
        for row in range(4):
            axes[row, index].axis("off")
    axes[0, 0].set_ylabel("target")
    axes[1, 0].set_ylabel("registered")
    axes[2, 0].set_ylabel("DVF")
    axes[3, 0].set_ylabel("Jacobian")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output
