from __future__ import annotations

from pathlib import Path

import numpy as np


def _ellipse(
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    center_y: float,
    center_x: float,
    radius_y: float,
    radius_x: float,
) -> np.ndarray:
    return ((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2 <= 1.0


def make_synthetic_sequence(
    *,
    frame_count: int = 7,
    size: int = 128,
    seed: int = 1234,
) -> np.ndarray:
    """Create a clearly synthetic moving-ellipse phantom, not derived from medical data."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2.")
    if size != 128:
        raise ValueError("The public DragNet implementation is fixed to 128x128 images.")

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    sequence = np.zeros((frame_count, size, size), dtype=np.float32)
    phase_offset = rng.uniform(-0.12, 0.12)

    for frame_index in range(frame_count):
        phase = 2.0 * np.pi * frame_index / frame_count + phase_offset
        center_y = 64.0 + 2.5 * np.sin(phase)
        center_x = 64.0 + 3.5 * np.cos(phase)
        scale = 1.0 + 0.12 * np.cos(phase)

        outer = _ellipse(
            yy,
            xx,
            center_y=center_y,
            center_x=center_x,
            radius_y=34.0 * scale,
            radius_x=27.0 * scale,
        )
        inner = _ellipse(
            yy,
            xx,
            center_y=center_y + 2.0,
            center_x=center_x,
            radius_y=19.0 * scale,
            radius_x=14.0 * scale,
        )
        second = _ellipse(
            yy,
            xx,
            center_y=center_y + 3.0,
            center_x=center_x + 27.0,
            radius_y=15.0 * (1.0 - 0.06 * np.cos(phase)),
            radius_x=10.0 * (1.0 - 0.06 * np.cos(phase)),
        )

        image = np.zeros((size, size), dtype=np.float32)
        image[outer] = 0.38
        image[inner] = 0.86
        image[second] = 0.72
        radial = np.exp(-((xx - 64.0) ** 2 + (yy - 64.0) ** 2) / (2.0 * 52.0**2))
        image += 0.06 * radial.astype(np.float32)
        sequence[frame_index] = np.clip(image, 0.0, 1.0)

    return (sequence * 255.0).round().astype(np.uint8)


def make_synthetic_dataset(
    sample_count: int = 8,
    *,
    frame_count: int = 7,
    seed: int = 1234,
) -> np.ndarray:
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    return np.stack(
        [
            make_synthetic_sequence(frame_count=frame_count, seed=seed + index)
            for index in range(sample_count)
        ],
        axis=0,
    )


def write_synthetic_npz(
    path: str | Path,
    *,
    sample_count: int = 8,
    frame_count: int = 7,
    seed: int = 1234,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sequences = make_synthetic_dataset(
        sample_count=sample_count,
        frame_count=frame_count,
        seed=seed,
    )
    np.savez_compressed(target, sequences=sequences)
    return target
