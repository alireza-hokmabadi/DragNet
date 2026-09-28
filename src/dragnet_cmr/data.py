from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

BlurMode = Literal["dragnet", "spatial"]


def _validate_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4:
        # (N, T, H, W)
        pass
    elif array.ndim == 5 and array.shape[2] == 1:
        array = array[:, :, 0]
    else:
        raise ValueError("Expected sequences with shape (N,T,H,W) or (N,T,1,H,W).")
    if array.shape[-2:] != (128, 128):
        raise ValueError("DragNet expects 128x128 images.")
    if array.shape[1] < 2:
        raise ValueError("Each sequence must contain at least two frames.")
    if not np.isfinite(array).all():
        raise ValueError("Sequence data contains NaN or infinite values.")
    return array


def load_sequences(path: str | Path, *, key: str = "sequences") -> np.ndarray:
    """Load cine sequences from a NumPy ``.npz`` or ``.npy`` file."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".npy":
        array = np.load(target, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(target, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"NPZ file does not contain key {key!r}.")
            array = archive[key]
    else:
        raise ValueError("Only .npy and .npz inputs are supported.")
    return _validate_array(np.asarray(array))


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Convert a sequence to float32 in [0, 1] without guessing arbitrary scaling."""
    sequence = np.asarray(sequence)
    if np.issubdtype(sequence.dtype, np.integer):
        if sequence.min() < 0 or sequence.max() > 255:
            raise ValueError("Integer image data must be in [0, 255].")
        return sequence.astype(np.float32) / 255.0

    sequence = sequence.astype(np.float32, copy=False)
    if float(sequence.min()) < -1e-6 or float(sequence.max()) > 1.0 + 1e-6:
        raise ValueError("Floating-point image data must already be in [0, 1].")
    return np.clip(sequence, 0.0, 1.0)


def blur_sequence(sequence: np.ndarray, sigma: float, *, mode: BlurMode = "dragnet") -> np.ndarray:
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if sigma == 0:
        return sequence.astype(np.float32, copy=True)
    if mode == "dragnet":
        # DragNet preprocessing applies the sigma across temporal and spatial axes.
        sigma_spec: float | tuple[float, float, float] = sigma
    elif mode == "spatial":
        sigma_spec = (0.0, sigma, sigma)
    else:
        raise ValueError("mode must be 'dragnet' or 'spatial'.")
    return ndimage.gaussian_filter(sequence.astype(np.float32), sigma_spec, mode="constant")


class CineSequenceDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        sequences: np.ndarray,
        *,
        sigma_blur: float = 0.2,
        blur_mode: BlurMode = "dragnet",
    ) -> None:
        self.sequences = _validate_array(np.asarray(sequences))
        self.sigma_blur = sigma_blur
        self.blur_mode = blur_mode

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        sequence = normalize_sequence(self.sequences[index])
        sequence = blur_sequence(sequence, self.sigma_blur, mode=self.blur_mode)
        return torch.from_numpy(sequence).unsqueeze(1).float()


def split_indices(
    sample_count: int,
    *,
    validation_fraction: float = 0.2,
    validation_count: int | None = None,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count < 2:
        raise ValueError("At least two sequences are required for a train/validation split.")
    if validation_count is None:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        validation_count = max(1, round(sample_count * validation_fraction))
    if validation_count <= 0 or validation_count >= sample_count:
        raise ValueError("validation_count must be between 1 and sample_count - 1.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(sample_count)
    validation = np.sort(order[:validation_count])
    training = np.sort(order[validation_count:])
    return training, validation
