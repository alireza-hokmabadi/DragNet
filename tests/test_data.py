import numpy as np
import pytest

from dragnet_cmr.data import (
    CineSequenceDataset,
    blur_sequence,
    load_sequences,
    normalize_sequence,
    split_indices,
)
from dragnet_cmr.synthetic import make_synthetic_dataset


def test_normalize_uint8():
    sequence = np.array([[[0, 255]]], dtype=np.uint8)
    output = normalize_sequence(sequence)
    assert output.dtype == np.float32
    assert output.min() == 0.0
    assert output.max() == 1.0


def test_blur_modes_keep_shape():
    sequence = np.zeros((7, 128, 128), dtype=np.float32)
    assert blur_sequence(sequence, 0.2, mode="dragnet").shape == sequence.shape
    assert blur_sequence(sequence, 0.2, mode="spatial").shape == sequence.shape


def test_load_npz_and_dataset(tmp_path):
    data = make_synthetic_dataset(2)
    path = tmp_path / "data.npz"
    np.savez_compressed(path, sequences=data)
    loaded = load_sequences(path)
    assert np.array_equal(loaded, data)
    dataset = CineSequenceDataset(loaded, sigma_blur=0)
    tensor = dataset[0]
    assert tensor.shape == (7, 1, 128, 128)
    assert tensor.dtype.is_floating_point


def test_pickle_is_rejected(tmp_path):
    path = tmp_path / "unsafe.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="Only .npy and .npz"):
        load_sequences(path)


def test_split_indices_is_reproducible():
    train_a, val_a = split_indices(10, validation_count=2, seed=3)
    train_b, val_b = split_indices(10, validation_count=2, seed=3)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert len(train_a) == 8
    assert len(val_a) == 2
