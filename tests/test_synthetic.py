import numpy as np

from dragnet_cmr.synthetic import (
    make_synthetic_dataset,
    make_synthetic_sequence,
    write_synthetic_npz,
)


def test_synthetic_sequence_is_deterministic_and_uint8():
    first = make_synthetic_sequence(seed=7)
    second = make_synthetic_sequence(seed=7)
    assert first.shape == (7, 128, 128)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert first.max() <= 255


def test_synthetic_dataset_and_write(tmp_path):
    dataset = make_synthetic_dataset(3, seed=10)
    assert dataset.shape == (3, 7, 128, 128)
    path = write_synthetic_npz(tmp_path / "synthetic.npz", sample_count=3, seed=10)
    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(archive["sequences"], dataset)
