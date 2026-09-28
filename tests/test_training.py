import torch

from dragnet_cmr.training import resolve_device, set_reproducible_seed


def test_resolve_cpu_device():
    assert resolve_device("cpu").type == "cpu"


def test_seed_is_reproducible():
    set_reproducible_seed(123)
    first = torch.rand(3)
    set_reproducible_seed(123)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_one_epoch_training_writes_numbered_and_final_checkpoints(tmp_path):
    from dragnet_cmr.synthetic import write_synthetic_npz
    from dragnet_cmr.training import TrainingConfig, train

    data = write_synthetic_npz(tmp_path / "train.npz", sample_count=2, frame_count=2)
    final = train(
        data,
        tmp_path / "checkpoints",
        config=TrainingConfig(
            epochs=1,
            batch_size=1,
            sigma_blur=0,
            validation_count=1,
            seed=5,
        ),
        device="cpu",
    )
    assert final.name == "dragnet_final.pt"
    assert final.exists()
    assert (tmp_path / "checkpoints" / "dragnet_epoch_001.pt").exists()
