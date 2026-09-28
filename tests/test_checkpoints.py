import torch

from dragnet_cmr.checkpoints import checkpoint_name, load_checkpoint, save_checkpoint
from dragnet_cmr.model import DragNet


def test_checkpoint_name_is_one_based():
    assert checkpoint_name(1) == "dragnet_epoch_001.pt"
    assert checkpoint_name(70) == "dragnet_epoch_070.pt"


def test_checkpoint_roundtrip(tmp_path):
    model = DragNet()
    path = save_checkpoint(tmp_path / "model.pt", model, epoch=1, metadata={"source": "synthetic"})
    restored = DragNet()
    payload = load_checkpoint(path, restored)
    assert payload["epoch"] == 1
    assert payload["metadata"]["source"] == "synthetic"
    first_a = next(model.parameters()).detach()
    first_b = next(restored.parameters()).detach()
    assert torch.equal(first_a, first_b)
