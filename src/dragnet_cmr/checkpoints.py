from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import DragNet


def checkpoint_name(epoch: int) -> str:
    if epoch < 1:
        raise ValueError("epoch numbering is one-based and must be positive.")
    return f"dragnet_epoch_{epoch:03d}.pt"


def save_checkpoint(
    path: str | Path,
    model: DragNet,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "model_config": {"displacement_sampling": model.displacement_sampling},
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    model: DragNet,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint does not use the DragNet public format.")
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    return payload
