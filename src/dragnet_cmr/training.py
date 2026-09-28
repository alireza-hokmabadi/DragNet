from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .checkpoints import checkpoint_name, save_checkpoint
from .data import BlurMode, CineSequenceDataset, load_sequences, split_indices
from .model import DisplacementSampling, DragNet


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 70
    batch_size: int = 10
    learning_rate: float = 1e-3
    sigma_blur: float = 0.2
    blur_mode: BlurMode = "dragnet"
    validation_fraction: float = 0.2
    validation_count: int | None = None
    seed: int = 1234
    displacement_sampling: DisplacementSampling = "dragnet"


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def train(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    config: TrainingConfig | None = None,
    device: str = "auto",
) -> Path:
    if config is None:
        config = TrainingConfig()

    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive.")

    set_reproducible_seed(config.seed)
    torch_device = resolve_device(device)
    sequences = load_sequences(data_path)
    train_indices, val_indices = split_indices(
        len(sequences),
        validation_fraction=config.validation_fraction,
        validation_count=config.validation_count,
        seed=config.seed,
    )
    dataset = CineSequenceDataset(
        sequences,
        sigma_blur=config.sigma_blur,
        blur_mode=config.blur_mode,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = DragNet(displacement_sampling=config.displacement_sampling).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        model.train()
        train_losses: list[float] = []
        for sequence in train_loader:
            sequence = sequence.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            output = model(sequence)
            output.losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_losses.append(float(output.losses.total.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for sequence in val_loader:
                sequence = sequence.to(torch_device)
                output = model(sequence)
                val_losses.append(float(output.losses.total.detach().cpu()))

        print(
            f"epoch={epoch:03d} "
            f"train_total={_mean(train_losses):.6f} "
            f"val_total={_mean(val_losses):.6f}"
        )
        save_checkpoint(
            destination / checkpoint_name(epoch),
            model,
            epoch=epoch,
            optimizer=optimizer,
            metadata={
                "train_samples": int(len(train_indices)),
                "validation_samples": int(len(val_indices)),
                "seed": config.seed,
                "blur_mode": config.blur_mode,
                "sigma_blur": config.sigma_blur,
            },
        )

    final_path = destination / "dragnet_final.pt"
    save_checkpoint(
        final_path,
        model,
        epoch=config.epochs,
        optimizer=optimizer,
        metadata={"seed": config.seed},
    )
    return final_path
