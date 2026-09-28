from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint
from .data import CineSequenceDataset, load_sequences
from .model import DragNet
from .synthetic import make_synthetic_sequence, write_synthetic_npz
from .training import TrainingConfig, resolve_device, set_reproducible_seed, train
from .version import __version__
from .visualization import save_registration_figure, save_sequence_preview


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dragnet-cmr",
        description="DragNet for cine CMR registration and sequence generation.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="Create a synthetic dataset for testing.")
    synthetic.add_argument("output", type=Path)
    synthetic.add_argument("--samples", type=int, default=8)
    synthetic.add_argument("--frames", type=int, default=7)
    synthetic.add_argument("--seed", type=int, default=1234)
    synthetic.add_argument("--preview", type=Path, default=None)

    smoke = subparsers.add_parser(
        "smoke",
        help="Run an untrained architecture smoke test on synthetic data.",
    )
    smoke.add_argument("--sampling", choices=("dragnet", "cholesky"), default="dragnet")
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--seed", type=int, default=1234)

    train_parser = subparsers.add_parser(
        "train",
        help="Train DragNet on an NPZ/NPY sequence dataset.",
    )
    train_parser.add_argument("data", type=Path)
    train_parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    train_parser.add_argument("--epochs", type=int, default=70)
    train_parser.add_argument("--batch-size", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--sigma-blur", type=float, default=0.2)
    train_parser.add_argument("--blur-mode", choices=("dragnet", "spatial"), default="dragnet")
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--validation-count", type=int, default=None)
    train_parser.add_argument("--sampling", choices=("dragnet", "cholesky"), default="dragnet")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=1234)

    register = subparsers.add_parser(
        "register",
        help="Run registration using user-supplied weights.",
    )
    register.add_argument("data", type=Path)
    register.add_argument("checkpoint", type=Path)
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--figure", type=Path, default=None)
    register.add_argument("--index", type=int, default=0)
    register.add_argument("--sigma-blur", type=float, default=0.2)
    register.add_argument("--blur-mode", choices=("dragnet", "spatial"), default="dragnet")
    register.add_argument("--sampling", choices=("dragnet", "cholesky"), default="dragnet")
    register.add_argument("--device", default="auto")
    register.add_argument("--seed", type=int, default=1234)

    generate = subparsers.add_parser(
        "generate",
        help="Generate a sequence using user-supplied weights.",
    )
    generate.add_argument("data", type=Path)
    generate.add_argument("checkpoint", type=Path)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--mode", choices=("one", "two"), default="one")
    generate.add_argument("--index", type=int, default=0)
    generate.add_argument("--frames", type=int, default=7)
    generate.add_argument("--sigma-blur", type=float, default=0.2)
    generate.add_argument("--blur-mode", choices=("dragnet", "spatial"), default="dragnet")
    generate.add_argument("--sampling", choices=("dragnet", "cholesky"), default="dragnet")
    generate.add_argument("--device", default="auto")
    generate.add_argument("--seed", type=int, default=1234)
    return parser


def _load_one_tensor(args: argparse.Namespace) -> tuple[np.ndarray, torch.Tensor]:
    sequences = load_sequences(args.data)
    if args.index < 0 or args.index >= len(sequences):
        raise IndexError("--index is outside the dataset.")
    dataset = CineSequenceDataset(
        sequences,
        sigma_blur=args.sigma_blur,
        blur_mode=args.blur_mode,
    )
    tensor = dataset[args.index].unsqueeze(0)
    return sequences[args.index], tensor


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "synthetic":
        target = write_synthetic_npz(
            args.output,
            sample_count=args.samples,
            frame_count=args.frames,
            seed=args.seed,
        )
        preview = args.preview or target.with_suffix(".png")
        save_sequence_preview(
            make_synthetic_sequence(frame_count=args.frames, seed=args.seed),
            preview,
        )
        print(f"Synthetic dataset: {target}")
        print(f"Preview: {preview}")
        return 0

    if args.command == "smoke":
        set_reproducible_seed(args.seed)
        device = resolve_device(args.device)
        model = DragNet(displacement_sampling=args.sampling).to(device).eval()
        sequence = make_synthetic_sequence(seed=args.seed).astype(np.float32) / 255.0
        tensor = torch.from_numpy(sequence).unsqueeze(0).unsqueeze(2).to(device)
        with torch.no_grad():
            output = model(tensor)
        print(f"device: {device}")
        print(f"parameters: {model.parameter_count():,}")
        print(f"registered shape: {tuple(output.registered.shape)}")
        print(f"displacement shape: {tuple(output.displacement.shape)}")
        print("Smoke test completed.")
        return 0

    if args.command == "train":
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            sigma_blur=args.sigma_blur,
            blur_mode=args.blur_mode,
            validation_fraction=args.validation_fraction,
            validation_count=args.validation_count,
            seed=args.seed,
            displacement_sampling=args.sampling,
        )
        final = train(args.data, args.output_dir, config=config, device=args.device)
        print(f"Final checkpoint: {final}")
        return 0

    set_reproducible_seed(args.seed)
    _, tensor = _load_one_tensor(args)
    device = resolve_device(args.device)
    model = DragNet(displacement_sampling=args.sampling).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    tensor = tensor.to(device)

    if args.command == "register":
        with torch.no_grad():
            output = model(tensor)
        registered = output.registered[0, :, 0].cpu().numpy()
        displacement = output.displacement[0].cpu().numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, registered=registered, displacement=displacement)
        if args.figure is not None:
            target_sequence = tensor[0, :, 0].cpu().numpy()
            save_registration_figure(
                target_sequence,
                registered,
                displacement,
                args.figure,
            )
        print(f"Output: {args.output}")
        return 0

    if args.command == "generate":
        first = tensor[:, 0]
        with torch.no_grad():
            if args.mode == "one":
                output = model.generate_from_one_frame(first, frame_count=args.frames)
            else:
                output = model.generate_from_two_frames(
                    first,
                    tensor[:, 1],
                    frame_count=args.frames,
                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            generated=output.generated[0, :, 0].cpu().numpy(),
            displacement=output.displacement[0].cpu().numpy(),
        )
        print(f"Output: {args.output}")
        return 0

    raise RuntimeError("Unhandled command.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
