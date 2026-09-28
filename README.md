# DragNet

**DragNet** is a probabilistic deep-learning framework for spatio-temporal deformable registration of cine cardiac MR sequences and for cardiac motion sequence generation from one or two reference frames.

This repository provides a PyTorch implementation of the method described in:

> Zakeri A, Hokmabadi A, Bi N, Wijesinghe I, Nix MG, Petersen SE, Frangi AF, Taylor ZA, Gooya A. **DragNet: Learning-based deformable registration for realistic cardiac MR sequence generation from a single frame.** *Medical Image Analysis*. 2023;83:102678. https://doi.org/10.1016/j.media.2022.102678

## Overview

DragNet combines an image encoder, a variational latent model, recurrent ConvLSTM features, a probabilistic displacement-field model, and a spatial transformer.

The implementation supports:

- deformable registration of complete cine sequences;
- cardiac motion sequence generation from one reference frame;
- sequence generation conditioned on two reference frames;
- probabilistic displacement-field estimation;
- training from 128 × 128 single-channel cine sequences;
- command-line and Python interfaces.

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/alireza-hokmabadi/DragNet.git
cd DragNet
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

DragNet requires Python 3.10 or later and uses PyTorch, NumPy, SciPy and Matplotlib.

## Data format

Input sequences can be stored as `.npy` or `.npz` files.

For NPZ files, the default array key is `sequences`.

```text
shape: (N, T, 128, 128)
   or: (N, T, 1, 128, 128)
```

Integer arrays should contain values in `[0, 255]`. Floating-point arrays should already be scaled to `[0, 1]`.

For example:

```python
import numpy as np

np.savez_compressed("data.npz", sequences=sequences)
```

The dataset and pretrained model weights used in the published study are not included in this repository.

## Training

```bash
dragnet-cmr train data.npz \
  --output-dir checkpoints \
  --epochs 70 \
  --batch-size 10 \
  --learning-rate 0.001 \
  --device auto
```

Training writes numbered checkpoints and a final checkpoint:

```text
checkpoints/
├── dragnet_epoch_001.pt
├── ...
├── dragnet_epoch_070.pt
└── dragnet_final.pt
```

## Registration

Run registration using a trained checkpoint:

```bash
dragnet-cmr register data.npz checkpoints/dragnet_final.pt \
  --index 0 \
  --output outputs/registration.npz \
  --figure outputs/registration.png
```

The output NPZ contains:

- `registered`: registered cine sequence, shape `(T, 128, 128)`
- `displacement`: displacement fields, shape `(T, 2, 128, 128)`

## Sequence generation

Generate a sequence from one reference frame:

```bash
dragnet-cmr generate data.npz checkpoints/dragnet_final.pt \
  --mode one \
  --frames 7 \
  --output outputs/generated.npz
```

Generation can also be conditioned on the first two frames:

```bash
dragnet-cmr generate data.npz checkpoints/dragnet_final.pt \
  --mode two \
  --frames 7 \
  --output outputs/generated.npz
```

The output contains the generated image sequence and corresponding displacement fields.

## Python API

```python
import torch

from dragnet_cmr import DragNet

model = DragNet()

sequence = torch.rand(1, 7, 1, 128, 128)
output = model(sequence)

print(output.registered.shape)
print(output.displacement.shape)
print(output.losses.total)
```

## Architecture

The network uses a 64-dimensional latent representation together with a two-layer ConvLSTM and a probabilistic displacement-field model. The published implementation operates on 128 × 128 single-channel images.

Further details are available in [docs/architecture.md](docs/architecture.md).

## Testing

Run the test suite with:

```bash
python -m pytest
```

Development checks:

```bash
python -m ruff check .
python -m mypy src/dragnet_cmr
python -m pytest --cov=dragnet_cmr --cov-report=term-missing --cov-fail-under=80
python -m build
```

Continuous integration tests Python 3.10, 3.11, 3.12 and 3.13.

## Citation

If you use DragNet, please cite:

```bibtex
@article{zakeri2023dragnet,
  title   = {DragNet: Learning-based deformable registration for realistic cardiac MR sequence generation from a single frame},
  author  = {Zakeri, Arezoo and Hokmabadi, Alireza and Bi, Ning and Wijesinghe, Isuru and Nix, Michael G. and Petersen, Steffen E. and Frangi, Alejandro F. and Taylor, Zeike A. and Gooya, Ali},
  journal = {Medical Image Analysis},
  volume  = {83},
  pages   = {102678},
  year    = {2023},
  doi     = {10.1016/j.media.2022.102678}
}
```
