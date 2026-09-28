# DragNet architecture

DragNet models cardiac motion using a latent-variable recurrent network coupled to a probabilistic displacement-field model and spatial transformer.

## Network structure

The implementation uses:

- single-channel 128 × 128 input images;
- a 64-dimensional latent representation;
- convolutional image features at 32 × 32 resolution;
- a two-layer ConvLSTM with hidden dimensions 32 and 16;
- a recurrent prior for the latent variable;
- posterior latent inference from the current image and recurrent state;
- a displacement posterior conditioned on latent and image features;
- spatial-transformer-based image warping.

The model contains **1,218,253 trainable parameters**.

During registration, each cardiac phase is related to the preceding phase, including the cyclic transition between the final and first phases.

## Loss

Training combines image similarity with three regularisation terms:

- latent KL divergence;
- displacement-field smoothness;
- displacement KL divergence.

The corresponding weights are:

```text
latent KL:       2e-4
smoothness:      0.03
displacement KL: 1e-4
```

## Displacement sampling

The default `dragnet` sampling mode follows the displacement sampling used by DragNet:

```text
D = mu + (0.5 * Sigma) @ epsilon
```

where `Sigma` is the predicted 2 × 2 covariance matrix.

An alternative `cholesky` mode is available for comparative experiments:

```bash
--sampling cholesky
```

## Gaussian preprocessing

The default `dragnet` preprocessing applies the Gaussian sigma across the temporal and spatial dimensions of the cine sequence.

A spatial-only option is available with:

```bash
--blur-mode spatial
```

## Reproducibility

The parameter count and core tensor shapes are covered by regression tests. The test suite also checks registration, generation, displacement sampling, spatial transformation, checkpoint handling and training.

GitHub Actions runs the package tests on Python 3.10–3.13.
