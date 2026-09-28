# Implementation notes

This refresh separates the published scientific architecture from historical software issues.

## Preserved scientific structure

The following are intentionally retained from the historical implementation and paper:

- 128 × 128 single-channel input;
- latent dimension 64;
- image feature pathway producing 32 × 32 feature maps;
- recurrent ConvLSTM stack with hidden dimensions 32 then 16;
- learned recurrent prior for the latent variable;
- posterior latent inference conditioned on current-frame features and recurrent state;
- displacement posterior conditioned on sampled latent features and moving-image features;
- mean plus low-rank covariance parameterisation for the displacement field;
- spatial-transformer warping;
- cyclic registration where the final frame is used as moving image for phase 0;
- loss weights `2e-4` for latent KL, `0.03` for smoothness and `1e-4` for displacement KL.

The parameter count remains 1,218,253, which is regression-tested.

## Engineering fixes

The refresh fixes or removes several historical software problems without changing the intended network topology:

- renamed the model class from `dragnet` to conventional `DragNet`;
- removed deprecated `torch.autograd.Variable` usage;
- removed hard-coded local Windows dataset paths;
- removed pickle-based public data loading;
- removed bundled participant-derived samples and figures;
- removed dependency on a bundled colour-wheel image;
- made `align_corners=False` explicit and consistent in the spatial transformer;
- fixed output allocation to use both height and width rather than width twice;
- moved checkpoint saving inside the training workflow;
- made epoch filenames one-based (`001` ... `070`) to remove the historical 69/70 mismatch;
- centralised data validation and intensity normalisation;
- made device selection explicit while inferring tensor device inside model operations;
- added deterministic split/seed controls;
- added tests, CI and package metadata.

## Historical displacement sampling

The original code constructs a covariance matrix

`Sigma = exp(log_var) * I + exp(log_v) exp(log_v)^T`

but then samples using approximately `mu + 0.5 * Sigma * epsilon`. For a Gaussian with covariance `Sigma`, conventional sampling uses a matrix square root `L` where `L L^T = Sigma`, for example a Cholesky factor.

Because silently changing the sampler would break implementation fidelity, `DragNet(displacement_sampling="legacy")` remains the default. `displacement_sampling="cholesky"` is available for new experiments and should be treated as a modified method until separately validated.

## Historical Gaussian filtering

The old dataset code called `scipy.ndimage.gaussian_filter(sequence, sigma)` with a scalar sigma. SciPy therefore applies that sigma across every sequence dimension, including time. This is retained as `blur_mode="legacy"`. `blur_mode="spatial"` uses `(0, sigma, sigma)` and is provided for new experiments.

## What has not been claimed

This refresh does not claim to reproduce the published numerical results without the original authorised dataset, split, trained weights and software/hardware environment. It preserves the implementation structure and makes known software choices explicit.
