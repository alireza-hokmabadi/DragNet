# Data and weights

## What is deliberately not distributed

This public-ready repository does not contain:

- UK Biobank participant data;
- CMR frames or screenshots copied or derived from UK Biobank;
- the historical `test_sample` file;
- model weights trained on UK Biobank;
- the paper PDF or figures containing participant images;
- patient identifiers, manifests or institutional file paths.

The synthetic examples are generated from analytic ellipses by `dragnet_cmr.synthetic` and do not originate from any patient or imaging dataset.

## Why weights are not included

A model checkpoint trained from a restricted dataset is a derived research artefact whose redistribution should be assessed under the applicable data-access agreement and institutional governance. This repository therefore requires users to provide checkpoints they are authorised to use.

## Expected user data

The code accepts safe NumPy `.npz` or `.npy` files. Pickle loading has been removed from the public implementation because it is both an unnecessary data-format dependency and unsafe for untrusted files.

The default NPZ key is `sequences`, with shape `(N,T,128,128)` or `(N,T,1,128,128)`.

## Publication-history warning

If an older Git repository ever contained restricted data, deleting the files from the latest commit does not remove them from Git history, branches, tags or releases. For that reason, this cleaned version is intended to start from a **fresh Git history** rather than republish the historical repository unchanged.
