# Changelog

## 0.1.0 — public implementation refresh

- rebuilt the repository from a clean history;
- removed participant-derived examples, figures, paper PDF and model weights;
- replaced public examples with analytic synthetic phantoms;
- converted the code into an installable `dragnet-cmr` package;
- modernised PyTorch usage and device handling;
- replaced pickle data I/O with NumPy NPZ/NPY;
- fixed checkpoint epoch naming and out-of-main saving;
- added deterministic training utilities;
- added registration/generation CLI commands;
- made historical displacement sampling and blur behaviour explicit;
- added tests, coverage, type/lint configuration and CI.
