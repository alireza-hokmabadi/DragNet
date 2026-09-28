# Examples

Generate the public synthetic dataset rather than committing medical images:

```bash
dragnet-cmr synthetic synthetic_demo.npz --samples 8 --preview synthetic_demo.png
```

The generated NPZ is ignored by Git and can be recreated deterministically from the source code.
