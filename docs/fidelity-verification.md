# Legacy fidelity verification

The refreshed implementation was compared programmatically with the supplied historical code snapshot before packaging.

For `displacement_sampling="legacy"`:

- both models contain **1,218,253 trainable parameters**;
- the ordered sequence of parameter tensor shapes is identical;
- after copying historical parameter tensors into the refreshed model and resetting the same PyTorch random seed, registration outputs were numerically identical in the tested forward pass;
- displacement outputs were numerically identical;
- all four weighted loss components and the total loss were numerically identical;
- one-frame generation was numerically identical in the tested case;
- two-frame generation was numerically identical in the tested case.

The comparison used the current PyTorch `align_corners=False` behaviour, which matches the effective default used by the historical code on modern PyTorch versions. The refresh now sets this option explicitly to remove version-dependent ambiguity.

This verification checks implementation compatibility with the supplied historical snapshot. It is **not** a reproduction of the paper's numerical results, which would additionally require the authorised research dataset, original trained weights, subject split and full experimental environment.
