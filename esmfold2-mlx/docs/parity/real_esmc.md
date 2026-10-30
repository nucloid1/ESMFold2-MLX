# Real ESMC-6B end to end: MLX vs torch on MPS

Real ESMC-6B hidden states (bf16), the checkpoint's loops and steps, 2 samples, seed 0. Torch on MPS folds twice with the same seed (`torch A` vs `torch B` is torch's own repeat spread on this device); MLX consumes torch's exact MPS random stream for the seed (`--match-torch-rng`). Coordinates: aligned RMSD in angstrom. Written by scripts/real_esmc_parity.py.

| case | tokens | sample | pTM torch A / B / MLX | ipTM torch A / B / MLX | pLDDT torch A / B / MLX | RMSD MLX-A | RMSD B-A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vegf2 | 247 | 0 | 0.7391 / 0.7392 / 0.7391 | 0.6305 / 0.6307 / 0.6306 | 0.7658 / 0.7659 / 0.7659 | 0.0059 | 0.0053 |
| vegf2 | 247 | 1 | 0.7348 / 0.7346 / 0.7347 | 0.6296 / 0.6290 / 0.6296 | 0.7691 / 0.7691 / 0.7691 | 0.0161 | 0.0214 |
| tnf2_msa | 419 | 0 | 0.5776 / 0.5775 / 0.5773 | 0.3756 / 0.3754 / 0.3752 | 0.7842 / 0.7842 / 0.7841 | 0.0071 | 0.0041 |
| tnf2_msa | 419 | 1 | 0.5290 / 0.5290 / 0.5289 | 0.2993 / 0.2993 / 0.2993 | 0.7817 / 0.7817 / 0.7817 | 0.0034 | 0.0054 |
