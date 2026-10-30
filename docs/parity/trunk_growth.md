# Trunk loop: MLX vs torch-CPU, relative error per loop

Synthetic ESMC states, replayed torch draws, 21 loop steps (the checkpoint's 20 loops). Written by tests/test_trunk_loop_vs_refs.py.

| case | loop | relative error |
| --- | --- | --- |
| m49 | 0 | 5.61e-07 |
| m49 | 1 | 5.09e-07 |
| m49 | 2 | 5.00e-07 |
| m49 | 5 | 4.87e-07 |
| m49 | 10 | 4.87e-07 |
| m49 | 15 | 4.87e-07 |
| m49 | 20 | 4.87e-07 |
| m49 | after readout + coda | 7.48e-07 |
| vegf1 | 0 | 7.09e-07 |
| vegf1 | 1 | 8.58e-07 |
| vegf1 | 2 | 8.00e-07 |
| vegf1 | 5 | 6.87e-07 |
| vegf1 | 10 | 8.44e-07 |
| vegf1 | 15 | 7.91e-07 |
| vegf1 | 20 | 8.11e-07 |
| vegf1 | after readout + coda | 1.05e-06 |
