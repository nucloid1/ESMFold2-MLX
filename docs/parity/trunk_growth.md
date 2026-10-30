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
| m49_q | 0 | 5.61e-07 |
| m49_q | 1 | 5.09e-07 |
| m49_q | 2 | 5.00e-07 |
| m49_q | 5 | 4.87e-07 |
| m49_q | 10 | 4.87e-07 |
| m49_q | 15 | 4.87e-07 |
| m49_q | 20 | 4.87e-07 |
| m49_q | after readout + coda | 7.48e-07 |
| tnf60_msa | 0 | 1.31e-06 |
| tnf60_msa | 1 | 2.29e-06 |
| tnf60_msa | 2 | 1.82e-06 |
| tnf60_msa | 5 | 1.04e-06 |
| tnf60_msa | 10 | 7.38e-07 |
| tnf60_msa | 15 | 7.22e-07 |
| tnf60_msa | 20 | 7.21e-07 |
| tnf60_msa | after readout + coda | 9.76e-07 |
| tnf60_shallow | 0 | 1.08e-06 |
| tnf60_shallow | 1 | 1.38e-06 |
| tnf60_shallow | 2 | 1.19e-06 |
| tnf60_shallow | 5 | 1.51e-06 |
| tnf60_shallow | 10 | 8.83e-07 |
| tnf60_shallow | 15 | 7.94e-07 |
| tnf60_shallow | 20 | 8.77e-07 |
| tnf60_shallow | after readout + coda | 1.05e-06 |
