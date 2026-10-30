# End to end: MLXFold vs torch CPU and torch MPS

Identical inputs (the esm featurizer's features, synthetic ESMC states) and identical randomness (the CPU run's recorded draws, replayed by MLX and by torch on MPS); checkpoint loops and steps, 2 samples. Coordinates: aligned RMSD in angstrom. `MPS-CPU` is torch's own device spread. Written by scripts/parity_report.py.

| case | distogram rel. err. vs CPU / MPS | bins flipped vs CPU | sample | RMSD MLX-CPU | MLX-MPS | MPS-CPU | pTM MLX / CPU / MPS | ipTM MLX / CPU / MPS | pLDDT MLX / CPU / MPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m49 | 4.83e-06 / 7.44e-07 | 0.000% | 0 | 0.0065 | 0.0047 | 0.0060 | 0.3738 / 0.3736 / 0.3738 | 0.0000 / 0.0000 / 0.0000 | 0.6337 / 0.6341 / 0.6337 |
| m49 | 4.83e-06 / 7.44e-07 | 0.000% | 1 | 0.0045 | 0.0038 | 0.0057 | 0.3635 / 0.3637 / 0.3637 | 0.0000 / 0.0000 / 0.0000 | 0.6247 / 0.6247 / 0.6247 |
| vegf1 | 7.48e-06 / 7.93e-07 | 0.000% | 0 | 0.0060 | 0.0083 | 0.0090 | 0.1944 / 0.1944 / 0.1944 | 0.0753 / 0.0755 / 0.0754 | 0.3583 / 0.3586 / 0.3584 |
| vegf1 | 7.48e-06 / 7.93e-07 | 0.000% | 1 | 0.0061 | 0.0050 | 0.0063 | 0.1882 / 0.1881 / 0.1882 | 0.0690 / 0.0689 / 0.0690 | 0.3569 / 0.3568 / 0.3568 |
| tnf60_msa | 2.05e-05 / 1.66e-06 | 0.000% | 0 | 0.0060 | 0.0046 | 0.0060 | 0.2251 / 0.2251 / 0.2251 | 0.0727 / 0.0727 / 0.0727 | 0.3921 / 0.3920 / 0.3920 |
| tnf60_msa | 2.05e-05 / 1.66e-06 | 0.000% | 1 | 0.0220 | 0.0080 | 0.0163 | 0.2076 / 0.2079 / 0.2080 | 0.0760 / 0.0761 / 0.0760 | 0.3960 / 0.3961 / 0.3960 |
| tnf60_shallow | 1.46e-05 / 2.68e-06 | 0.000% | 0 | 0.0227 | 0.0062 | 0.0180 | 0.2104 / 0.2106 / 0.2103 | 0.0698 / 0.0698 / 0.0698 | 0.3924 / 0.3925 / 0.3923 |
| tnf60_shallow | 1.46e-05 / 2.68e-06 | 0.000% | 1 | 0.0588 | 0.0089 | 0.0521 | 0.2153 / 0.2152 / 0.2154 | 0.0608 / 0.0609 / 0.0608 | 0.3844 / 0.3844 / 0.3844 |
