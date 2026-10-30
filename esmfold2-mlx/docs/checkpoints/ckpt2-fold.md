# Checkpoint 2: the whole fold, MLX vs torch/MPS

Everything after ESMC (embed, trunk with MSA encoder and 20 loops, distogram, diffusion sampler at the checkpoint's steps, confidence head) on each case's real features with synthetic ESMC states (bf16). Torch: trunk-only copy loaded as the rescore worker loads it, eval, no grad, default chunk size, the forward bracketed by `torch.mps.synchronize()`. MLX: MLXFold in a clean process. 1 warmup, 3 timed runs (Checkpoint 1 runs varied by under 1 percent); `first` is the cold run. Memory: MLX re-run under a hard cap equal to torch's sampled peak driver allocation.

| case | tokens | MSA | samples | torch median (min-max) | MLX median (min-max) | speedup | torch first | MLX first | torch peak | MLX under that cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vegf2 | 247 | none | 1 | 60.1 s (57.9-61.5) | 35.5 s (35.4-35.6) | 1.69x | 57.4 s | 35.5 s | 4.4 GB | survived (4.1 GiB; MLX peak 3.6 GiB) |
| vegf2 | 247 | none | 3 | 60.4 s (60.2-60.5) | 36.1 s (35.9-36.2) | 1.67x | 61.7 s | 35.6 s | 8.8 GB | survived (8.2 GiB; MLX peak 6.5 GiB) |
| il17 | 369 | none | 1 | 134.1 s (132.9-135.4) | 76.0 s (75.9-76.2) | 1.76x | 138.1 s | 76.0 s | 6.6 GB | survived (6.1 GiB; MLX peak 5.5 GiB) |
| il17 | 369 | none | 3 | 134.5 s (134.4-135.0) | 76.8 s (76.8-77.0) | 1.75x | 135.0 s | 76.5 s | 14.0 GB | survived (13.1 GiB; MLX peak 12.5 GiB) |
| tnf2_nomsa | 419 | none | 1 | 175.6 s (175.6-175.9) | 99.2 s (98.9-99.7) | 1.77x | 175.9 s | 98.3 s | 8.4 GB | survived (7.9 GiB; MLX peak 6.9 GiB) |
| tnf2_nomsa | 419 | none | 3 | 179.0 s (179.0-179.1) | 100.8 s (100.2-101.0) | 1.78x | 179.1 s | 99.9 s | 16.5 GB | KILLED (15.4 GiB) |
| tnf2_msa | 419 | real, 2225 rows | 1 | 180.4 s (180.1-180.7) | 114.1 s (114.0-114.2) | 1.58x | 180.2 s | 113.9 s | 11.7 GB | survived (10.9 GiB; MLX peak 8.9 GiB) |
| tnf2_msa | 419 | real, 2225 rows | 3 | 183.5 s (183.5-183.7) | 115.1 s (115.1-115.6) | 1.59x | 183.4 s | 114.9 s | 18.3 GB | survived (17.0 GiB; MLX peak 15.3 GiB) |
| il17_scfv | 499 | none | 1 | 254.6 s (254.6-254.6) | 141.7 s (141.1-142.1) | 1.80x | 254.6 s | 140.7 s | 11.0 GB | survived (10.3 GiB; MLX peak 9.2 GiB) |
| il17_scfv | 499 | none | 3 | 259.7 s (259.4-259.9) | 143.3 s (141.0-143.7) | 1.81x | 259.6 s | 143.4 s | 23.3 GB | KILLED (21.7 GiB) |

Parity of the same fold: docs/parity/end_to_end.md.

**FAIL** (first run, on memory only: MLX was faster than torch in all 10 configurations)

## Fix and re-run (2026-10-30)

The two kills were the confidence head scoring all samples' [S, L, L, 256] pair stack at once: 22.75 GiB for 3 samples at 499 tokens, against 8.9 GiB for the whole trunk; torch chunks those pair ops. Samples are independent there, so the head now scores them one at a time (identical outputs; tests/test_heads_vs_refs.py and test_fold_vs_refs.py unchanged and green), and its 3-sample peak at 499 tokens is 9.2 GiB. The 3-sample configurations were re-run (MLX: 1 warmup and 3 timed runs, then once under the same cap as above; torch numbers unchanged):

| case | tokens | samples | torch median (first run above) | MLX median (min-max) | speedup | torch peak | MLX under that cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vegf2 | 247 | 3 | 60.4 s | 33.6 s (33.3-33.7) | 1.80x | 8.8 GB | survived (8.2 GiB; MLX peak 3.8 GiB) |
| il17 | 369 | 3 | 134.5 s | 76.6 s (76.5-76.7) | 1.76x | 14.0 GB | survived (13.1 GiB; MLX peak 6.1 GiB) |
| tnf2_nomsa | 419 | 3 | 179.0 s | 100.2 s (100.1-100.6) | 1.79x | 16.5 GB | survived (15.4 GiB; MLX peak 7.3 GiB) |
| tnf2_msa | 419 | 3 | 183.5 s | 114.8 s (114.5-115.0) | 1.60x | 18.3 GB | survived (17.0 GiB; MLX peak 8.9 GiB) |
| il17_scfv | 499 | 3 | 259.7 s | 143.3 s (143.1-143.6) | 1.81x | 23.3 GB | survived (21.7 GiB; MLX peak 9.7 GiB) |

1-sample configurations are unaffected (the per-sample path only applies with more than one sample).

Speed: MLX at least 10 percent faster than torch in all 10 configurations (1.58-1.81x). Memory: MLX survives under torch's peak in all 10, at roughly half of it with 3 samples.

**PASS** (after the fix)
