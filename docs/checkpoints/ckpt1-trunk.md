# Checkpoint 1: the full trunk, MLX vs torch/MPS

Trunk = LM shim + MSA encoder + 21 loop steps of 48 pair blocks (the checkpoint's 20 loops) + readout + coda, on each case's real features with synthetic ESMC states (bf16, as ESMC hands them over; timing is value-independent). Torch: trunk-only copy loaded as the rescore worker loads it, eval, no grad, default chunk size, segments bracketed by `torch.mps.synchronize()` inside the real forward. MLX: a clean process, no torch. 2 warmups, 5 timed runs each; `first` is the first (cold) run. Memory: MLX re-run under a hard cap equal to torch's sampled peak driver allocation for the whole forward.

| case | tokens | MSA | torch median (min-max) | MLX median (min-max) | speedup | torch first | MLX first | torch peak | MLX under that cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vegf2 | 247 | none (single sequence) | 59.71 s (58.63-65.85) | 34.53 s (34.46-34.68) | 1.73x | 59.69 s | 34.74 s | 4.4 GB | survived (4.1 GiB; MLX peak 3.2 GiB) |
| il17 | 369 | none (single sequence) | 130.59 s (130.41-131.89) | 75.07 s (74.76-75.19) | 1.74x | 132.70 s | 74.56 s | 7.1 GB | survived (6.6 GiB; MLX peak 5.7 GiB) |
| tnf2_nomsa | 419 | none (single sequence) | 172.99 s (172.89-173.53) | 98.81 s (98.35-99.13) | 1.75x | 172.81 s | 96.90 s | 8.5 GB | survived (7.9 GiB; MLX peak 6.9 GiB) |
| tnf2_msa | 419 | real, 2225 rows on both TNF chains | 178.19 s (178.17-178.80) | 113.28 s (113.05-113.67) | 1.57x | 178.39 s | 112.14 s | 11.7 GB | survived (10.9 GiB; MLX peak 8.9 GiB) |
| il17_scfv | 499 | none (single sequence) | 251.77 s (251.60-252.25) | 141.28 s (140.70-141.56) | 1.78x | 251.50 s | 140.62 s | 11.0 GB | survived (10.3 GiB; MLX peak 9.2 GiB) |

Parity of the same trunk: tests/test_trunk_loop_vs_refs.py (torch-CPU references, 21 loops, 43 to 145 tokens, with and without MSA): 5e-7 to 2.3e-6 per loop, no growth with loop count (docs/parity/trunk_growth.md).

**PASS**
