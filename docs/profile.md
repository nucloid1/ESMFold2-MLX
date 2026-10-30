# Where the MLX fold's time goes

One sample, the checkpoint's 20 loops and steps (10 denoiser calls), synthetic ESMC states. Phases: one warm fold, synchronised per phase. Op classes: one instance timed at the case's size (median of 5) times the number of instances a fold runs. Written by bench/profile_fold.py.

## vegf2 (247 tokens, 2016 atoms)

| phase | seconds | share |
| --- | --- | --- |
| embed | 0.01 | 0% |
| trunk (LM shim, MSA encoder, 21 loops, coda) | 32.64 | 99% |
| diffusion sampler | 0.24 | 1% |
| confidence head | 0.16 | 0% |
| other (distogram, conversion) | 0.02 | 0% |
| total | 33.07 | 100% |

| op class | one instance (ms) | instances per fold | seconds per fold | share of fold |
| --- | --- | --- | --- | --- |
| triangle multiplication (x2 per pair block) | 18.0 | 1098 | 19.80 | 60% |
| pair transition | 11.7 | 1094 | 12.85 | 39% |
| token transformer attention (12 per denoiser call) | 0.8 | 120 | 0.10 | 0% |
| token transformer transition (12 per denoiser call) | 0.7 | 120 | 0.08 | 0% |
| atom attention block (6 per denoiser call + 3 embedder) | 1.1 | 63 | 0.07 | 0% |

## tnf2_nomsa (419 tokens, 3296 atoms)

| phase | seconds | share |
| --- | --- | --- |
| embed | 0.02 | 0% |
| trunk (LM shim, MSA encoder, 21 loops, coda) | 98.65 | 99% |
| diffusion sampler | 0.42 | 0% |
| confidence head | 0.45 | 0% |
| other (distogram, conversion) | 0.05 | 0% |
| total | 99.60 | 100% |

| op class | one instance (ms) | instances per fold | seconds per fold | share of fold |
| --- | --- | --- | --- | --- |
| triangle multiplication (x2 per pair block) | 55.1 | 1098 | 60.45 | 61% |
| pair transition | 33.3 | 1094 | 36.43 | 37% |
| token transformer attention (12 per denoiser call) | 1.1 | 120 | 0.14 | 0% |
| token transformer transition (12 per denoiser call) | 0.9 | 120 | 0.11 | 0% |
| atom attention block (6 per denoiser call + 3 embedder) | 1.5 | 63 | 0.09 | 0% |

# Triangle multiplication at N=419: where the time goes

| part | ms |
| --- | --- |
| whole op (with residual) | 27.5 |
| LayerNorm + 1024-wide projection | 9.5 |
| sigmoid routing and mask (compiled) | 3.3 |
| contraction as written (transposes + batched matmul + transpose back) | 7.1 |
| contraction matmul alone (contiguous inputs, no transpose back) | 4.4 |
| LayerNorm + output projection + output gate | 8.8 |

Contraction arithmetic: 37.7 GFLOP; the matmul alone runs at 8.48 TFLOP/s.

## Decision: no Metal kernel qualifies (Task 14, 2026-10-30)

The plan's rule: an op class gets a hand-written kernel only if it is at least 25 percent of the fold AND an ablation
shows the arithmetic itself is the cost. Triangle multiplication (about 60 percent) and the pair transition (about
38 percent) pass the first test and fail the second: their matmuls already run near this GPU's fp32 peak (the
contraction at 8.5 TFLOP/s, the 1024-wide projection at about 9.7, the transition's two projections at about 8.3;
Mol-1-MLX measured 10-11 TFLOP/s for isolated fp32 matmuls), so a kernel cannot make the arithmetic faster. Everything
else in the fold (sampler, confidence head, embedder) is about 1 percent.

What is left in a triangle multiplication is memory traffic: the transposes around the contraction (2.7 of 27.5 ms at
N=419) and the elementwise passes (sigmoid routing and mask 3.3 ms, LayerNorms and gates). Future work, not done: keep
the pair channel-major between the two triangle updates to drop the transposes, or fuse routing and transpose; the
ceiling is roughly 10 to 20 percent on the trunk.
