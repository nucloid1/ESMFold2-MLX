# Checkpoint 0: one pair-trunk block, MLX vs torch/MPS

fp32, block 0 with real weights, random pair input (timing is value-independent), 2 warmups and 7 timed runs per cell, each cell in its own process. Torch is the trunk-only copy loaded as the rescore worker loads it, eval, no grad. `torch (64)` is production (checkpoint default chunk size 64); `torch (none)` calls `set_chunk_size(None)`.

| N | MLX | torch (64, production) | torch (none) | MLX speedup vs 64 | MLX speedup vs none |
| --- | --- | --- | --- | --- | --- |
| 250 | 30 ms (min 30, max 30) | 48 ms (min 48, max 49) | 49 ms (min 49, max 49) | 1.61x | 1.64x |
| 400 | 79 ms (min 79, max 79) | 132 ms (min 131, max 132) | 127 ms (min 126, max 127) | 1.67x | 1.60x |

Per-op medians (ms), each op alone on the same input:

| N | impl | tri_mul_out | tri_mul_in | pair_transition |
| --- | --- | --- | --- | --- |
| 250 | MLX | 9 | 9 | 12 |
| 250 | torch (64) | 17 | 17 | 14 |
| 250 | torch (none) | 15 | 15 | 19 |
| 400 | MLX | 25 | 25 | 30 |
| 400 | torch (64) | 47 | 47 | 36 |
| 400 | torch (none) | 39 | 39 | 49 |

MLX beats the production call by at least 10 percent at every size: **yes**. MLX also beats torch with chunking off by at least 10 percent at every size: **yes**.

**PASS**

Parity of the same block against torch-CPU captures (real forward, first loop, no ESMC states; `tests/test_pair_block_vs_refs.py`, 27 tests, gate 1e-5): relative error 4e-8 to 1.8e-7 for blocks 0, 23 and 47 at 43 and 145 tokens.

Run on 2026-10-29 with the disk at 94% under the opt-in `ESMFOLD2_MLX_MIN_FREE_GB=50`, which Ethan approved for this checkpoint's light runs.
