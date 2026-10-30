# Atom attention: torch's own bf16 casts, and which torch MLX matches

Torch rounds the atom RoPE tables to bf16 (`build_3d_rope`) and, on the non-flash path that CPU and MPS take, runs the
sliding-window atom attention on bf16 q, k, v (`SWA3DRoPEAttention`). MLX mirrors both casts (q, k, v rounded to bf16,
fp32 accumulation, output rounded to bf16). Measured on the inputs embedder's atom encoding (relative Frobenius error,
`tests/test_embed_vs_refs.py`, 2026-10-29):

| case | torch MPS vs torch CPU | MLX vs torch CPU | MLX vs torch MPS |
| --- | --- | --- | --- |
| m49 (43 tokens) | 1.31e-04 | 1.31e-04 | 9.77e-07 |
| vegf1 (145) | 7.47e-05 | 7.47e-05 | 7.60e-07 |
| tnf60_msa (103) | 8.52e-05 | 8.53e-05 | 6.46e-06 |

Torch's CPU bf16 kernel is the outlier; MLX agrees with torch on MPS, the production device, to about 1e-6. Stages that
contain this attention are therefore gated against MPS references at 1e-5 (and against CPU at twice torch's own CPU/MPS
spread).
