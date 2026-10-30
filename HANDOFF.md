# ESMFold2 MLX port: handoff

Last updated 2026-10-30. Machine: M3 Max, 48 GB, macOS; env `esmfold2-binder-design` (MLX 0.32.2, torch 2.14.0, esm 3.4.1).
Claims are tagged VERIFIED (a command ran and its output was read), INFERRED (arithmetic or code reading, not measured) or
NOT DONE. Spec: `docs/superpowers/specs/2026-10-29-esmfold2-mlx-design.md`; plan: `docs/superpowers/plans/2026-10-29-esmfold2-mlx.md`.

> **Status: done and faster.** Everything ESMFold2 does after ESMC-6B (inputs embedder, LM shim, MSA encoder, the looped
> 48-block pair trunk, distogram head, diffusion sampler, confidence head) runs in MLX in a persistent worker process;
> ESMC-6B stays in torch (bf16). The whole fold is **1.58-1.81x faster** than torch on MPS called the way the
> BindCraft-Hunter rescore worker calls it, at every size from 247 to 499 tokens, with and without a real MSA, at 1 and 3
> samples, with **lower peak memory** (about half of torch's with 3 samples). On real ESMC-6B states MLX matches torch on
> MPS to pTM 3e-4, ipTM 4e-4, pLDDT 1e-4 and 0.003-0.016 A in coordinates: of the same order as torch's own same-seed
> repeat spread on MPS (0.004-0.021 A; on one of the four samples MLX sits 2-3x outside it, on the others inside).
> Use it: `bin/esmfold2-mlx pred -i job.json -o out/` (see "Using it").

## Decisions made by the user (Ethan)

- ESMC-6B stays in torch; everything after it in MLX/Metal. Standalone, with its own CLI; not wired into BindCraft-Hunter yet.
- Goal: faster than the torch version, outputs as close as possible. **If it doesn't beat torch, stop** (checkpoints at one
  block, the trunk, the whole fold; all three beat torch).
- MSA support is required. fp32 (see the bf16 note below for the two casts torch itself makes).
- One GPU run at a time (enforced by `gpu_lock()`). Inline execution, not subagents.
- The disk was 98% then 93-94% full; Ethan approved running at that level with an opt-in floor of 50 GB free
  (`ESMFOLD2_MLX_MIN_FREE_GB=50`); the 90% default in `scripts/preflight.py` is unchanged.

## Results (all VERIFIED)

### Speed and memory (`docs/checkpoints/`)

| checkpoint | what | result |
| --- | --- | --- |
| 0 | one pair block, N=250 / 400 | 1.61x / 1.67x vs production torch; 1.64x / 1.60x vs torch with chunking off |
| 1 | the trunk (LM shim, MSA encoder, 21 loops x 48 blocks, coda) | 1.57-1.78x at 247-499 tokens; MLX peak below torch's everywhere |
| 2 | the whole fold, 1 and 3 samples | 1.58-1.81x in all 10 configurations; after the confidence-head fix, MLX peak 3.8-9.7 GiB vs torch 8.8-23.3 GB with 3 samples |

Whole fold, 1 sample (Checkpoint 2): vegf2 247 tokens 60.1 s torch vs 35.5 s MLX; il17 369: 134.1 vs 76.0; tnf2 419: 175.6 vs
99.2; tnf2 with a real 2225-row MSA: 180.4 vs 114.1; il17_scfv 499: 254.6 vs 141.7. MLX runs repeat to about 1%.
The CLI adds about 6 s per run (CCD dictionary, featurizing, worker start, decode, writing), not per fold: two 247-token jobs
took 72.2 s wall for 66.0 s of folding. ESMC-6B is not in these numbers (it runs once per distinct complex and is cached).

### Closeness to torch (`docs/parity/`)

| stage | vs torch CPU | vs torch MPS |
| --- | --- | --- |
| pair block, trunk loop (21 steps), MSA encoder, heads | 4e-8 to 2.3e-6 relative, no growth with loop count | |
| inputs embedder (bf16 atom attention) | ~1e-4 (= torch's own CPU/MPS spread) | 6e-7 to 6.5e-6 |
| denoiser call | within 2x torch's CPU/MPS spread | closer to MPS than torch CPU is |
| end to end, replayed draws (`end_to_end.md`) | distogram 5e-6 to 2e-5, 0 bins flipped | distogram 7e-7 to 3e-6; coords 0.004-0.009 A; pTM/ipTM/pLDDT to 3-4 decimals |
| end to end, real ESMC-6B (`real_esmc.md`) | | pTM 3e-4, ipTM 4e-4, pLDDT 1e-4, coords 0.003-0.016 A; torch vs itself 0.004-0.021 A (same order; tnf2_msa sample 0 is 2-3x torch's own spread) |

The same seed gives a bit-identical MLX fold (`MLXDraws`); `--match-torch-rng` reproduces torch's own random stream for a
seed bit for bit (verified against draws recorded from real torch runs) so an MLX fold can be compared with torch's.

## Findings that changed decisions

1. **Every fold runs the MSA encoder.** The esm featurizer always emits an MSA (query-only in single-sequence mode) and the
   encoder's output *replaces* `z_init` as the loop injection (`msa_encoder.overwrite`). There is no "trunk without MSA".
   With depth <= 1024 its inputs are identical every loop, so MLX computes it once; torch recomputes it 21 times (exact).
2. **Torch itself runs the atom attention in bf16** on CPU and MPS (it casts q/k/v and the RoPE tables). MLX mirrors those
   casts; torch's CPU bf16 kernel disagrees with its MPS one by ~1e-4 and MLX tracks MPS to ~1e-6.
3. **esm's sigma cap shortens the schedule**: 14 configured sampling steps become 10 denoiser calls (its docstring says it
   inflates the schedule; the code filters it). Mirrored.
4. **The trunk is 99% of the fold** (`docs/profile.md`); triangle multiplication ~60%, pair transition ~38%. Their matmuls run
   at 8.3-9.7 TFLOP/s against ~10-11 peak, so **no hand-written Metal kernel qualifies** (the plan's rule); the remaining
   headroom is layout copies and elementwise passes (future work, ~10-20% ceiling on the trunk, INFERRED).
5. **Atomic scatter-adds made a fixed seed fold differently** (2.4e-4 relative); replaced by one-hot-matmul segment sums.
6. **The 3-sample confidence head held every sample's pair stack** (22.75 GiB at 499 tokens), above torch's chunked peak;
   it now scores samples one at a time (identical outputs, 9.2 GiB). This is why Checkpoint 2's first verdict was FAIL.
7. Capture hygiene: on CPU `.cpu().numpy()` shares memory, and torch's in-place `pair.add_` overwrote a captured array;
   every capture now copies.

## Using it

```
bin/esmfold2-mlx pred -i job.json -o out/ [--samples N] [--loops N] [--steps N] [--seed S] [--msa-depth D]
                      [--lm-dropout P] [--cap-gib G] [--lm-cache DIR] [--match-torch-rng [--torch-rng-device mps|cpu]]
```
`job.json`: `{"jobs": [{"name": "c1", "chains": [{"id": "A", "sequence": "..."}, {"id": ["B", "C"], "sequence": "...", "msa": "t.a3m"}]}]}`.
Protein complexes only (ligands, DNA/RNA, modifications are rejected with a message naming the chain). Per sample:
`out/<job>/sample_<i>/{complex.cif, pae_complex.npz, plddt_complex.npz, tokens_complex.npz, summary.json}` (the
BindCraft-Hunter rescore layout) and `out/<job>/coords.npz`. Exit 2: bad input; 3: the MLX worker failed (message names the
cause, e.g. its memory cap); 4: the preflight checks refused (disk over 90% full unless `ESMFOLD2_MLX_MIN_FREE_GB` is set,
or another process holding over 8 GB). The worker runs under a 28 GiB memory cap by default (`--cap-gib`, 0 disables), so an
over-large complex stops with a message instead of swapping the machine. ESMC-6B hidden states are cached in
`~/.cache/esmfold2_mlx/lm_states/` as exact bf16 bits (about 200 MB per 500-token complex). From Python: `esmfold2_mlx.fold.MLXFold(weights).fold(features, lm_states, FoldOptions())`
in an MLX-only process, `esmfold2_mlx.worker.RemoteFold` from a torch process, or `esmfold2_mlx.hybrid.install(model, weights)`
to make an esm `EsmFold2Model` run its post-ESMC forward in MLX.

## Reproducing

```
PY=/opt/homebrew/Caskroom/miniforge/base/envs/esmfold2-binder-design/bin/python
PYTHONPATH=. $PY scripts/convert_weights.py                         # weights/ (0.9 GB, gitignored)
PYTHONPATH=. $PY scripts/capture_refs.py CASE --device cpu|mps --stages STAGES   # refs/ (gitignored)
PYTHONPATH=. $PY -m pytest tests -q                                  # 231 pass, 2 integration tests skipped
ESMFOLD2_MLX_INTEGRATION=1 PYTHONPATH=. $PY -m pytest tests -q -k "esmc or install"   # ESMC-6B + install(): 2 pass
PYTHONPATH=. $PY bench/ckpt0.py | bench/ckpt1.py | bench/ckpt2.py | scripts/parity_report.py | scripts/real_esmc_parity.py | bench/profile_fold.py
```
Captures per test file (case list in `scripts/cases.py`): `pair_block` (m49, vegf1, cpu); `trunk` (m49, m49_q, vegf1,
tnf60_msa, tnf60_shallow, cpu); `heads` (m49, vegf1, tnf60_msa, cpu); `embed` (m49, vegf1, tnf60_msa, cpu and mps);
`sampler` (m49, vegf1, cpu then mps); `fold` (m49, vegf1, tnf60_msa, tnf60_shallow, cpu then mps); `features` for every case.
A missing ref fails the test with the exact capture command. Every real-model run (the CLI, captures, benchmarks) takes the
gpu lock and runs the checks in `esmfold2_mlx/preflight.py` (disk under `ESMFOLD2_MLX_MAX_DISK_PCT`, default 90, unless
`ESMFOLD2_MLX_MIN_FREE_GB` is set; no other process over 8 GB); scripts and benchmarks go through `scripts/preflight.py`.

## Open items

- NOT DONE: wiring into BindCraft-Hunter's rescore worker (Ethan's "later, separate step"; `hybrid.install` or `RemoteFold`
  are the two seams; the output layout already matches).
- NOT DONE: ESMFold2-Fast and the Experimental checkpoints; ligand and nucleic-acid inputs (rejected, not validated).
- NOT DONE: the layout/fusion work in finding 4.
- The integration tests print three warnings from esm's own torch code (config field rename, single-sequence notice, CUDA
  autocast notice); not from this port.
- Free disk: the machine was at 93-94% during this work. The ESMC hidden-state cache has no eviction or size cap (about
  200 MB per 500-token complex); clear `~/.cache/esmfold2_mlx/lm_states/` by hand.
- `--match-torch-rng` hands the worker one L x L x 256 dropout mask per loop (about 1.4 GB at 500 tokens) on top of the fold.
- One complex at a time: `MLXFold.fold` refuses batched features (the esm processor never batches).
