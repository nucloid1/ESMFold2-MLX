# ESMFold2-MLX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ESMFold2 (everything except ESMC-6B) in MLX, faster than the torch/MPS version with outputs as close to it as the arithmetic allows, as a standalone package with its own CLI.

**Architecture:** A torch parent process featurizes with the esm package's builder and runs ESMC-6B (bf16), frees it, then hands numpy arrays through shared memory to a persistent MLX worker that never imports torch and runs inputs-embedder, language-model shim, MSA encoder, looped pair trunk, distogram head, diffusion sampler and confidence head in fp32. Each stage is ported against torch-CPU captures, and three speed checkpoints (single block, full trunk, full fold) end the work if MLX does not beat torch/MPS.

**Tech Stack:** Python 3.12, MLX 0.32.2, numpy, safetensors, pytest 9.1.1; torch 2.14.0 and esm 3.4.1 in the parent process and the capture scripts only.

**Spec:** `docs/superpowers/specs/2026-10-29-esmfold2-mlx-design.md` (read it first; this plan refers to its sections as "spec §n"). Template to read alongside: `/Users/ethanputnam/Mol-1-MLX` (`mol_mlx/layers.py`, `msa.py`, `worker.py`, `guard.py`, `HANDOFF.md`).

## Global Constraints

- Python for everything: `PY=/opt/homebrew/Caskroom/miniforge/base/envs/esmfold2-binder-design/bin/python`. Never `conda run`. Install nothing into that env (it already has mlx 0.32.2, torch 2.14.0, esm 3.4.1, safetensors, accelerate, pytest); if a package is missing, stop and ask. Run tests as `PYTHONPATH=. $PY -m pytest tests -q`. Scripts that import torch import it first.
- fp32 throughout the MLX side. No bf16/fp16, no skipped layers or loops, no fewer sampling steps (spec §3, §4).
- Plain `biohub/ESMFold2` only, from the trunk-only copy `~/.cache/huggingface/esmfold2-trunk/ESMFold2` (1616 tensors, 937,903,372 bytes, HF key layout; map with `esm.models.esmfold2.hf_checkpoint.hf_state_dict_to_native`). ESMC-6B comes from the `biohub/ESMC-6B` cache. Protein complexes only are validated.
- Config: hidden 384, pair 256, `folding_trunk_num_hidden_layers` 48, `num_loops` 20 (so `total_steps = num_loops + 1 = 21`), `parcae_num_coda_layers` 2, 14 sampling steps, sampler `noise_scale=1.003, step_scale=1.5, max_inference_sigma=256.0`, schedule `inference_p=7.0, inference_s_max=160.0, inference_s_min=0.0004`, worker MSA column-mask rate 0.1.
- The MLX worker never imports torch. MLX timings are taken in a clean process; torch timings in their own process.
- References: torch CPU fp32 from the installed esm copy (its MPS patch is one dtype cast, spec §4) for stage gates; torch on MPS, loaded and called as the rescore worker does (`load_trunk_only`, eval, no_grad, default chunk size, same loops/steps/samples/MSA), for end-to-end comparison and every speed and memory baseline.
- Gates (spec §9): a stage blocks the next if fp32 relative error `rel_err(a, b) = ||a-b||_F / ||b||_F` against CPU references exceeds 1e-5; composite trunk and distogram block at 1e-3 relative or more than 0.5 percent of distogram-bin flips; end-to-end misses are reported as drift against sampling steps, not stops. No stage passes on near-zero activations (assert reference norm > 1e-3 in every parity test).
- Speed protocol (spec §7): warm means the second prediction onward in one process; at least 2 warmups and 5 timed runs; noise floor 10 percent, so "faster" means median at least 10 percent below torch's median; cold first-call time is reported too; torch spread is reported alongside. Memory by hard-cap survival (`memory_guard`) and a sampled watchdog, never `mx.get_peak_memory`.
- **Stop rule (spec §2):** if a checkpoint fails (Tasks 3, 6, 10), commit its report, stop, and tell Ethan with the numbers and which op the profile blames; do not start the next task and do not write Metal to rescue it.
- Machine: M3 Max, about 36 GB usable. Before any real-model run `scripts/preflight.py` must pass (disk below 90 percent, no other process holding more than 8 GB, GPU otherwise idle). Swap is fine; do not kill on swap.
- **One GPU run at a time (Ethan).** Any process that touches the GPU (pytest sessions, capture scripts, benchmarks, the CLI, the parent step) holds `gpu_lock()` (Task 1) for its whole lifetime and fails immediately, naming the holder's pid, if another run holds it; child processes it spawns (the MLX worker, test subprocesses) run under the parent's lock. Whoever executes this plan runs tasks strictly one after another: no parallel subagents, no reviewer running tests or benchmarks while an implementer's run is active, no background capture alongside another GPU job. The lock only covers this project; the other-process check in `preflight` covers everything else on the machine.
- Nothing is pushed. Commit messages end with the `Co-Authored-By` trailer the session specifies. Unit tests need `refs/` (gitignored, regenerate with Task 2's script); a missing ref fails the test with the regenerate command unless `ESMFOLD2_MLX_ALLOW_MISSING_REFS=1`. Tests that need the real model, ESMC or the GPU require `ESMFOLD2_MLX_INTEGRATION=1`.

## Review Focus

Failure modes the spec implies that no stage-parity test would hit, most likely first. Each has its test in the named task.

1. Unsupported molecules (ligands, DNA/RNA, modifications), duplicate chain ids, empty sequences: rejected with a clear message naming "protein complexes only", never a garbage fold. (Task 12)
2. MSA edge cases: query-only alignment, fewer rows than `max_depth`, one chain with an MSA and one without. (Task 5)
3. Padded tokens and atoms: results at real positions independent of padding; atom counts that are not a multiple of the attention window. (Tasks 3, 8)
4. More than one diffusion sample: each sample equal to running alone with the same draws. (Task 9)
5. Oversized input, memory cap hit, or a dead worker: a clear error naming the cap, non-zero exit, no hang, shared memory unlinked. (Tasks 11, 12)

---

### Task 1: Skeleton, weight conversion, key accounting

**Files:**
- Create: `.gitignore` (`weights/`, `refs/`, `__pycache__/`, `.pytest_cache/`), `README.md` (how to run tests and the CLI, three lines), `esmfold2_mlx/__init__.py`, `esmfold2_mlx/weights.py`, `esmfold2_mlx/config.py`, `scripts/convert_weights.py`, `esmfold2_mlx/gpu_lock.py`, `tests/conftest.py`, `tests/helpers.py`, `tests/test_weights.py`, `tests/test_gpu_lock.py`
- Run `git init` in `/Users/ethanputnam/ESMFold2-MLX` (the spec already sits there, uncommitted).

**Interfaces:**
- Produces (`weights.py`): `class Weights(dict)` recording reads, with `unused(prefix="") -> list[str]` and `assert_all_used(prefix="", allow=()) -> None` raising `KeyError` that lists unread keys; `key(prefix: str, name: str) -> str`; `load_group(weights_dir: str, group: str) -> Weights` (reads `<weights_dir>/<group>.safetensors` via `mx.load`).
- Produces (`config.py`): `@dataclass ModelConfig` loaded by `load_config(weights_dir) -> ModelConfig` from `<weights_dir>/config.json` (the converter copies the needed subset of the checkpoint's config so runtime never imports esm): at least `d_single_inputs, d_pair, n_trunk_layers, n_coda_layers, n_lm_encoder_layers, num_loops, n_sampling_steps`, MSA encoder fields (`max_depth`, `column_mask_rate`, layer and head counts), atom encoder/decoder and structure-head fields; and `@dataclass FoldOptions` with `num_loops: int | None = None`, `num_sampling_steps: int | None = None`, `num_diffusion_samples: int = 1`, `msa_max_depth: int | None = None`, `msa_column_mask_rate: float = 0.1`, `noise_scale: float = 1.003`, `step_scale: float = 1.5`, `max_inference_sigma: float = 256.0`, `seed: int = 0` (None means "the checkpoint's value").
- Produces (`gpu_lock.py`): `gpu_lock() -> ContextManager[None]`, an exclusive non-blocking `fcntl.flock` on `~/.cache/esmfold2_mlx/gpu.lock` that writes the holder's pid into the file and raises `SystemExit` with a message naming that pid when the lock is taken; the OS releases it when the holder dies. The holder exports `ESMFOLD2_MLX_LOCK_OWNER=<pid>`; `gpu_lock()` is a no-op in a process whose environment names a live ancestor as the owner, which is how the CLI, worker and bench scripts launched from a pytest session or another script run under that parent's lock while an unrelated process still fails. `tests/conftest.py` holds it in a session-scoped autouse fixture so every pytest session is one GPU run.
- Produces (`tests/helpers.py`): `rel_err(a, b) -> float` as defined in Global Constraints; `ref_path(case: str, stage: str, device: str = "cpu") -> Path` and `load_ref(case, stage, device="cpu") -> dict[str, np.ndarray]` implementing the missing-refs rule (layout in Task 2); `assert_live(x, floor=1e-3)`.
- Produces (`scripts/convert_weights.py`): `convert(src_dir: str, out_dir: str) -> dict` writing one fp32 safetensors file per top-level native module or parameter name (`folding_trunk`, `lm_encoder`, `msa_encoder`, `structure_head`, `confidence_head`, `parcae_log_a`, ...), `config.json`, and `manifest.json` (source key to native key map, group file per key, counts, element totals).

- [ ] **Step 1: Write the failing tests** in `tests/test_weights.py`: `test_assert_all_used_names_unread_keys` (a `Weights({"a.w": x, "b.w": y})` that only reads `a.w` raises `KeyError` mentioning `b.w`; `unused("a")` is empty), `test_key_joins_without_leading_dot`, `test_conversion_accounts_for_every_source_tensor` (runs `scripts/convert_weights.py` in a subprocess into `tmp_path`; asserts `manifest["source_tensors"] == 1616`, converted element total equals source element total, and every source key appears in the map), `test_converted_tensors_equal_native_mapped_source` (50 sampled keys bitwise equal to `hf_state_dict_to_native(raw)[native_key]` loaded fresh in the test via subprocess output saved as npz), `test_load_group_returns_tracked_weights`; in `tests/test_gpu_lock.py`: `test_second_holder_fails_fast_naming_pid` (a subprocess that calls `gpu_lock()` while this process holds it exits non-zero within 5 s with the holder's pid in stderr; the test takes the lock itself through a private lock path override so it does not fight the session fixture) `test_lock_released_when_holder_is_killed` (kill the holder with SIGKILL, a new holder then succeeds) and `test_descendant_inherits_lock_but_stranger_does_not` (a subprocess spawned by the holder calls `gpu_lock()` and succeeds; a process started with the owner variable stripped from its environment fails).
- [ ] **Step 2: Run** `PYTHONPATH=. $PY -m pytest tests/test_weights.py tests/test_gpu_lock.py -q`. Expected: FAIL (modules missing).
- [ ] **Step 3: Implement** `Weights`, `key`, `load_group` (adapt Mol-1-MLX `layers.py` `Weights`), `ModelConfig`/`FoldOptions`/`load_config`, `gpu_lock` with the conftest fixture (a lock path override via `ESMFOLD2_MLX_LOCK` for the test), `convert`. The converter reads `~/.cache/huggingface/esmfold2-trunk/ESMFold2/model.safetensors` and `config.json` (override with `--src`), maps with `hf_state_dict_to_native`, and writes to `weights/` (override `--out`). Print the native top-level group names it found; later tasks use those names.
- [ ] **Step 4: Run** the new tests, then run the converter for real into `weights/`. Expected: PASS; `weights/manifest.json` exists.
- [ ] **Step 5: Commit** (`git add -A` excluding ignored dirs): "skeleton: weights, config, converter with key accounting".

---

### Task 2: Reference corpus, torch loaders, parity baselines

**Files:**
- Create: `data/cases.json` and `data/tnfa_AC.a3m` (copied from `/Users/ethanputnam/Bindcraft-Hunter/docs/superpowers/spikes/spike4b-fast-vs-plain/cases.json` and `.../pilots/tnfa-ab/tnfa_AC.a3m`, plus `data/README.md` giving provenance), `scripts/preflight.py`, `scripts/torch_model.py`, `scripts/cases.py`, `scripts/capture_refs.py`, `scripts/cpu_vs_mps.py`, `tests/test_refs_present.py`, `docs/parity/cpu_vs_mps.md`, `docs/parity/mps_repeat_spread.md`

**Interfaces:**
- Produces (`scripts/preflight.py`): `preflight(max_disk_pct: float = 90.0, max_other_gb: float = 8.0) -> None` raising `SystemExit` with the reason; it takes `gpu_lock()` first and keeps it for the life of the process; every bench and capture script calls it first.
- Produces (`scripts/torch_model.py`, torch-first): `load_trunk_only(device: str)` (copy of the rescore worker's `load_trunk_only` in `/Users/ethanputnam/Bindcraft-Hunter/bindcraft_hunter/rescore/worker/worker_models.py`, same checks), `load_esmc_lazy(device)` (copy likewise), `featurize(case: Case, seed: int, device: str) -> tuple[dict, list]` (esm `ESMFold2InputBuilder.prepare_input`), `esmc_states(esmc, features) -> torch.Tensor` (`compute_lm_hidden_states` with `lm_mask_pct=0.0`), and `class DrawRecorder` (context manager that wraps `torch.randn`, `randn_like`, `rand`, `randperm`, `torch.nn.functional.dropout` and `torch.nn.init.trunc_normal_`, recording `(site, call_index, tensor)` only for calls whose caller frame is under `esm/models/esmfold2/`; dropout is stored as a bool keep-mask).
- Produces (`scripts/cases.py`): `@dataclass Case(name, chains: list[tuple[ids, sequence, a3m_path | None]])` and `CASES: dict[str, Case]`: `m49` (binder `CAAELAALEAELAALEGPWKGYPIPYGKLQFLIKKLKQLKVAC` alone, 43 tokens), `vegf1` (one VEGF target chain of case `VEGF_9KKU` plus M49, 145), `tnf60_msa` (first 60 columns of the TNF target with `tnfa_AC.a3m` cropped to match, plus M49, 103), `mixed_msa` (the 60-residue TNF chain with its MSA and M49 without), `vegf2` (`VEGF_9KKU`, 247), `il17` (`IL17_8B7W`, 369), `tnf2_msa` (two TNF target copies with the full `tnfa_AC.a3m` plus the 115-residue binder of `TNF_9BN7`, 429; verify the a3m query equals the target sequence, otherwise crop with the logic in `.../pilots/tnfa-ab/crop_a3m.py` and `prep_msa.py`), `il17_scfv` (`IL17_8DY1`, 499).
- Produces (`capture_refs.py`): CLI `capture_refs.py CASE --device {cpu,mps} --stages a,b,c [--seed 0]` with a registry `CAPTURES: dict[str, Callable]` that later tasks extend; always writes, shared by both devices, `refs/<case>/features.npz` (the feature tensors `EsmFold2Model.forward` consumes, MSA arrays included when the case has an MSA) and `refs/<case>/lm_states.npy` (float32 upcast of the bf16 ESMC output, computed once per case and reused), and, per device, `refs/<case>/<device>/draws.npz` (recorded random stream, keys `<site>/<call_index>`), `torch_out.npz` (full forward outputs under the worker's seed context and sampler settings) and one `<stage>.npz` per captured stage. CPU captures cover `m49, vegf1, tnf60_msa, mixed_msa`; MPS captures cover every case.
- Produces (`scripts/torch_profile.py`, torch-first): per-phase wall time of the torch/MPS fold (ESMC excluded; embed, trunk loops, distogram, sampler, confidence head) at `vegf2` and `tnf2_msa`, written to `docs/torch_baseline_profile.md`, so the share of each phase is known before any kernel work (spec §8).

- [ ] **Step 1: Write the failing test** `tests/test_refs_present.py::test_refs_exist_for_every_case` asserting, for each case and device it is meant to cover, that `features.npz`, `lm_states.npy` and the device's `draws.npz`, `torch_out.npz` exist and that `torch_out["sample_atom_coords"]` is finite and `distogram_logits` has shape `[1, N, N, 64]`.
- [ ] **Step 2: Run** it. Expected: FAIL (refs missing, message shows the capture command).
- [ ] **Step 3: Implement** the files above; copy the data files and write `data/README.md`; run `scripts/preflight.py`.
- [ ] **Step 4: Capture** `m49`, `vegf1`, `tnf60_msa`, `mixed_msa` on cpu and mps and every case on mps (long; run cpu captures in the background, one at a time, never alongside an MPS job). Then `scripts/cpu_vs_mps.py` writes `docs/parity/cpu_vs_mps.md` (relative difference of distogram logits and confidences between the two devices at `m49`, `vegf1`, same seed and same recorded draws replayed where the device allows) and `docs/parity/mps_repeat_spread.md` (two MPS runs of the same seed on `vegf2` and `tnf2_msa`: spread of ipTM, pTM, mean pLDDT and aligned coordinate RMSD). These two files set the end-to-end thresholds in Tasks 10 and 13. Run `scripts/torch_profile.py` too.
- [ ] **Step 5: Run** `tests/test_refs_present.py`. Expected: PASS.
- [ ] **Step 6: Commit**: "references: cases, torch loaders, recorded draws, cpu-vs-mps and mps spread".

---

### Task 3: Layers, `PairUpdateBlock`, and Checkpoint 0

**Files:**
- Create: `esmfold2_mlx/layers.py`, `esmfold2_mlx/trunk.py`, `tests/test_pair_block_vs_refs.py`, `bench/block_bench.py` (MLX, no torch), `bench/torch_block_bench.py` (torch-first), `docs/checkpoints/ckpt0-block.md`
- Modify: `scripts/capture_refs.py` (add stage `pair_block`)

**Interfaces:**
- Consumes: `Weights`, `key`, `load_group`, `ModelConfig`, `rel_err`, `load_ref`.
- Produces (`layers.py`): `LayerNorm.load(w, prefix, eps=1e-5)`, `Linear.load(w, prefix)` (`y = x @ W.T (+ b)`), `Transition(w, prefix, expansion_ratio=4)` mirroring esm `Transition` (layers.py:2503; residual inside, exactly as the torch module returns), `TriangleMultiplication(w, prefix, outgoing: bool)` mirroring `TriangleMultiplicativeBlock` (layers.py:2352; `__call__(z, mask) -> delta`, the update before the residual add, with einsum `bikd,bjkd->bijd` outgoing and `bkid,bkjd->bijd` incoming).
- Produces (`trunk.py`): `PairUpdateBlock(w, prefix)` with `__call__(z: mx.array [1,N,N,256], mask: mx.array [1,N,N] | None) -> mx.array`, equal to esm `PairUpdateBlock.forward` (layers.py:2601): `z = z + tri_mul_out(z)`, `z = z + tri_mul_in(z)`, `z = transition(z)`. Weight prefix inside `folding_trunk` is `blocks.{i}`.
- Native speed work belongs here, before the checkpoint measures anything: one concatenated matmul for the packed projections, `mx.compile(shapeless=True)` for the sigmoid gates and the SwiGLU elementwise pass (Mol-1-MLX `layers.py`), no hand-written Metal.

- [ ] **Step 1: Write failing tests** (`tests/test_pair_block_vs_refs.py`): `test_tri_mul_out_matches[case, block]`, `test_tri_mul_in_matches[case, block]`, `test_transition_matches[case, block]`, `test_pair_block_matches[case, block]` for cases `m49, vegf1` and blocks 0, 23, 47, each asserting `assert_live(ref)` and `rel_err(out, ref) < 1e-5`; `test_mask_excludes_padded_tokens`: a random `[1,45,45,256]` input with the last 5 tokens masked, block 0 weights, gives the same top-left `[40,40]` result (within 1e-6 relative) as the unpadded `[1,40,40,256]` input.
- [ ] **Step 2: Run** `PYTHONPATH=. $PY -m pytest tests/test_pair_block_vs_refs.py -q`. Expected: FAIL.
- [ ] **Step 3: Add the `pair_block` capture** (forward hooks on `folding_trunk.blocks[0|23|47]` and on each block's `tri_mul_out`, `tri_mul_in`, `pair_transition`, first loop only, plus the pair mask) and run it for `m49` and `vegf1` on cpu. **Implement** `layers.py` and `PairUpdateBlock`.
- [ ] **Step 4: Run** the tests. Expected: PASS. If a block misses 1e-5, fix before going on; do not widen the tolerance.
- [ ] **Step 5: Checkpoint 0 measurement.** `bench/torch_block_bench.py` (torch on MPS, trunk-only copy loaded as the worker does, default chunk size, `torch.no_grad`, `torch.mps.synchronize()` around each timing) and `bench/block_bench.py` (MLX, `mx.eval` sync) time block 0 with real weights at N of 250 and 400, fp32, 2 warmups, 7 timed runs, each script in its own process after `preflight`. Write medians, min, max for both, the ratio, and which of the three ops dominates in MLX to `docs/checkpoints/ckpt0-block.md`, ending in one line: `PASS` only if MLX's median is at least 10 percent below torch's at both sizes, else `FAIL`.
- [ ] **Step 6: Commit** the code and the report: "pair block in MLX, checkpoint 0 report". If the verdict is `FAIL`, stop here and report to Ethan (Global Constraints, stop rule).

---

### Task 4: Randomness interface, trunk loop without MSA

**Files:**
- Create: `esmfold2_mlx/rng.py`, `tests/test_trunk_loop_vs_refs.py`, `tests/test_rng.py`
- Modify: `esmfold2_mlx/trunk.py`, `scripts/capture_refs.py` (stages `lm_shim`, `loop_states`, `trunk_out`, `trunk_inputs`)

**Interfaces:**
- Consumes: Task 3 layers and `PairUpdateBlock`; `FoldOptions`; the recorded `draws.npz`.
- Produces (`rng.py`): `class Draws(Protocol)` with one method per random-draw site in the torch source (named here, signatures fixed by reading each call site and recorded in the Protocol docstring): `pair_init(shape) -> mx.array` (`_init_pair_state`, model.py:884, truncated normal with std `sqrt(2/(5*c))` clipped at 3 std), `msa_column_keep(shape, rate)` (layers.py:224), `msa_row_indices(loop, depth, max_depth)` (layers.py:201), `lm_dropout_keep(loop, shape, p)` (the per-loop `F.dropout` in `_run_one_loop`, model.py:890 onward), `diffusion_init(shape)`, `rotation_quat(step, n)`, `translation(step, shape)`, `step_noise(step, shape)` (layers.py:1710-1750, 1846, 1872). `class MLXDraws(seed: int)` (default; `mx.random`), `class ReplayDraws(npz: Mapping[str, np.ndarray])` (returns recorded tensors in call order, raises `IndexError` naming the site when asked for more than recorded). Sites a later task owns raise `NotImplementedError` in `MLXDraws` until that task.
- Produces (`trunk.py`): `LanguageModelShim(w, prefix, cfg)` mirroring esm `LanguageModelShim` (layers.py:2141; input `lm_hidden_states` exactly as `EsmFold2Model.forward` receives it, cast to float32); `FoldingTrunk(w, prefix, n_layers)` (list of `PairUpdateBlock`); `ParcaeTrunk(weights_dir: str, cfg: ModelConfig)` with `__call__(x_inputs, z_init, lm_hidden_states, pair_mask, num_loops: int, draws: Draws, msa_inputs: dict[str, np.ndarray] | None = None) -> mx.array [1,N,N,256]` (`msa_inputs` is the MSA entries of the feature dict: `msa`, `msa_attention_mask`, `has_deletion`, `deletion_value`; `None` skips the encoder) implementing `_run_one_loop` (model.py:890-1006) for `num_loops + 1` loops (`lm_encoder`, `z_init` re-injection, discretised `a` and `b_mat` from `_discretized_dynamics`, model.py:878), then `parcae_readout` and `parcae_coda` (2 blocks), returning float32 `z`. Weights are consumed through `Weights` and `assert_all_used` for `folding_trunk`, `lm_encoder`, `language_model`, `parcae_*` before the first call.

- [ ] **Step 1: Write failing tests:** `test_replay_draws_exhausted_raises` and `test_mlx_draws_same_seed_identical` (`tests/test_rng.py`); in `test_trunk_loop_vs_refs.py`: `test_lm_shim_matches` (1e-5), `test_single_loop_update_matches` (feed captured `z_in, z_init, lm_z` for loop 0, 1e-5), `test_loop_states_track_refs[case]` (replay draws; for each of the 21 loops `rel_err(z_k, ref_k) < 1e-3`), `test_trunk_out_matches` (`m49`, `vegf1`; `rel_err < 1e-3`; writes the per-loop error table to `docs/parity/trunk_growth.md`), `test_all_trunk_weights_consumed`.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** `rng.py`, the shim, `FoldingTrunk` and `ParcaeTrunk`; add the capture stages (the loop-state hooks capture `z` after each of the 21 loops and the inputs to the first loop; `trunk_inputs` also runs on mps for `vegf2, il17, tnf2_msa, il17_scfv` for Task 6).
- [ ] **Step 4: Run** the tests. Expected: PASS; read `docs/parity/trunk_growth.md` and note any growth with loop count in the commit message.
- [ ] **Step 5: Commit**: "pair trunk loop, randomness interface, replayed draws".

---

### Task 5: MSA encoder, memory guard

**Files:**
- Create: `esmfold2_mlx/msa.py`, `esmfold2_mlx/guard.py` (adapt Mol-1-MLX `guard.py`), `tests/test_msa_vs_refs.py`, `tests/test_guard.py`
- Modify: `esmfold2_mlx/rng.py` (implement `msa_column_keep`, `msa_row_indices` in `MLXDraws`), `esmfold2_mlx/trunk.py` (call the encoder in the loop when `msa_inputs` is given), `scripts/capture_refs.py` (stage `msa_encoder`)

**Interfaces:**
- Consumes: Task 4 `ParcaeTrunk`, `Draws`; `memory_guard` pattern.
- Produces (`msa.py`): `OuterProductMean`, `MSAPairWeightedAveraging`, `MSAEncoderBlock`, `MSAEncoder(weights_dir, cfg)` mirroring esm `OuterProductMean` (layers.py:2731; read `divide_outer_before_proj` from the config, Mol-1 divides after the projection), `MSAPairWeightedAveraging`, `MSAEncoderBlock`, `MSAEncoder` (model.py:1427-1538), `maybe_apply_msa_column_masking` and `maybe_subsample_msa` (layers.py:176-230) with their draws taken from `Draws`; `auto_chunks(n_tokens: int, n_rows: int, budget_bytes: int = 512 * 2**20) -> Chunks` (row and outer-product chunk sizes from a byte budget, never a constant; Mol-1-MLX `hybrid.auto_chunks`); `MSAEncoder.__call__(msa: np.ndarray | mx.array [M, N], msa_attention_mask, has_deletion, deletion_value, x_inputs, z, loop, draws, chunks) -> mx.array`.
- Produces (`guard.py`): `memory_guard(max_gib, interval_s=0.1, cache_gib=1.0, extra=())` context manager yielding a peak dict, as in Mol-1-MLX.

- [ ] **Step 1: Write failing tests** (`tests/test_msa_vs_refs.py`): `test_opm_matches`, `test_pair_weighted_averaging_matches`, `test_msa_block_matches`, `test_msa_encoder_matches` on `tnf60_msa` with replayed draws (each 1e-5, `assert_live`); `test_chunking_does_not_change_result` (chunk sizes 1, 7 and unchunked agree within 1e-6 relative); `test_query_only_msa_runs_and_matches` (alignment of one row), `test_rows_below_max_depth_do_not_subsample_or_fail`, `test_mixed_msa_and_no_msa_chains_match` (`mixed_msa` case); `test_msa_encoder_survives_cap_at_realistic_depth` (subprocess under `memory_guard(max_gib=12)` running the encoder at N=429 with the full 2225-row `tnfa_AC.a3m` and with a synthetic 7000-row alignment; must exit 0). In `tests/test_guard.py`: `test_guard_kills_process_over_cap` (a subprocess allocating past a 1 GiB cap dies with SIGKILL and prints the cap).
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Add the `msa_encoder` capture** (inputs and outputs of the encoder at loop 0 and the recorded row and column draws), capture it for `tnf60_msa`, `mixed_msa` on cpu. **Implement** `msa.py`, `guard.py`, the draws and the trunk hook.
- [ ] **Step 4: Run** the tests, then re-run `tests/test_trunk_loop_vs_refs.py` with the MSA case added to its parametrization (`rel_err < 1e-3` per loop). Expected: PASS.
- [ ] **Step 5: Commit**: "MSA encoder with byte-budget chunking, memory guard".

---

### Task 6: Checkpoint 1: full trunk, with and without MSA

**Files:**
- Create: `bench/trunk_bench.py` (MLX, no torch), `bench/torch_trunk_bench.py` (torch-first), `docs/checkpoints/ckpt1-trunk.md`

**Interfaces:**
- Consumes: Task 4 `trunk_inputs` captures on mps (`vegf2` 247, `il17` 369, `tnf2_msa` 429, `il17_scfv` 499); `ParcaeTrunk`, `MSAEncoder`, `memory_guard`.

- [ ] **Step 1: Write the harness.** The torch side times the trunk portion of the real forward on MPS (wrap `model._run_one_loop`, `parcae_readout` and `parcae_coda` with synchronised timers; run the forward with `num_sampling_steps=1, num_diffusion_samples=1` so only trunk time is read, defaults for loops) and records torch's driver-allocation peak with a sampler like the worker's `PeakMemory`. The MLX side loads the captured inputs and runs `ParcaeTrunk` with `MLXDraws`, 20 loops, synchronised. Cases: `vegf2`, `il17`, `il17_scfv` without MSA; `tnf2_msa` with MSA and again with its MSA dropped. Each in a fresh process after `preflight`; 2 warmups, 5 timed runs.
- [ ] **Step 2: Run both sides**, then re-run the MLX side under `memory_guard(max_gib=<torch's peak for that case>)`: it must complete.
- [ ] **Step 3: Write** `docs/checkpoints/ckpt1-trunk.md`: per case, torch median/min/max, MLX median/min/max, ratio, cold first call, memory result, trunk-output `rel_err` against the MPS capture (reported, expected well under 1e-3), and a profile of where MLX time goes per op class. End with `PASS` only if MLX is at least 10 percent faster at every case and memory survived at every case, else `FAIL`.
- [ ] **Step 4: Commit**: "checkpoint 1: trunk speed and memory". On `FAIL`, stop and report (stop rule).

---

### Task 7: Distogram and confidence heads

**Files:**
- Create: `esmfold2_mlx/heads.py`, `tests/test_heads_vs_refs.py`
- Modify: `scripts/capture_refs.py` (stage `heads`)

**Interfaces:**
- Consumes: `Linear`, `LayerNorm`, `FoldingTrunk`, `PairUpdateBlock`, `Weights`.
- Produces: `DistogramHead(w)` with `__call__(z) -> mx.array [1,N,N,64]` computing `distogram_head(z + z.swapaxes(1, 2))` (model.py:1322); `ConfidenceHead(weights_dir, cfg)` mirroring esm `ConfidenceHead` (model.py:104-405: input embedder, `dist_bin_pairwise_embed` with `boundaries`, 4-block `folding_trunk`, `row_attention_pooling`, pLDDT, PAE, PDE, resolved heads) with `__call__(s_inputs, z, x_pred, distogram_atom_idx, token_attention_mask, atom_to_token, atom_attention_mask, asym_id, mol_type, num_diffusion_samples, relative_position_encoding, token_bonds_encoding) -> dict[str, mx.array]` returning exactly the keys the torch head returns.

- [ ] **Step 1: Write failing tests**: `test_distogram_matches` (1e-5, `assert_live`), `test_confidence_head_outputs_match` (every output key, 1e-5, driven by captured `x_pred` and trunk `z`), `test_confidence_keys_equal_torch_keys`, `test_all_head_weights_consumed`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Capture** inputs and outputs of both heads at the real forward for `m49, vegf1, tnf60_msa` on cpu, and **implement** `heads.py`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit**: "distogram and confidence heads".

---

### Task 8: Input embedder, atom encoder/decoder, pair init

**Files:**
- Create: `esmfold2_mlx/atoms.py`, `esmfold2_mlx/embed.py`, `tests/test_embed_vs_refs.py`
- Modify: `scripts/capture_refs.py` (stage `embed`)

**Interfaces:**
- Produces (`atoms.py`): `SWA3DRoPEAttention`, `SWAAtomBlock`, `SWAAtomTransformer`, `EsmFold2AtomEncoder`, `EsmFold2AtomDecoder` mirroring layers.py:561-1017 (sliding-window attention of window `sliding_window` over atoms with 3D rotary embeddings; implement as blocked attention, never a dense atom-by-atom matrix).
- Produces (`embed.py`): `InputsEmbedder(weights_dir, cfg)`, `RelativePositionEncoding(w, cfg)` (esm `ResIdxAsymIdSymIdEntityIdEncoding`, layers.py:2024), and `class Embedder(weights_dir: str, cfg: ModelConfig)` (owns `InputsEmbedder`, `RelativePositionEncoding`, `z_init_1`, `z_init_2`, `token_bonds`) with `__call__(features: dict[str, np.ndarray]) -> Embedded` where `Embedded` has `x_inputs [1,N,451]`, `z_init [1,N,N,256]` (`z_init_1(x)+z_init_2(x)` outer sum, plus relative position and token-bond encodings), `relative_position_encoding`, `token_bonds_encoding`, `pair_mask`, and the one-hot preparation of `forward` (model.py:1076-1126) including the zeroing of padded atoms and tokens.

- [ ] **Step 1: Write failing tests**: `test_atom_encoder_matches`, `test_inputs_embedder_matches`, `test_rel_pos_matches`, `test_embedder_z_init_matches` (all 1e-5, `assert_live`, on `m49` and `vegf1`); `test_atom_window_with_padding` (a case whose atom count is not a multiple of the window, and the same input with extra padded atoms masked, agree at real atoms within 1e-6 relative); `test_all_embed_weights_consumed`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Capture** (`embed` stage: feature inputs, `x_inputs`, `z_init`, rel_pos, token bonds, atom encoder outputs) for `m49, vegf1` on cpu; **implement**.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit**: "inputs embedder and atom attention".

---

### Task 9: Diffusion module and sampler

**Files:**
- Create: `esmfold2_mlx/diffusion.py`, `tests/test_diffusion_vs_refs.py`
- Modify: `esmfold2_mlx/rng.py` (implement the four diffusion sites in `MLXDraws`), `scripts/capture_refs.py` (stages `denoise`, `sampler_trace`)

**Interfaces:**
- Consumes: `atoms.py`, `embed.py` outputs, `Draws`, `FoldOptions`.
- Produces: `DiffusionConditioning`, `DiffusionTransformer` (`AttentionPairBias`, `ConditionedTransitionBlock`), `DiffusionModule` mirroring layers.py:1019-1643; `weighted_rigid_align(x, target, weights, mask) -> mx.array` (layers.py:1756; the 3x3 SVD runs on `mx.cpu` or uses a closed-form rotation, whichever matches torch within 1e-5 on aligned coordinates and is not slower; the choice and its measurement go in the module docstring); `DiffusionSampler(weights_dir, cfg).sample(z_trunk, embedded, features, options: FoldOptions, draws: Draws) -> mx.array [n_samples, atoms, 3]` mirroring `DiffusionStructureHead.sample` (layers.py:1791-1937, including the schedule `inference_p/s_max/s_min`, 14 steps by default, per-step centring, random rotation and translation, noise, denoise, rigid align, step update).

- [ ] **Step 1: Write failing tests**: `test_denoise_call_matches` (one captured denoiser call, 1e-5, `assert_live`), `test_rigid_align_matches_torch_svd` (random 3x3 covariances including near-degenerate and reflection cases, compared on aligned coordinates, 1e-5), `test_sampler_trace_tracks_refs` (replayed draws; per-step aligned coordinate RMSD against the capture, final RMSD asserted below 0.5 angstrom on `m49` and `vegf1`, and the full per-step table written to `docs/parity/sampler_drift.md`), `test_multi_sample_batch_matches_single` (`num_diffusion_samples=3` equals three single-sample runs given the same per-sample draws, 1e-5), `test_all_diffusion_weights_consumed`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Capture** the denoiser inputs and outputs for steps 0, 7 and 13 and the full per-step coordinate trace for `m49, vegf1` (cpu), and **implement**. The 0.5 angstrom bound is a first guess: if torch-versus-torch noise (Task 2's spread file) is larger, set the bound to that spread and say so in the test.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit**: "diffusion module and sampler".

---

### Task 10: `MLXFold` assembly and Checkpoint 2

**Files:**
- Create: `esmfold2_mlx/fold.py`, `tests/test_fold_vs_refs.py`, `scripts/parity_report.py`, `bench/fold_bench.py` (MLX), `bench/torch_fold_bench.py` (torch-first), `docs/checkpoints/ckpt2-fold.md`, `docs/parity/end_to_end.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `class MLXFold(weights_dir: str)` with `fold(features: dict[str, np.ndarray], lm_hidden_states: np.ndarray, options: FoldOptions, draws: Draws | None = None) -> dict[str, np.ndarray]` (the MSA arrays travel inside `features`; the encoder runs when they are present) returning numpy arrays under exactly the keys `EsmFold2Model.forward` returns (`distogram_logits`, `sample_atom_coords`, the confidence outputs, `atom_pad_mask`, `residue_index`, `entity_id`); `draws=None` means `MLXDraws(options.seed)`. It assembles embed, language-model shim, trunk (with MSA when `msa` is given), distogram head, sampler and confidence head in the order of `EsmFold2Model.forward` (model.py:1008-1401), calling `mx.eval` and `mx.clear_cache()` between phases to bound memory.

- [ ] **Step 1: Write failing tests**: `test_fold_output_keys_equal_torch` (`m49`), `test_fold_matches_torch_cpu[case]` for `m49, vegf1, tnf60_msa, mixed_msa` with `ReplayDraws`: composite `rel_err(distogram_logits) < 1e-3`, distogram bin flips under 0.5 percent, and every other output compared and tabulated (not asserted) into `docs/parity/end_to_end.md` by `scripts/parity_report.py`, which also shows the confidences (ipTM, pTM, mean pLDDT via esm's own `decode`) and the aligned RMSD against the two spread files from Task 2; `test_same_seed_twice_identical` (two `MLXDraws(7)` runs agree within 1e-6 relative) and `test_different_seeds_differ` (coordinates differ).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `fold.py` and the report script; run the parity report on all CPU cases and on `vegf2, il17, tnf2_msa` against the MPS captures.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Checkpoint 2.** Benchmark the whole fold, MLX against torch/MPS, cases `vegf2`, `il17`, `tnf2_msa` (with MSA and with it dropped), `il17_scfv`, each at `num_diffusion_samples` 1 and 3, 20 loops, the checkpoint's default 14 steps, protocol in Global Constraints, torch driver peak against MLX run under `memory_guard(max_gib=<torch peak>)`. Write `docs/checkpoints/ckpt2-fold.md` with the table, cold times and a per-phase profile of the MLX fold; end with `PASS` only if MLX is at least 10 percent faster at every case and configuration and memory survived, else `FAIL`.
- [ ] **Step 6: Commit**: "MLXFold, end-to-end parity, checkpoint 2". On `FAIL`, stop and report.

---

### Task 11: Worker process

**Files:**
- Create: `esmfold2_mlx/worker.py`, `tests/test_worker.py`
- Reference: `/Users/ethanputnam/Mol-1-MLX/mol_mlx/worker.py` and `tests/test_worker.py`

**Interfaces:**
- Consumes: `MLXFold`, `memory_guard`.
- Produces: protocol helpers `pack(arrays) -> (SharedMemory, layout)`, `allocate(spec)`, `views(shm, layout)`; worker entry `python -m esmfold2_mlx.worker --weights DIR [--cap-gib G]` (length-prefixed pickle on stdin/stdout, arrays through shared memory, stray prints to stderr, the first message `("ready", {"pid": int, "torch_imported": bool})`); `class RemoteFold(weights_dir: str, cap_gib: float | None = None)` with `__call__(features, lm_hidden_states, options: FoldOptions, torch_draws: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]`, `close()`, `last_seconds`, and `class WorkerError(RuntimeError)`. A dead worker raises `WorkerError` naming the exit code and the cap; the next call restarts it; it never falls back to torch.

- [ ] **Step 1: Write failing tests**: `test_worker_matches_in_process` (`m49`, within 1e-5 relative of `MLXFold`), `test_worker_is_reused_across_calls` (same pid on the second call), `test_worker_never_imports_torch` (ready info `torch_imported is False`), `test_dead_worker_raises_then_restarts` (kill it mid-idle, next call raises `WorkerError`, the call after succeeds), `test_cap_exceeded_error_names_cap` (a 0.5 GiB cap on `vegf2` raises `WorkerError` whose message contains the cap), `test_shared_memory_unlinked_after_error` (no `psm_` or leftover blocks from this run remain in `/dev/shm`-equivalent listing via `multiprocessing.shared_memory` bookkeeping).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** by adapting Mol-1-MLX's worker (same framing, shared-memory attach/unregister workaround, guard in the serve loop).
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit**: "MLX worker process with memory cap".

---

### Task 12: Torch parent step, CLI, `install` hook

**Files:**
- Create: `esmfold2_mlx/esmc_step.py`, `esmfold2_mlx/inputs.py`, `esmfold2_mlx/cli.py`, `esmfold2_mlx/hybrid.py`, `bin/esmfold2-mlx`, `tests/test_inputs.py`, `tests/test_cli.py`, `tests/test_hybrid.py`

**Interfaces:**
- Consumes: `RemoteFold`, `FoldOptions`; esm `ESMFold2InputBuilder`, `processor.decode`; the rescore worker's `load_esmc_lazy`, `embed_request`, `export_sample` (copy and adapt; do not import `bindcraft_hunter`).
- Produces (`inputs.py`): `parse_jobs(path: str) -> list[Job]` for a JSON file `{"jobs": [{"name": str, "chains": [{"id": str | [str], "sequence": str, "msa": str | null}]}]}`; raises `ValueError` whose message contains "protein complexes only" for any chain that is not a plain amino-acid sequence, any `modifications` key, duplicate chain ids, or an empty sequence; `Job.to_spec() -> StructurePredictionInput` (esm `ProteinInput` with `MSA.from_a3m`).
- Produces (`esmc_step.py`, torch-first): `compute_states(jobs, esmc_repo: str, device: str) -> list[ParentJob]` loading ESMC once, computing every job's features and `lm_hidden_states`, then freeing ESMC (`del`, `gc.collect()`, `torch.mps.empty_cache()`) and returning numpy `ParentJob(name: str, spec: StructurePredictionInput, features: dict[str, np.ndarray], lm_hidden_states: np.ndarray, chain_infos: list)`; it logs "esmc freed" before returning.
- Produces (`cli.py`): `esmfold2-mlx pred -i job.json -o out/ [--samples N] [--loops N] [--steps N] [--seed S] [--msa-depth D] [--weights DIR] [--cap-gib G] [--match-torch-rng]` (Task 13 implements the last flag; until then it exits with a clear "not implemented" message). All jobs' states are computed first, ESMC is freed, then the worker folds each job in turn, and each sample is decoded in the parent and written as `out/<job>/sample_<i>/{complex.cif, pae_complex.npz, plddt_complex.npz, tokens_complex.npz, summary.json}` (same layout as the rescore worker; `summary.json` last). Exit status is non-zero on any `WorkerError` or `ValueError`, with the message on stderr. The parent holds `gpu_lock()` for the whole run; the worker runs under it.
- Produces (`hybrid.py`): `install(model, weights_dir: str, cap_gib: float | None = None) -> None` replacing the post-ESMC computation of an `EsmFold2Model` instance's `forward` with a `RemoteFold` call so that `model(**features, lm_hidden_states=..., ...)` returns the same keys as before.

- [ ] **Step 1: Write failing tests**: `test_inputs.py`: `test_rejects_ligand_chain`, `test_rejects_dna_and_rna_sequences`, `test_rejects_modifications`, `test_rejects_duplicate_chain_ids`, `test_rejects_empty_sequence` (each: `ValueError`, message contains "protein complexes only" or names the offending chain), `test_accepts_homo_multimer_ids_list`, `test_msa_path_must_exist`. `test_cli.py`: `test_cli_exits_nonzero_on_unsupported_input` (stderr names the chain), `test_cli_exits_nonzero_when_worker_killed_by_cap` (tiny cap; exit status non-zero, message names the cap, no hang within 120 s), `test_esmc_freed_before_worker_start` (a fake ESMC and recorded event order: "esmc freed" precedes "worker ready"), `test_cli_folds_several_jobs_in_one_process` and `test_same_seed_twice_identical_outputs` (integration, `ESMFOLD2_MLX_INTEGRATION=1`, `m49` twice). `test_hybrid.py`: `test_install_returns_torch_keys` (integration; keys equal the torch forward's, `m49`).
- [ ] **Step 2: Run** `tests/test_inputs.py tests/test_cli.py tests/test_hybrid.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the four modules and `bin/esmfold2-mlx` (a two-line launcher setting `PYTHONPATH` to the repo root and running `python -m esmfold2_mlx.cli`).
- [ ] **Step 4: Run** the unit tests, then with `ESMFOLD2_MLX_INTEGRATION=1` the integration ones (after `preflight`). Expected: PASS.
- [ ] **Step 5: Commit**: "CLI, ESMC parent step, install hook".

---

### Task 13: `--match-torch-rng`

**Files:**
- Create: `esmfold2_mlx/torch_draws.py`, `tests/test_torch_draws.py`
- Modify: `esmfold2_mlx/rng.py` (add `PackedDraws`), `esmfold2_mlx/worker.py` and `esmfold2_mlx/cli.py` (carry the packed draws), `esmfold2_mlx/esmc_step.py`

**Interfaces:**
- Produces (parent, torch-first): `draw_torch_stream(plan: DrawPlan, seed: int, device: str) -> dict[str, np.ndarray]` drawing, under the same seed context the rescore worker uses (`esm.models.esmfold2.processor._seed_context`) and in the order the torch forward consumes randomness (pair init, MSA column mask, then per loop the LM-dropout keep mask and the MSA row indices, then the sampler's initial noise and per step rotation, translation and noise), every site `Draws` defines, for shapes derived from the features and options (`DrawPlan.from_features(features, options)`); dropout masks are stored as bool. Produces (worker) `PackedDraws(arrays: Mapping[str, np.ndarray])`, a `Draws` reading from shared memory. The option `--torch-rng-device {mps,cpu}` (default `mps`) selects which torch generator; CPU and MPS streams differ.

- [ ] **Step 1: Write failing tests**: `test_torch_stream_equals_recorded_stream[m49, vegf1]` (integration; `draw_torch_stream` with the capture's seed and device equals `refs/<case>/mps/draws.npz` bitwise, every site), `test_packed_draws_exhausted_raises`, `test_cli_match_torch_rng_runs` (integration, `m49`; completes and writes outputs), `test_end_to_end_same_seed_close_to_torch` (integration, `vegf1`: confidences and aligned coordinate RMSD against `torch_out.npz` within the spreads in `docs/parity/mps_repeat_spread.md`, otherwise write the drift into `docs/parity/end_to_end.md` and fail with the numbers so a human decides).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** If the recorded and generated order disagree at some site, fix the order in `draw_torch_stream`, not the test.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit**: "match-torch-rng: torch-drawn randomness for cross-implementation seeds".

---

### Task 14: Profile-driven Metal pass (conditional)

**Files:**
- Create: `bench/profile_fold.py`, `bench/matmul_peak.py`, `docs/profile.md`; `esmfold2_mlx/metal/<op>.py` and `tests/test_metal_<op>.py` only for ops that qualify

**Interfaces:**
- Consumes: `MLXFold`, the per-phase profile in `docs/checkpoints/ckpt2-fold.md`.

- [ ] **Step 1: Profile.** `bench/profile_fold.py` times each op class of the fold with per-op synchronisation at N of 250 and 400 (triangle multiplication, pair transition, MSA encoder ops, atom attention, diffusion transformer, rigid align, confidence head); `bench/matmul_peak.py` re-measures this GPU's fp32 matmul rate. Write `docs/profile.md`.
- [ ] **Step 2: Qualify.** An op class gets a kernel only if it is at least 25 percent of MLX fold time and a Mol-1-MLX-style ablation (remove one component at a time, measure) shows the arithmetic itself, not layout copies, is the cost. If nothing qualifies, record that in `docs/profile.md` and finish the task.
- [ ] **Step 3: For each qualifying op**, one commit each: write `tests/test_metal_<op>.py` first (kernel equals the native MLX op within 1e-5 relative, including a size that is not a multiple of the tile, masked padding, and large logits where attention is involved), see it fail, write the kernel in `esmfold2_mlx/metal/<op>.py` with a switch to the native path, see it pass, then keep it only if it gives at least 1.1x at stage level in the profile and the full-fold parity tests still pass; otherwise delete it and record the attempt.

---

### Task 15: Final benchmark matrix and handoff

**Files:**
- Create: `bench/matrix.py`, `HANDOFF.md`, `docs/results.md`

- [ ] **Step 1: Run the matrix** through the real CLI and worker against torch/MPS: cases `vegf2`, `il17`, `tnf2_msa` (with MSA and with it dropped), `il17_scfv`; `--samples` 1 and 3; defaults for loops and steps; cold first call and 5 warm runs per cell; MLX peak by `memory_guard` survival and torch peak by driver allocation; torch spread reported next to every ratio.
- [ ] **Step 2: Write** `HANDOFF.md` in Mol-1-MLX's style: status, decisions, scope as read, every claim tagged VERIFIED (a command ran and its output was read), INFERRED or NOT DONE, the benchmark and parity tables, the findings that changed decisions, how to reproduce (`scripts/convert_weights.py`, `scripts/capture_refs.py`, test and bench commands), and open items. Copy the final numbers into `docs/results.md`.
- [ ] **Step 3: Run** the full unit suite `PYTHONPATH=. $PY -m pytest tests -q`, then the integration set with `ESMFOLD2_MLX_INTEGRATION=1`. Expected: all pass; report counts honestly in `HANDOFF.md`.
- [ ] **Step 4: Commit**: "final benchmark matrix and handoff".
