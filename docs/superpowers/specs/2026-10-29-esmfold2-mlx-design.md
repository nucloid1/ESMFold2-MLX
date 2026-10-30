# ESMFold2-MLX: design

Status: draft for Ethan's review, 2026-10-29. Sections say who decided what: **Decided** is Ethan's, **My call** is the author's and open to override, **Found** is read from code or logs on 2026-10-29.

## 1. Intent

Run ESMFold2 (everything except ESMC-6B) in MLX with hand-written Metal where MLX's own ops lose, as a standalone package with its own CLI, modelled on `/Users/ethanputnam/Mol-1-MLX`. Reason (Ethan): MLX/Metal is slightly more memory efficient and, for Mol-1, 1.5 to 2x faster than PyTorch on this Mac. Success (Ethan): **faster than the PyTorch version, with outputs as close as possible.**

## 2. Decided

- ESMC-6B stays in torch. Everything after it moves to MLX: input embedder, language-model pair shim, MSA encoder, the looped folding trunk, distogram head, diffusion structure head (atom encoder and decoder, sampler), confidence head.
- Standalone, like Mol-1-MLX: its own CLI. Wiring it into the `bindcraft_hunter` rescore worker is a later, separate step.
- Faster than the torch version; outputs as close as possible.
- If the MLX version does not beat torch on MPS, stop. (Ethan's words, in answer to the reviewer: "if it doesn't beat torch, stop".) My reading, to be checked: at each checkpoint in section 9, a stage that is not faster than its torch/MPS counterpart ends the work and is reported to Ethan, who decides what next. There is no pushing on with more Metal past a failed checkpoint and no shipping a torch stage to cover for it.
- MSA support is required for "done" and for the speed claim. (Ethan.)
- Scope is the ESMFold2 port and nothing else. The source is `github.com/Biohub/esm` (`esm.models.esmfold2`; the installed copy is 3.4.1, upstream's latest is 3.4.1.post1, and its README mentions CUDA, CPU and ROCm only, no MLX, Metal or MPS path). Nothing from the BindCraft-Hunter design pipeline is part of this for now, and nothing in this project imports `bindcraft_hunter`.

## 3. Goals, non-goals, anti-goals

**Goals**
1. Faster wall-clock per fold than the torch/MPS path, same inputs and settings, warm, at the sizes in use (about 250 to 480 tokens), both without an MSA and with a real one.
2. Same inputs and same seed give outputs as close to torch's as the arithmetic allows.
3. Peak memory no higher than torch's at the same size (a lower figure is welcome; no number is promised).

**Non-goals**
- ESMC in MLX; training; gradients; CUDA; replacing the rescore worker now.
- ESMFold2-Fast and the Experimental checkpoints. The first target is plain `biohub/ESMFold2`. (My call.)
- Ligand and nucleic-acid validation. Features for them go through the same graph, but only protein complexes are tested and claimed. (My call.)

**Anti-goals (a failure even if it technically works)**
- Speed bought by changing outputs: half precision, skipped layers or loops, fewer sampling steps, any approximation that moves results past the parity bar.
- A speed or parity claim with no baseline, or from a test that compares near-zero to near-zero. Mol-1-MLX's `attention_pair_bias` weights were about 1e-32, so matching captures proved nothing there and random-weight tests were needed.
- Weights silently unused. Mol-1-MLX made the model refuse to build unless every key was consumed; keep that.

## 4. Constraints

- Machine: M3 Max, 48 GB unified, about 36 GB usable. Swap is fine; memory is the constraint. Check the GPU is idle and the disk is below about 90 percent before real-model runs.
- fp32 throughout the MLX side. Mol-1-MLX measured bf16 at only about 1.13x for the stack with 0.4 to 0.7 percent of pair tokens changing; Ethan chose fp32 only there. (My call that it carries over; it also matches "as close as possible".)
- Env: `esmfold2-binder-design`, which already has MLX 0.32.2, torch 2.14.0 and esm 3.4.1. The rescore env is never modified; install nothing into it. If the port needs extra packages, ask first.
- References come from the real checkpoint on torch CPU, one module at a time, so the GPU holds only MLX tensors during a test (Mol-1-MLX's method). The installed `layers.py` has a local MPS patch; a diff against `layers.py.pre-mps-patch.bak` shows one added dtype cast and nothing else, so CPU references from the installed copy are valid.
- Nothing is pushed anywhere.

## 5. What was found

**The model** (`esm.models.esmfold2`, installed in the env above):
- Config: hidden 384, pair 256, `folding_trunk_num_hidden_layers` 48, default `num_loops` 20, `parcae_num_coda_layers` 2, atom encoder of 3 layers, 4 heads, head_dim 32, sliding window 128, default 14 sampling steps. The trunk-only copy at `~/.cache/huggingface/esmfold2-trunk/ESMFold2` holds 1616 tensors, 937,903,372 bytes, and was verified bitwise against the source shards.
- The trunk is pair-only: each `PairUpdateBlock` is `tri_mul_out`, `tri_mul_in`, `pair_transition`. **There is no triangle attention**, so Mol-1-MLX's main kernel finding (head_dim 32 defeating MLX's fused attention) does not transfer. The expensive pieces are the N^3 triangle multiplications and the pair transitions, run for `num_loops + 1` loops over 48 blocks, plus 2 coda blocks.
- The loop is a recurrent "Parcae" update (`z` mixed through a discretised `a`, `b_mat`, with `z_init` re-injected each loop), then readout and coda. `s_trunk` is `None`: there is no single track in the trunk.
- Randomness in one fold: the initial pair state, one MSA column mask, a fresh MSA row subsample every loop, per-loop dropout on the language-model pair when the config enables it, initial diffusion noise, per-step noise, a random rotation and translation per step, and an SVD in the rigid alignment (`torch.linalg.svd` on 3x3 matrices).
- The package's fused kernels are Triton and CUDA (`kernels/`: fused dual GEMM, trimul, layernorm-residual and others); the rescore worker already disables them on MPS (`set_kernel_backend(None)`).
- `forward` takes `lm_hidden_states` as an input, which is the seam: ESMC runs in torch and hands its hidden states over. The rescore worker already does exactly this (`embed_request`, then `fold_request`).

**The baseline** (Spike 4b, `folds_plain.jsonl`, plain ESMFold2, trunk-only, MPS, config defaults): 247 tokens, no MSA, seed 0: 32.55 s and 4.41 GB peak driver memory. The sample count of that row is not recorded in what was read. Memory notes record ESMFold2 as near-deterministic on MPS (repeats agree to about 1e-4); Boltz is not.

**Mol-1-MLX, the template** (its `HANDOFF.md`): hybrid port, trunk in MLX in a worker process, everything else torch; per-module parity about 1e-7 to 1e-6 relative in fp32; confidences matched to three decimals; speedup depends strongly on size (about 1.05 to 1.3x at 128 tokens, about 2x at 320 to 384, 1.27x end to end at 97 tokens because diffusion stayed in torch). **Mol-1-MLX never ported diffusion**, so ESMFold2's sampler has no template. Other lessons carried over: MLX's peak-memory counter is unreliable (use hard-cap survival or a sampled watchdog); run-to-run noise is about 10 percent so smaller differences are noise; benchmark torch the way production calls it.

**Mol-1-MLX lessons that shape this design** (read from its `mol_mlx/` and `HANDOFF.md` on 2026-10-29):
- **MLX ran about 33 percent slower in a process where torch had already done CPU-side work** (loading a checkpoint, featurizing); the cause was never identified, and the cure was a separate persistent MLX worker process (N=97 stack: 568 ms clean, about 760 ms in-process). Its first prediction in a process cost the same as torch's (about 7.8 s) because worker start-up and kernel warm-up cancel the gain, so the saving appears from the second prediction on.
- The worker protocol is an 8-byte length prefix plus a pickle over stdin/stdout, with arrays passed through shared memory and never over the pipe. A dead worker raises; it never falls back to torch silently. A process-level memory watchdog (`guard.py`) kills the process when MLX's active-plus-cache bytes or resident memory pass a cap, because torch/MPS buffers are invisible to both counters and `mx.set_memory_limit` is advisory.
- **The MSA module is a direct precedent.** Mol-1's `msa.py` ports outer-product-mean and pair-weighted-averaging blocks that share their shape with ESMFold2's `OuterProductMean`, `MSAPairWeightedAveraging` and `MSAEncoderBlock`. It keeps the per-cycle MSA row sampling in torch and replays the pre-drawn samples in MLX (shipped to the worker as uint8, which is exact), runs row-wise ops in chunks sized from a memory budget, and found real alignments much deeper than synthetic ones: 7000 rows at N=586 peaked at 18.3 GiB in the worker, so the chunk size is derived from a byte budget and not a constant. One precision detail: Mol-1 divides by the MSA row count after the output projection, bias included; ESMFold2 makes that order selectable (`divide_outer_before_proj`), so it must be read from the config.
- Method: profile before guessing (Mol-1's guesses that the MSA module and template embedder were slow were wrong; the stack was 87 percent), and ablate before restructuring a kernel (removing exps, bias loads and the accumulator rescale changed nothing; the cost was elsewhere). Elementwise fusion with `mx.compile(shapeless=True)` and one concatenated matmul gave only about 1.065x on the whole stack, so it is worth doing but is not where a 1.5x comes from.
- Torch/MPS timings are noisy (28 percent run to run at one N); MLX repeats to about 1 percent. Quote speedups as ranges and report the torch spread alongside.

## 6. Shape (my call)

- Package `esmfold2_mlx`, CLI `bin/esmfold2-mlx`, directory `~/ESMFold2-MLX`, own git repo, weights converted to safetensors from the trunk-only copy into a gitignored `weights/`, references in a gitignored `refs/`.
- `esmfold2-mlx pred <input> -o <dir>` takes what `StructurePredictionInput` takes (multi-chain complexes, optional MSAs and modifications) plus options for diffusion samples, loops, sampling steps and seed. It writes `complex.cif`, PAE and pLDDT arrays and `summary.json` (ipTM, pTM, mean pLDDT, tokens, seconds), in the same layout the BindCraft-Hunter rescore worker uses so a later wiring step is trivial; that is a convenience, not a dependency.
- Featurization (`ESMFold2InputBuilder.prepare_input`) and decoding to mmCIF stay as the esm package's code.
- Process layout, following Mol-1-MLX: a torch parent process featurizes with the esm package's builder and runs ESMC-6B in bf16 (code adapted from the worker's `load_esmc_lazy` and `embed_request`), then frees ESMC; a long-lived MLX worker process, which never imports torch, loads the roughly 0.9 GB of weights once and folds, taking numpy arrays through shared memory and returning the coordinates, distogram and confidence outputs. ESMC's 12 GB and the MLX side are never resident together. The worker has a memory watchdog and a cap flag, as in Mol-1-MLX.
- Shape of the code, following Mol-1-MLX's `hybrid.install`: the core is an `MLXFold` object (numpy in, numpy out) behind the worker; the CLI uses it directly, and an `install(model, weights)` hook swaps it in for the post-ESMC part of `EsmFold2Model.forward` so anything that already drives esm's processor can use it. Wiring into any other project is not part of this work.
- Build order: weight conversion with strict key accounting; folding trunk and loop; confidence head; input embedder, relative position and language-model shim; MSA encoder; atom encoder and decoder; diffusion sampler; CLI and worker. Each stage is gated on parity against CPU references before the next.
- Metal only where measurement says MLX's native ops lose. Candidates, to be confirmed by profiling and not assumed: triangle multiplication, the SwiGLU pair transition, sliding-window atom attention with 3D RoPE.

## 7. Verification (my proposal; Ethan to adjust)

- **Per stage:** fp32 relative error on real-checkpoint captures at N of about 48, 97, 250 and 400, aiming at Mol-1-MLX's 1e-6 or better; no stage passes on near-zero weights or activations.
- **End to end:** same seed, with the random tensors drawn in torch under the worker's seed context and handed to MLX, so a torch run and an MLX run consume the same randomness. Compare ipTM, pTM, mean pLDDT, PAE and coordinates (aligned RMSD). The aim is agreement within torch's own repeat spread (about 1e-4 on confidences). If the 14-step sampler amplifies small differences past that, report drift against step count rather than hide it. Where handing over a tensor is impractical (per-loop dropout masks, which are L x L x c each), the builder may fall back to a statistical comparison for that source of randomness and must say so.
- **Speed and memory:** warm, torch/MPS as production calls it (eval, no_grad, trunk-only copy, same loops, steps, samples, MSA), several sizes, repeated runs, with the roughly 10 percent noise floor stated. "Warm" means the second prediction onward in one process; Mol-1-MLX's first prediction only tied torch because worker start-up and kernel warm-up cancel the gain. The cold first-call time is measured and reported too, and the CLI accepts several inputs per run so the saving is actually collected. Both a no-MSA baseline and an MSA baseline (a real alignment of realistic depth, not a synthetic one) are built from torch. Faster at every size from 250 to 480 tokens is the bar. A speedup of about 1.5x or more is a hope based on Mol-1-MLX's trunk-only numbers, not a promise. Sizes where MLX is not faster are reported as such. Memory by hard-cap survival and a sampled watchdog against torch's driver-allocation peak.

## 8. Risks and unknowns

- Where the time goes in the torch path has not been profiled. Profiling the torch baseline per stage (trunk loops, sampler, confidence head) is the first task, before any kernel work, because it decides where Metal pays.
- MLX's linear algebra runs on the CPU stream, so the 3x3 SVD in rigid alignment will need a CPU hop or a closed-form rotation; either is small but must be checked for parity.
- The trunk runs about 960 pair blocks per fold at N^2 x 256 floats per block (N=480 is 230k pairs), so pair-tensor memory traffic, not FLOPs, may be the limit; Mol-1-MLX saw about 6 of 10 to 11 TFLOP/s for the same reason.
- Sampler stochasticity and long chains of small differences may make "as close as possible" weaker end to end than per stage; section 7 says how that gets reported.
- MSA support is required (Ethan). Mol-1-MLX's `msa.py` is a close template for the encoder blocks, but memory is the hazard: the MSA tensor is rows x N x channels, real alignments are deep, and ESMFold2 re-subsamples rows every loop. The encoder is chunked from a byte budget and tested at realistic depth and at N up to the largest size in use. Its order in the build is after the trunk and before the sampler (my call), so the speed claim can be made with and without MSA before the sampler work starts.

## 9. Calls made after the builder check (my calls, open to override)

- **Reference torch.** Stage gates use torch CPU fp32 captures. End-to-end comparison and all speed and memory baselines use torch on MPS, as production calls it. The CPU-versus-MPS gap is measured once at a small size (N of about 48 and 97) and recorded, so that an MLX-versus-MPS difference can be read against it.
- **Thresholds.** A stage blocks the next stage if its fp32 relative error against CPU references exceeds 1e-5. The composite trunk after all loops and the distogram are reported against CPU references at small N, with the observed growth with depth and loop count; they block only if they exceed 1e-3 relative or flip more than 0.5 percent of distogram bins. End to end, confidences (ipTM, pTM, mean pLDDT) are compared against the MPS repeat spread, and coordinates by aligned RMSD against the spread between two torch MPS runs of the same seed; a miss is reported with its drift against sampling steps, not treated as a stop. These numbers are guesses to be revised after the first measurements; changing a threshold is recorded in the repo, not done silently.
- **Allowed optimisations.** Fused or Metal kernels, reordered accumulation, `mx.fast` ops and a closed-form 3x3 rotation are allowed if the stage still meets its gate. Anything that cannot meet its gate is dropped, not shipped.
- **Seed.** The default CLI draws randomness with MLX's generator (no torch inside the fold loop, no transfer cost, nothing counted against the speed bar). A `--match-torch-rng` option and the parity tests draw the same tensors in torch under the worker's seed context and hand them to MLX, so a given seed can be compared across the two implementations. The CLI makes no promise that the default `--seed` reproduces torch's sample.
- **Checkpoints (each ends the work if it fails, per section 2).** Checkpoint 0, before the full port: one `PairUpdateBlock` (both triangle multiplications and the transition) in MLX with real weights, timed against the same block in torch/MPS at N of about 250 and 400, fp32, warm. The trunk runs about 960 such blocks per fold, so if this does not beat torch the port cannot, and it costs hours instead of weeks to find out. Checkpoint 1: the full trunk loop, with and without MSA, for parity and speed. Checkpoint 2: the sampler and confidence head, then the whole fold. Stage boundaries are swappable torch/MLX handoffs so a stage can be tested against torch in either direction; that is a test aid, not a shipping fallback.

## 10. Answered by Ethan

1. Speed bar missed: stop (section 2).
2. MSA support: required for done and for the speed claim (sections 2, 3, 7, 8).

## 11. Left to the builder

Module-by-module implementation, whether and where to write Metal, `mx.compile` use, chunking, the weight-file layout, test layout, the worker's request protocol, how randomness is handed over, and the SVD approach. The builder reads `_run_one_loop`, `structure_head.sample` and the confidence head directly; this document summarises them and may be wrong in detail.
