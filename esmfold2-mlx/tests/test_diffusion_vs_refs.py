"""Diffusion module and sampler against torch captures (stage ``sampler``).

CPU: the real forward (synthetic ESMC states, 14 steps, 2 samples) records the sampler's inputs, every
denoiser call and every random draw. MPS: the same sampler inputs and the SAME recorded draws are
replayed through torch on MPS (scripts/torch_model.DrawReplayer), and the CPU-recorded denoiser calls
are re-run on MPS, so CPU, MPS and MLX are compared on identical inputs and randomness.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from helpers import REPO, assert_live, kabsch_rmsd, load_ref, rel_err

from esmfold2_mlx.config import load_config
from esmfold2_mlx.diffusion import DiffusionModule, DiffusionSampler, noise_schedule, weighted_rigid_align
from esmfold2_mlx.rng import ReplayDraws

CASES = ["m49", "vegf1"]
GATE = 1e-5
SAMPLE_KEYS = ("z_trunk", "s_inputs", "relative_position_encoding", "ref_pos", "ref_charge", "ref_mask", "ref_element",
               "ref_atom_name_chars", "ref_space_uid", "tok_idx", "asym_id", "residue_index", "entity_id", "token_index",
               "sym_id", "token_attention_mask")


@pytest.fixture(scope="module")
def cfg(weights_dir):
    return load_config(weights_dir)


@pytest.fixture(scope="module")
def module(weights_dir, cfg):
    return DiffusionModule(weights_dir, cfg)


@pytest.fixture(scope="module")
def sampler(weights_dir, cfg):
    return DiffusionSampler(weights_dir, cfg)


def _sample_inputs(ref):
    return {k: mx.array(ref[f"in_{k}"]) for k in SAMPLE_KEYS}


def _call(ref, i):
    keys = sorted(k[len(f"call{i}_"):] for k in ref if k.startswith(f"call{i}_") and k != f"call{i}_out")
    kw = {k: mx.array(ref[f"call{i}_{k}"]) for k in keys}
    kw["num_diffusion_samples"] = int(ref["num_diffusion_samples"])
    return kw


@pytest.mark.parametrize("case", CASES)
def test_noise_schedule_reproduces_torch_t_hats(cfg, case):
    ref = load_ref(case, "sampler")
    sched = noise_schedule(cfg, steps=None, max_sigma=256.0)
    gam = [cfg.structure_head.gamma_0 if s > cfg.structure_head.gamma_min else 0.0 for s in sched]
    t_hats = [float(sched[i]) * (1.0 + gam[i + 1]) for i in range(len(sched) - 1)]
    np.testing.assert_allclose(t_hats, ref["t_hats"], rtol=1e-6)


@pytest.mark.parametrize("call", ["first", "last"])
@pytest.mark.parametrize("case", CASES)
def test_denoise_call_matches_cpu_and_mps(module, case, call):
    cpu, mps = load_ref(case, "sampler"), load_ref(case, "sampler", "mps")
    i = 0 if call == "first" else int(cpu["n_calls"]) - 1
    assert_live(cpu[f"call{i}_out"])
    got = np.array(module(**_call(cpu, i)))
    spread = rel_err(mps[f"replay{i}_out"], cpu[f"call{i}_out"])  # torch's own CPU/MPS disagreement
    assert rel_err(got, cpu[f"call{i}_out"]) < max(GATE, 2.0 * spread)
    # MLX must sit closer to torch on MPS (production) than torch on CPU does. Measured 2026-10-29: at
    # t_hat = 460.8 torch CPU/MPS differ by 3-5e-4 and MLX/MPS by 5-9e-5; at the last call all ~1e-6.
    assert rel_err(got, mps[f"replay{i}_out"]) < max(GATE, spread)


def test_rigid_align_matches_numpy_kabsch():
    rng = np.random.default_rng(0)
    for trial in range(6):
        x = rng.standard_normal((2, 50, 3)).astype(np.float32) * 10
        if trial == 1:  # a reflection: the best proper rotation needs the det correction
            gt = x * np.array([1, 1, -1], np.float32)
        elif trial == 2:  # near-degenerate: nearly collinear points
            gt = (np.linspace(0, 1, 50)[None, :, None] * np.array([1, 2, 3])[None, None]).astype(np.float32).repeat(2, 0)
            gt += rng.standard_normal(gt.shape).astype(np.float32) * 1e-3
        else:
            gt = x + rng.standard_normal(x.shape).astype(np.float32)
        mask = np.ones((2, 50), np.float32)
        mask[1, 40:] = 0
        got = np.array(weighted_rigid_align(mx.array(x), mx.array(gt), mx.array(mask), mx.array(mask)))
        want = _numpy_align(x, gt, mask)
        # Trial 2 is ill-conditioned on purpose (the rotation about the near-line is undetermined); a
        # float32 SVD, torch's included, is only good to ~1e-4 there (measured 6.5e-5).
        assert rel_err(got, want) < (1e-3 if trial == 2 else 1e-5), trial


def _numpy_align(x, gt, mask):
    out = []
    for b in range(x.shape[0]):
        w = mask[b][:, None].astype(np.float64)
        mu, mu_gt = (x[b] * w).sum(0) / w.sum(), (gt[b] * w).sum(0) / w.sum()
        xc, gc = x[b] - mu, gt[b] - mu_gt
        u, _, vt = np.linalg.svd((w * gc).T @ xc)
        r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        out.append(xc @ r.T + mu_gt)
    return np.stack(out)


@pytest.mark.parametrize("case", CASES)
def test_sampler_trace_tracks_refs(sampler, case):
    cpu, mps = load_ref(case, "sampler"), load_ref(case, "sampler", "mps")
    trace = []
    got = sampler.sample(**_sample_inputs(cpu), num_diffusion_samples=int(cpu["num_diffusion_samples"]),
                         draws=ReplayDraws(load_ref(case, "sampler_draws")), on_step=lambda k, x: trace.append(np.array(x)))
    mask = cpu["in_ref_mask"][0].astype(bool)
    rows, final = [], {}
    for k in range(len(trace)):
        for s in range(trace[k].shape[0]):
            r_cpu = kabsch_rmsd(trace[k][s], cpu["denoised"][k][s], mask)
            r_mps = kabsch_rmsd(trace[k][s], mps["denoised"][k][s], mask)
            r_dev = kabsch_rmsd(mps["denoised"][k][s], cpu["denoised"][k][s], mask)
            rows.append(f"| {case} | {k} | {s} | {r_cpu:.4f} | {r_mps:.4f} | {r_dev:.4f} |")
    g = np.array(got)
    for s in range(g.shape[0]):
        final[s] = (kabsch_rmsd(g[s], cpu["out"][s], mask), kabsch_rmsd(g[s], mps["out"][s], mask),
                    kabsch_rmsd(mps["out"][s], cpu["out"][s], mask))
        rows.append(f"| {case} | final x | {s} | {final[s][0]:.4f} | {final[s][1]:.4f} | {final[s][2]:.4f} |")
    _write_drift(case, rows)
    for s, (vs_cpu, vs_mps, dev) in final.items():
        assert vs_cpu <= max(0.5, 2 * dev), (s, vs_cpu, dev)
        assert vs_mps <= max(0.05, 2 * dev), (s, vs_mps, dev)


def test_multi_sample_batch_matches_single(sampler):
    """Three samples in one batch equal three single-sample runs given the same per-sample draws, and
    stay distinct from each other: no sample leaks into another. Batched and single matmuls round the
    bf16 atom attention slightly differently, so the match is to bf16 noise (measured 0.0036 A, below
    torch's own CPU/MPS spread of 0.005-0.007 A), not bitwise; a leak would be angstroms."""
    ref = load_ref("m49", "sampler")
    inputs = _sample_inputs(ref)
    A = inputs["ref_pos"].shape[1]
    steps = len(ref["t_hats"])
    rng = np.random.default_rng(3)
    init = rng.standard_normal((3, A, 3)).astype(np.float32)
    quats = rng.standard_normal((steps, 3, 4)).astype(np.float32)
    trans = rng.standard_normal((steps, 3, 1, 3)).astype(np.float32)
    noise = rng.standard_normal((steps, 3, A, 3)).astype(np.float32)

    def draws_for(idx):
        arrays = {"sample:randn/0": init[idx]}
        for k in range(steps):
            arrays[f"_random_rotations:randn/{k}"] = quats[k][idx]
            arrays[f"_center_random_augmentation:randn_like/{k}"] = trans[k][idx]
            arrays[f"sample:randn_like/{k}"] = noise[k][idx]
        return ReplayDraws(arrays)

    batch = np.array(sampler.sample(**inputs, num_diffusion_samples=3, draws=draws_for(slice(0, 3))))
    every = np.ones(A, bool)
    for i in range(3):
        single = np.array(sampler.sample(**inputs, num_diffusion_samples=1, draws=draws_for(slice(i, i + 1))))
        assert kabsch_rmsd(single[0], batch[i], every) < 0.02, i
    assert min(kabsch_rmsd(batch[i], batch[j], every) for i, j in ((0, 1), (0, 2), (1, 2))) > 0.5


def test_diffusion_weights_consumed(module):
    assert module.weights.unused() == []


def _write_drift(case: str, rows: list[str]) -> None:
    path = Path(REPO) / "docs" / "parity" / "sampler_drift.md"
    header = ("# Sampler: aligned RMSD (angstrom) per step, MLX vs torch CPU and MPS on identical inputs and draws\n\n"
              "Denoised coordinates after each step (and the final x). `torch MPS vs CPU` is torch's own device spread. "
              "Written by tests/test_diffusion_vs_refs.py.\n\n"
              "| case | step | sample | MLX vs CPU | MLX vs MPS | torch MPS vs CPU |\n| --- | --- | --- | --- | --- | --- |\n")
    old = path.read_text() if path.exists() else header
    kept = [line for line in old.splitlines() if not line.startswith(f"| {case} |")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(kept + rows) + "\n")
