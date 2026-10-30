"""--match-torch-rng: the parent reproduces, without running the model, every random draw the torch
forward makes, in torch's order, so an MLX fold can consume torch's exact randomness for a seed."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from helpers import REPO, kabsch_rmsd, load_ref, synthetic_lm_states

from esmfold2_mlx.config import FoldOptions, load_config
from esmfold2_mlx.inputs import parse_jobs


@pytest.mark.parametrize("case", ["m49", "tnf60_msa"])
def test_torch_stream_equals_the_recorded_stream(weights_dir, case):
    """The capture recorded torch-CPU's draws under the same seed context; the regenerated stream must
    have the same keys and be bitwise equal (single-sequence and deep-MSA cases: dropout, pair init,
    column mask, row subsampling, sampler)."""
    from esmfold2_mlx.torch_draws import draw_torch_stream

    recorded = load_ref(case, "fold_draws")
    got = draw_torch_stream(load_ref(case, "features"), FoldOptions(num_diffusion_samples=2), load_config(weights_dir),
                            seed=0, device="cpu", has_lm=True)
    assert set(got) == set(recorded)
    for k in recorded:
        assert np.array_equal(got[k], recorded[k]), k


def test_cli_match_torch_rng_reproduces_the_torch_cpu_fold(tmp_path):
    """vegf1 through the CLI with torch-CPU randomness for seed 0 lands on the torch-CPU fold of the
    capture (same features, same stand-in ESMC states, same draws): inside torch's own device spread."""
    m49 = "CAAELAALEAELAALEGPWKGYPIPYGKLQFLIKKLKQLKVAC"
    vegf_target = json.loads((REPO / "data" / "cases.json").read_text())["cases"]["VEGF_9KKU"]["target_seq"]
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"jobs": [{"name": "vegf1", "chains": [{"id": "A", "sequence": m49}, {"id": "B", "sequence": vegf_target}]}]}))
    cache = tmp_path / "lm"
    cache.mkdir()
    for j in parse_jobs(str(job)):
        np.save(cache / f"{j.states_key()}.npy", synthetic_lm_states(j.tokens()))
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-m", "esmfold2_mlx.cli", "pred", "-i", str(job), "-o", str(out), "--samples", "2",
                           "--seed", "0", "--match-torch-rng", "--torch-rng-device", "cpu", "--lm-cache", str(cache)],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    torch_cpu = load_ref("vegf1", "fold")
    for s in range(2):
        summary = json.loads((out / "vegf1" / f"sample_{s}" / "summary.json").read_text())
        assert abs(summary["ptm"] - float(torch_cpu["ptm"][s])) < 2e-3
        assert abs(summary["iptm"] - float(torch_cpu["iptm"][s])) < 2e-3
    coords = np.load(out / "vegf1" / "coords.npz")["sample_atom_coords"]
    mask = load_ref("vegf1", "features")["atom_attention_mask"][0].astype(bool)
    for s in range(2):
        assert kabsch_rmsd(coords[s], torch_cpu["sample_atom_coords"][s], mask) < 0.05
