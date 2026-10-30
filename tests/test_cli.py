"""The ``esmfold2-mlx pred`` CLI end to end, in a subprocess (it is the torch-side parent; the fold runs
in the MLX worker). ESMC is skipped by pre-filling the hidden-state cache with stand-in states, so
these run without ESMC-6B; the ESMC path itself is covered by the integration test."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import REPO, synthetic_lm_states

from esmfold2_mlx.inputs import parse_jobs

M49 = "CAAELAALEAELAALEGPWKGYPIPYGKLQFLIKKLKQLKVAC"
INTEGRATION = os.environ.get("ESMFOLD2_MLX_INTEGRATION") == "1"


def _job(tmp_path, jobs):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"jobs": jobs}))
    return path


def _cache(tmp_path, job_path):
    cache = tmp_path / "lm"
    cache.mkdir(exist_ok=True)
    for job in parse_jobs(str(job_path)):
        np.save(cache / f"{job.states_key()}.npy", synthetic_lm_states(job.tokens()))
    return cache


def _cli(*args, timeout=600):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, "-m", "esmfold2_mlx.cli", "pred", *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=timeout)


def test_cli_exits_nonzero_on_unsupported_input(tmp_path):
    job = _job(tmp_path, [{"name": "lig", "chains": [{"id": "A", "sequence": M49}, {"id": "L", "ccd": "ATP"}]}])
    proc = _cli("-i", job, "-o", tmp_path / "out")
    assert proc.returncode == 2
    assert "chain L" in proc.stderr and "protein complexes only" in proc.stderr


def test_cli_folds_several_jobs_in_one_process(tmp_path):
    job = _job(tmp_path, [{"name": "a", "chains": [{"id": "A", "sequence": M49}]},
                          {"name": "b", "chains": [{"id": "A", "sequence": M49[:30]}, {"id": "B", "sequence": M49[10:]}]}])
    out = tmp_path / "out"
    proc = _cli("-i", job, "-o", out, "--samples", "2", "--lm-cache", _cache(tmp_path, job))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stderr.count("worker ready") == 1
    for name in ("a", "b"):
        for s in (0, 1):
            d = out / name / f"sample_{s}"
            summary = json.loads((d / "summary.json").read_text())
            assert {"iptm", "ptm", "plddt_mean", "tokens", "seconds"} <= set(summary)
            assert (d / "complex.cif").read_text().startswith("data_")
            assert np.load(d / "pae_complex.npz")["pae"].shape[-1] == summary["tokens"]


def test_same_seed_twice_identical_outputs(tmp_path):
    job = _job(tmp_path, [{"name": "a", "chains": [{"id": "A", "sequence": M49}]}])
    cache = _cache(tmp_path, job)
    runs = []
    for k in range(2):
        out = tmp_path / f"out{k}"
        proc = _cli("-i", job, "-o", out, "--seed", "5", "--lm-cache", cache)
        assert proc.returncode == 0, proc.stderr[-2000:]
        runs.append(((out / "a" / "sample_0" / "complex.cif").read_text(),
                     json.loads((out / "a" / "sample_0" / "summary.json").read_text())))
    assert runs[0][0] == runs[1][0]
    assert {k: v for k, v in runs[0][1].items() if k != "seconds"} == {k: v for k, v in runs[1][1].items() if k != "seconds"}


def test_cli_exits_nonzero_when_worker_killed_by_cap(tmp_path):
    job = _job(tmp_path, [{"name": "a", "chains": [{"id": "A", "sequence": M49}]}])
    proc = _cli("-i", job, "-o", tmp_path / "out", "--cap-gib", "0.3", "--lm-cache", _cache(tmp_path, job), timeout=120)
    assert proc.returncode == 3
    assert "0.3 GiB" in proc.stderr


@pytest.mark.skipif(not INTEGRATION, reason="needs ESMC-6B and the GPU: set ESMFOLD2_MLX_INTEGRATION=1")
def test_esmc_runs_and_is_freed_before_the_worker_starts(tmp_path):
    job = _job(tmp_path, [{"name": "a", "chains": [{"id": "A", "sequence": M49}]}])
    proc = _cli("-i", job, "-o", tmp_path / "out", "--lm-cache", tmp_path / "empty-cache", timeout=1800)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stderr.index("esmc freed") < proc.stderr.index("worker ready")
    assert list((tmp_path / "empty-cache").glob("*.npy"))  # states were cached for the next run
