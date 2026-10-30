"""The MLX worker process: same result as in-process, reused across calls, never imports torch,
fails loudly (never silently falls back) and leaves no shared memory behind."""

import os
import signal
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from helpers import load_ref, rel_err, synthetic_lm_states

from esmfold2_mlx.config import FoldOptions
from esmfold2_mlx.fold import MLXFold
from esmfold2_mlx.rng import ReplayDraws
from esmfold2_mlx.worker import RemoteFold, WorkerError


@pytest.fixture(scope="module")
def remote(weights_dir):
    worker = RemoteFold(weights_dir)
    yield worker
    worker.close()


def _m49():
    f = load_ref("m49", "features")
    return f, synthetic_lm_states(f["res_type"].shape[1]), dict(load_ref("m49", "fold_draws"))


def test_worker_matches_in_process(remote, weights_dir):
    f, lm, draws = _m49()
    options = FoldOptions(num_diffusion_samples=2)
    got = remote(f, lm, options, torch_draws=draws)
    want = MLXFold(weights_dir).fold(f, lm, options, draws=ReplayDraws(draws))
    assert set(got) == set(want)
    for k in ("distogram_logits", "sample_atom_coords", "plddt", "pae"):
        assert rel_err(got[k], want[k]) < 1e-5, k


def test_worker_is_reused_across_calls(remote):
    f, lm, _ = _m49()
    remote(f, lm, FoldOptions())
    pid = remote.pid
    remote(f, lm, FoldOptions(seed=1))
    assert remote.pid == pid and remote.last_seconds > 0


def test_worker_never_imports_torch(remote):
    remote.start()
    assert remote.ready_info["torch_imported"] is False


def test_dead_worker_raises_then_restarts(weights_dir):
    worker = RemoteFold(weights_dir)
    try:
        f, lm, _ = _m49()
        worker.start()
        os.kill(worker.pid, signal.SIGKILL)
        time.sleep(0.5)
        with pytest.raises(WorkerError, match="exit code"):
            worker(f, lm, FoldOptions())
        out = worker(f, lm, FoldOptions())  # restarts on the next call
        assert np.isfinite(out["sample_atom_coords"]).all()
    finally:
        worker.close()


def test_cap_exceeded_error_names_cap(weights_dir):
    f = load_ref("vegf2", "features")
    lm = synthetic_lm_states(f["res_type"].shape[1])
    worker = RemoteFold(weights_dir, cap_gib=0.5)
    try:
        with pytest.raises(WorkerError, match="0.5 GiB"):
            worker(f, lm, FoldOptions())
    finally:
        worker.close()


def test_shared_memory_unlinked_after_success_and_error(weights_dir, remote):
    f, lm, _ = _m49()
    remote(f, lm, FoldOptions())
    names = list(remote.last_shm_names)
    worker = RemoteFold(weights_dir, cap_gib=0.5)
    try:
        with pytest.raises(WorkerError):
            worker(load_ref("vegf2", "features"), synthetic_lm_states(247), FoldOptions())
        names += worker.last_shm_names
    finally:
        worker.close()
    assert names
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)
