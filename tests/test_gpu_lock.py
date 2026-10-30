"""gpu_lock: one GPU run at a time, fail fast naming the holder, descendants run under the parent's lock.

Every contender here is started with ESMFOLD2_MLX_LOCK_OWNER removed from its environment (the
pytest session fixture exports it, and a process that inherits it from a live ancestor is a
legitimate descendant, not a contender) and ESMFOLD2_MLX_LOCK pointed at a private lock file so
these tests never touch the session lock.
"""

import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from esmfold2_mlx.gpu_lock import gpu_lock

REPO = Path(__file__).resolve().parents[1]
OWNER = "ESMFOLD2_MLX_LOCK_OWNER"

TRY = (
    "from esmfold2_mlx.gpu_lock import gpu_lock\n"
    "with gpu_lock():\n"
    "    print('acquired', flush=True)\n"
)
HOLD = (
    "import time\n"
    "from esmfold2_mlx.gpu_lock import gpu_lock\n"
    "with gpu_lock():\n"
    "    print('ready', flush=True)\n"
    "    time.sleep(120)\n"
)


def _env(lock_path, owner=None, keep_owner=False):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["ESMFOLD2_MLX_LOCK"] = str(lock_path)
    if not keep_owner:
        env.pop(OWNER, None)
    if owner is not None:
        env[OWNER] = str(owner)
    return env


def _run(script, env, timeout=10):
    return subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=REPO,
        capture_output=True, text=True, timeout=timeout,
    )


@pytest.fixture
def lock_path(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("ESMFOLD2_MLX_LOCK", str(path))
    return path


def _start_holder(lock_path):
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLD], env=_env(lock_path), cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    ready, _, _ = select.select([proc.stdout], [], [], 15)
    line = proc.stdout.readline() if ready else ""
    if line.strip() != "ready":
        proc.kill()
        proc.wait()
        pytest.fail(f"holder never reported ready: {line!r} {proc.stderr.read()!r}")
    return proc


def test_second_holder_fails_fast_naming_pid(lock_path):
    with gpu_lock():
        start = time.monotonic()
        res = _run(TRY, _env(lock_path))
        elapsed = time.monotonic() - start
    assert res.returncode != 0
    assert elapsed < 5
    assert str(os.getpid()) in res.stderr
    assert "acquired" not in res.stdout


def test_lock_released_when_holder_is_killed(lock_path):
    holder = _start_holder(lock_path)
    try:
        blocked = _run(TRY, _env(lock_path))
        assert blocked.returncode != 0
        assert str(holder.pid) in blocked.stderr
        holder.kill()  # SIGKILL: no cleanup code runs, the OS must release the flock
        holder.wait(timeout=5)
        res = _run(TRY, _env(lock_path))
        assert res.returncode == 0, res.stderr
        assert "acquired" in res.stdout
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()
        holder.stdout.close()
        holder.stderr.close()


def test_descendant_inherits_lock_but_stranger_does_not(lock_path):
    with gpu_lock():
        # A child (and a grandchild) of the holder keeps the owner variable and runs under the lock.
        child = _run(TRY, _env(lock_path, keep_owner=True))
        assert child.returncode == 0, child.stderr
        assert "acquired" in child.stdout
        grandchild_script = (
            "import subprocess, sys\n"
            f"r = subprocess.run([sys.executable, '-c', {TRY!r}], capture_output=True, text=True)\n"
            "sys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)\n"
        )
        grandchild = _run(grandchild_script, _env(lock_path, keep_owner=True))
        assert grandchild.returncode == 0, grandchild.stderr
        assert "acquired" in grandchild.stdout
        # The same process with the owner variable stripped is a stranger and fails, naming the holder.
        stranger = _run(TRY, _env(lock_path))
        assert stranger.returncode != 0
        assert str(os.getpid()) in stranger.stderr


def test_owner_variable_naming_a_non_ancestor_does_not_grant_the_lock(lock_path):
    holder = _start_holder(lock_path)
    try:
        # Forged: names the live holder, but the holder is not this process's ancestor.
        forged = _run(TRY, _env(lock_path, owner=holder.pid))
        assert forged.returncode != 0
        assert str(holder.pid) in forged.stderr
        # Stale: names a pid that does not exist.
        stale = _run(TRY, _env(lock_path, owner=2**22 + 12345))
        assert stale.returncode != 0
        assert str(holder.pid) in stale.stderr
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()
        holder.stderr.close()


def test_normal_exit_releases_the_lock_and_restores_the_owner_variable(lock_path):
    before = os.environ.get(OWNER)
    with gpu_lock():
        assert os.environ[OWNER] == str(os.getpid())
    assert os.environ.get(OWNER) == before
    res = _run(TRY, _env(lock_path))
    assert res.returncode == 0, res.stderr


def test_lock_file_is_created_under_a_missing_directory(tmp_path, monkeypatch):
    path = tmp_path / "deep" / "er" / "gpu.lock"
    monkeypatch.setenv("ESMFOLD2_MLX_LOCK", str(path))
    with gpu_lock():
        assert path.read_text().strip() == str(os.getpid())
