"""One GPU run at a time.

``gpu_lock()`` takes an exclusive non-blocking flock on a lock file (default
``~/.cache/esmfold2_mlx/gpu.lock``, override with ``ESMFOLD2_MLX_LOCK``), writes the holder's pid
into it and exports ``ESMFOLD2_MLX_LOCK_OWNER=<pid>``. A second run fails at once, naming the
holder. The OS drops the flock when the holder dies, however it dies.

A process whose environment names a live *proper ancestor* as the owner runs under that ancestor's
lock (the CLI, worker and bench scripts a pytest session or another script launches): for it,
``gpu_lock()`` does nothing. A pid that is not an ancestor, forged or stale, grants nothing; a
process naming itself as owner (the lock holder re-entering) is not its own ancestor either.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess

PATH_ENV = "ESMFOLD2_MLX_LOCK"
OWNER_ENV = "ESMFOLD2_MLX_LOCK_OWNER"
DEFAULT_PATH = "~/.cache/esmfold2_mlx/gpu.lock"


def _lock_path() -> str:
    return os.path.expanduser(os.environ.get(PATH_ENV, DEFAULT_PATH))


def _parent_of(pid: int) -> int | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return int(out) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _is_proper_ancestor(pid: int) -> bool:
    """Whether ``pid`` is a live ancestor of this process, excluding this process itself."""
    if pid == os.getpid():
        return False
    current, seen = os.getppid(), set()
    while current and current > 1 and current not in seen:
        if current == pid:
            return True
        seen.add(current)
        current = _parent_of(current)
    return False


@contextlib.contextmanager
def gpu_lock():
    owner = os.environ.get(OWNER_ENV, "")
    if owner.isdigit() and _is_proper_ancestor(int(owner)):
        yield
        return
    path = _lock_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = os.pread(fd, 32, 0).decode(errors="replace").strip() or "unknown"
            raise SystemExit(
                f"another GPU run holds {path} (pid {holder}); only one GPU run at a time"
            )
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        previous = os.environ.get(OWNER_ENV)
        os.environ[OWNER_ENV] = str(os.getpid())
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(OWNER_ENV, None)
            else:
                os.environ[OWNER_ENV] = previous
    finally:
        os.close(fd)  # closing the descriptor releases the flock
