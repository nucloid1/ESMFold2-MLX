"""Checks every real-model run starts with, and the GPU lock that makes it the only one.

``preflight()`` takes ``gpu_lock()`` and keeps it for the life of the process (children inherit it),
then refuses to go on if the disk is nearly full (ESMFold2Score's README records a kernel panic from
a full disk plus load pressure) or another process is holding a lot of memory.
"""

import atexit
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esmfold2_mlx.gpu_lock import gpu_lock  # noqa: E402

_STACK = contextlib.ExitStack()
atexit.register(_STACK.close)


def _heavy_processes(max_other_gb: float) -> list[str]:
    out = subprocess.run(["ps", "-axo", "pid=,rss=,comm="], capture_output=True, text=True, timeout=10).stdout
    mine = {os.getpid(), os.getppid()}
    heavy = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and int(parts[0]) not in mine:
            gb = int(parts[1]) / 1024 / 1024  # ps reports KiB
            if gb > max_other_gb:
                heavy.append(f"pid {parts[0]} {parts[2]} {gb:.1f} GB")
    return heavy


MIN_FREE_ENV = "ESMFOLD2_MLX_MIN_FREE_GB"


def check_disk(max_disk_pct: float) -> None:
    """Fail at ``max_disk_pct`` full. ``ESMFOLD2_MLX_MIN_FREE_GB=<gb>`` is an explicit opt-in for
    light runs (approved by Ethan for Checkpoint 0 on 2026-10-29): over the limit, the run goes ahead
    only if at least that much is free, and says so on stderr. Without it the default never loosens."""
    usage = shutil.disk_usage("/")
    pct = 100.0 * usage.used / usage.total
    if pct < max_disk_pct:
        return
    free_gb = usage.free / 1e9
    floor = os.environ.get(MIN_FREE_ENV)
    if floor is None:
        raise SystemExit(f"preflight: disk is {pct:.0f}% full (limit {max_disk_pct:.0f}%)")
    if free_gb < float(floor):
        raise SystemExit(f"preflight: disk is {pct:.0f}% full with {free_gb:.0f} GB free, below {MIN_FREE_ENV}={floor}")
    print(f"preflight WARNING: disk is {pct:.0f}% full (limit {max_disk_pct:.0f}%); running on the "
          f"opt-in {MIN_FREE_ENV}={floor} with {free_gb:.0f} GB free", file=sys.stderr, flush=True)


def preflight(max_disk_pct: float = 90.0, max_other_gb: float = 8.0) -> None:
    _STACK.enter_context(gpu_lock())
    check_disk(max_disk_pct)
    heavy = _heavy_processes(max_other_gb)
    if heavy:
        raise SystemExit(f"preflight: other processes hold more than {max_other_gb} GB: {heavy}")
