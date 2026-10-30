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


def preflight(max_disk_pct: float = 90.0, max_other_gb: float = 8.0) -> None:
    _STACK.enter_context(gpu_lock())
    usage = shutil.disk_usage("/")
    pct = 100.0 * usage.used / usage.total
    if pct >= max_disk_pct:
        raise SystemExit(f"preflight: disk is {pct:.0f}% full (limit {max_disk_pct:.0f}%)")
    heavy = _heavy_processes(max_other_gb)
    if heavy:
        raise SystemExit(f"preflight: other processes hold more than {max_other_gb} GB: {heavy}")
