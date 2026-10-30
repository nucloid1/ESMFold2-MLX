"""``preflight()`` for scripts and benchmarks: take the gpu lock for the life of the process (children
inherit it), then run the package's checks (esmfold2_mlx/preflight.py: disk, other heavy processes)."""

import atexit
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esmfold2_mlx.gpu_lock import gpu_lock  # noqa: E402
from esmfold2_mlx.preflight import MIN_FREE_ENV, check_disk, checks, heavy_processes  # noqa: E402,F401

_STACK = contextlib.ExitStack()
atexit.register(_STACK.close)


def preflight(max_disk_pct: float | None = None, max_other_gb: float = 8.0) -> None:
    _STACK.enter_context(gpu_lock())
    checks(max_disk_pct, max_other_gb)
