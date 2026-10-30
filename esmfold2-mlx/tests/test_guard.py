"""The process memory watchdog kills a runaway allocation instead of letting the machine swap to death."""

import subprocess
import sys

from helpers import REPO

PROBE = """
import sys
sys.path.insert(0, {repo!r})
import mlx.core as mx
from esmfold2_mlx.guard import memory_guard
with memory_guard(max_gib=1.0, interval_s=0.05):
    held = []
    for _ in range(40):  # 40 x 256 MiB, far past the 1 GiB cap
        held.append(mx.ones((64, 1024, 1024)))
        mx.eval(held[-1])
        import time; time.sleep(0.05)
print("survived")
"""


def test_guard_stops_the_process_over_its_cap():
    """Either MLX's own limit (set just under the cap) raises, or the watchdog kills; both stop the run."""
    proc = subprocess.run([sys.executable, "-c", PROBE.format(repo=str(REPO))], capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "survived" not in proc.stdout
    assert "cap 1.0 GiB" in proc.stderr or "memory" in proc.stderr.lower(), proc.stderr[-500:]


def test_guard_lets_a_small_run_finish():
    code = PROBE.format(repo=str(REPO)).replace("range(40)", "range(1)").replace("(64, 1024, 1024)", "(16, 1024, 1024)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "survived" in proc.stdout
