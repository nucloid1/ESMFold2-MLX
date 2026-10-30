"""Every pytest session is one GPU run: it holds the project's gpu lock for its whole life.

Child processes a test starts (the converter, the MLX worker, the CLI) inherit the lock through
ESMFOLD2_MLX_LOCK_OWNER; an unrelated GPU run started meanwhile fails fast naming this session.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esmfold2_mlx.gpu_lock import gpu_lock  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _session_gpu_lock():
    with gpu_lock():
        yield
