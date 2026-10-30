"""Shared test helpers: relative error, liveness check, and the reference-file layout.

Reference layout (written by scripts/capture_refs.py): ``features`` and ``lm_states`` are shared by
both devices (``refs/<case>/features.npz``, ``refs/<case>/lm_states.npy``); every other stage is
``refs/<case>/<device>/<stage>.npz``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DEVICES = ("cpu", "mps")
SHARED_STAGES = {"features": "features.npz", "lm_states": "lm_states.npy"}


def rel_err(a, b) -> float:
    """``||a - b||_F / ||b||_F`` in float64; 0.0 for two zero arrays, inf for a zero reference only."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = float(np.linalg.norm(a - b))
    ref = float(np.linalg.norm(b))
    if ref == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / ref


def assert_live(x, floor: float = 1e-3) -> None:
    """A parity test is meaningless on a reference that is all zeros or non-finite."""
    arr = np.array(x, dtype=np.float64)
    assert np.all(np.isfinite(arr)), "reference contains non-finite values"
    norm = float(np.linalg.norm(arr))
    assert norm >= floor, f"near-zero reference (norm {norm:.3g} < {floor:.3g}): a match proves nothing"


def synthetic_lm_states(n_tokens: int, seed: int = 1234) -> np.ndarray:
    """Stand-in for ESMC-6B hidden states, [1, N, 81, 2560] float32, reproducible from the seed.

    Light captures feed these through the real torch modules so the language-model path is ported
    without loading ESMC; real ESMC states replace them when Task 2's corpus exists.
    """
    return np.random.default_rng(seed).standard_normal((1, n_tokens, 81, 2560), dtype=np.float32)


def refs_root() -> Path:
    return Path(os.environ.get("ESMFOLD2_MLX_REFS", REPO / "refs"))


def ref_path(case: str, stage: str, device: str = "cpu") -> Path:
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if stage in SHARED_STAGES:
        return refs_root() / case / SHARED_STAGES[stage]
    return refs_root() / case / device / f"{stage}.npz"


def load_ref(case: str, stage: str, device: str = "cpu") -> dict[str, np.ndarray]:
    path = ref_path(case, stage, device)
    if not path.exists():
        message = (
            f"missing reference {path}; run: PYTHONPATH=. {sys.executable} "
            f"scripts/capture_refs.py {case} --device {device} --stages {stage}"
        )
        if os.environ.get("ESMFOLD2_MLX_ALLOW_MISSING_REFS") == "1":
            pytest.skip(message)
        pytest.fail(message, pytrace=False)
    if path.suffix == ".npy":
        return {stage: np.load(path)}
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}
