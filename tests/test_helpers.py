"""Tests for the shared test helpers (tests/helpers.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import assert_live, load_ref, ref_path, rel_err

REPO = Path(__file__).resolve().parents[1]


def test_rel_err_known_pair():
    b = np.array([3.0, 4.0])  # ||b|| = 5
    a = b + np.array([0.3, 0.4])  # ||a - b|| = 0.5
    assert rel_err(a, b) == pytest.approx(0.1, rel=1e-12)


def test_rel_err_divides_by_the_reference_not_the_candidate():
    a, b = np.array([2.0, 0.0]), np.array([1.0, 0.0])
    assert rel_err(a, b) == pytest.approx(1.0)
    assert rel_err(b, a) == pytest.approx(0.5)


def test_rel_err_accepts_mlx_arrays_and_works_in_float64():
    import mlx.core as mx

    b = np.full((4, 4), 1.0, dtype=np.float32)
    a = b + np.float32(1e-3)
    assert rel_err(mx.array(a), b) == pytest.approx(1e-3, rel=1e-4)
    # A float32 norm over a million elements is only good to ~1e-7; a float64 one matches the
    # float64 reference to rounding, so this tight tolerance tells the two apart.
    rng = np.random.default_rng(0)
    big_b = rng.standard_normal(1 << 20).astype(np.float32)
    big_a = big_b + rng.standard_normal(1 << 20).astype(np.float32) * np.float32(1e-3)
    a64, b64 = big_a.astype(np.float64), big_b.astype(np.float64)
    expected = np.linalg.norm(a64 - b64) / np.linalg.norm(b64)
    assert rel_err(big_a, big_b) == pytest.approx(expected, rel=1e-12)


def test_rel_err_zero_reference_is_infinite_unless_equal():
    assert rel_err(np.zeros(3), np.zeros(3)) == 0.0
    assert rel_err(np.ones(3), np.zeros(3)) == float("inf")


def test_rel_err_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        rel_err(np.zeros((2, 3)), np.zeros((3, 2)))


def test_assert_live_raises_below_the_floor():
    with pytest.raises(AssertionError, match="near-zero"):
        assert_live(np.full((10,), 1e-5))  # norm ~3e-5 < 1e-3
    with pytest.raises(AssertionError, match="near-zero"):
        assert_live(np.zeros((4, 4)))


def test_assert_live_passes_above_the_floor_and_honours_a_custom_floor():
    assert_live(np.ones((4,)))
    assert_live(np.full((10,), 1e-5), floor=1e-6)
    with pytest.raises(AssertionError):
        assert_live(np.ones((4,)), floor=10.0)


def test_assert_live_rejects_non_finite():
    with pytest.raises(AssertionError):
        assert_live(np.array([np.nan, 1.0]))


def test_ref_path_layout():
    root = REPO / "refs"
    # Shared across devices.
    assert ref_path("m49", "features") == root / "m49" / "features.npz"
    assert ref_path("m49", "features", "mps") == root / "m49" / "features.npz"
    assert ref_path("m49", "lm_states") == root / "m49" / "lm_states.npy"
    assert ref_path("m49", "lm_states", "mps") == root / "m49" / "lm_states.npy"
    # Everything else is per device, cpu by default.
    assert ref_path("vegf1", "pair_block") == root / "vegf1" / "cpu" / "pair_block.npz"
    assert ref_path("vegf1", "pair_block", "mps") == root / "vegf1" / "mps" / "pair_block.npz"
    assert ref_path("vegf1", "draws", "cpu") == root / "vegf1" / "cpu" / "draws.npz"
    assert ref_path("vegf1", "torch_out", "mps") == root / "vegf1" / "mps" / "torch_out.npz"


def test_ref_path_is_anchored_to_the_repo_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("ESMFOLD2_MLX_REFS", raising=False)
    monkeypatch.chdir(tmp_path)
    assert ref_path("m49", "features").is_absolute()
    assert ref_path("m49", "features") == REPO / "refs" / "m49" / "features.npz"


def test_ref_path_rejects_unknown_device():
    with pytest.raises(ValueError, match="device"):
        ref_path("m49", "pair_block", "cuda")


def test_load_ref_missing_fails_with_the_exact_capture_command(tmp_path, monkeypatch):
    monkeypatch.setenv("ESMFOLD2_MLX_REFS", str(tmp_path))
    monkeypatch.delenv("ESMFOLD2_MLX_ALLOW_MISSING_REFS", raising=False)
    with pytest.raises(pytest.fail.Exception) as info:
        load_ref("m49", "pair_block", "mps")
    expected = (
        f"run: PYTHONPATH=. {sys.executable} scripts/capture_refs.py m49 "
        "--device mps --stages pair_block"
    )
    assert expected in str(info.value)


def test_load_ref_missing_shared_stage_names_the_command_with_the_device(tmp_path, monkeypatch):
    monkeypatch.setenv("ESMFOLD2_MLX_REFS", str(tmp_path))
    monkeypatch.delenv("ESMFOLD2_MLX_ALLOW_MISSING_REFS", raising=False)
    with pytest.raises(pytest.fail.Exception) as info:
        load_ref("m49", "lm_states")
    assert "scripts/capture_refs.py m49 --device cpu --stages lm_states" in str(info.value)


@pytest.mark.parametrize("value", ["1"])
def test_load_ref_missing_is_skipped_when_allowed(tmp_path, monkeypatch, value):
    monkeypatch.setenv("ESMFOLD2_MLX_REFS", str(tmp_path))
    monkeypatch.setenv("ESMFOLD2_MLX_ALLOW_MISSING_REFS", value)
    with pytest.raises(pytest.skip.Exception) as info:
        load_ref("m49", "pair_block")
    assert "scripts/capture_refs.py m49 --device cpu --stages pair_block" in str(info.value)


def test_load_ref_other_values_of_the_escape_hatch_do_not_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("ESMFOLD2_MLX_REFS", str(tmp_path))
    monkeypatch.setenv("ESMFOLD2_MLX_ALLOW_MISSING_REFS", "0")
    with pytest.raises(pytest.fail.Exception):
        load_ref("m49", "pair_block")


@pytest.mark.parametrize("allow", [None, "1"])
def test_load_ref_returns_arrays_when_present(tmp_path, monkeypatch, allow):
    monkeypatch.setenv("ESMFOLD2_MLX_REFS", str(tmp_path))
    if allow is None:
        monkeypatch.delenv("ESMFOLD2_MLX_ALLOW_MISSING_REFS", raising=False)
    else:
        monkeypatch.setenv("ESMFOLD2_MLX_ALLOW_MISSING_REFS", allow)
    z = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = ref_path("c", "pair_block", "cpu")
    path.parent.mkdir(parents=True)
    np.savez(path, z=z, mask=np.ones((2,), dtype=bool))
    got = load_ref("c", "pair_block", "cpu")
    assert sorted(got) == ["mask", "z"]
    np.testing.assert_array_equal(got["z"], z)
    assert isinstance(got["z"], np.ndarray)

    lm = np.ones((1, 5, 3), dtype=np.float32)
    lm_path = ref_path("c", "lm_states")
    np.save(lm_path, lm)
    got = load_ref("c", "lm_states")
    assert list(got) == ["lm_states"]
    np.testing.assert_array_equal(got["lm_states"], lm)
