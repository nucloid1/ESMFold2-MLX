"""The disk check in scripts/preflight.py: 90% by default, an explicit opt-in floor of free GB otherwise."""

import shutil
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import preflight  # noqa: E402

Usage = namedtuple("Usage", "total used free")
GB = 1e9


def _disk(monkeypatch, used_pct: float, free_gb: float):
    total = free_gb * GB / (1 - used_pct / 100)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(total, total - free_gb * GB, free_gb * GB))


def test_under_the_limit_passes_silently(monkeypatch, capsys):
    monkeypatch.delenv("ESMFOLD2_MLX_MIN_FREE_GB", raising=False)
    _disk(monkeypatch, 80.0, 200.0)
    preflight.check_disk(90.0)
    assert capsys.readouterr().err == ""


def test_over_the_limit_fails_without_the_override(monkeypatch):
    monkeypatch.delenv("ESMFOLD2_MLX_MIN_FREE_GB", raising=False)
    _disk(monkeypatch, 94.0, 61.0)
    with pytest.raises(SystemExit, match="94% full"):
        preflight.check_disk(90.0)


def test_override_passes_with_enough_free_space_and_warns(monkeypatch, capsys):
    monkeypatch.setenv("ESMFOLD2_MLX_MIN_FREE_GB", "50")
    _disk(monkeypatch, 94.0, 61.0)
    preflight.check_disk(90.0)
    err = capsys.readouterr().err
    assert "WARNING" in err and "ESMFOLD2_MLX_MIN_FREE_GB=50" in err and "94%" in err


def test_override_fails_when_free_space_is_below_it(monkeypatch):
    monkeypatch.setenv("ESMFOLD2_MLX_MIN_FREE_GB", "50")
    _disk(monkeypatch, 96.0, 40.0)
    with pytest.raises(SystemExit, match="40 GB free"):
        preflight.check_disk(90.0)
