"""Weight dictionaries that record which keys were read, and the converted-weights loader.

A mistyped weight prefix must not look like "that module has no such parameter": after a module is
built, ``assert_all_used`` fails if any weight under it was never consumed (the silent-miss hazard
Mol-1-MLX hit with optional LayerNorm affines).
"""

from __future__ import annotations

import os

import mlx.core as mx


class Weights(dict):
    """A weight dict that records which keys were read through ``[]`` or ``get``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.used: set[str] = set()

    def __getitem__(self, k):
        value = super().__getitem__(k)  # a miss raises before anything is recorded
        self.used.add(k)
        return value

    def get(self, k, default=None):
        if k in self:
            self.used.add(k)
        return super().get(k, default)

    def unused(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self if k.startswith(prefix) and k not in self.used)

    def assert_all_used(self, prefix: str = "", allow: tuple = ()) -> None:
        left = [k for k in self.unused(prefix) if not any(k.startswith(a) for a in allow)]
        if left:
            raise KeyError(
                f"{len(left)} weight(s) under {prefix!r} were never used -- a mistyped key or an "
                f"unported parameter: {left[:6]}{' ...' if len(left) > 6 else ''}"
            )


def key(prefix: str, name: str) -> str:
    """Join a weight-path prefix and a name; an empty prefix adds no leading dot."""
    return f"{prefix}.{name}" if prefix else name


def load_group(weights_dir: str, group: str) -> Weights:
    """Read ``<weights_dir>/<group>.safetensors``; keys have the leading ``<group>.`` stripped."""
    path = os.path.join(weights_dir, f"{group}.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no converted weights for group {group!r} at {path}; "
            f"build them once with: python scripts/convert_weights.py"
        )
    return Weights(mx.load(path))
