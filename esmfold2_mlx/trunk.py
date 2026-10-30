"""The pair trunk. This task: ``PairUpdateBlock``; the loop, shim and coda arrive in later tasks."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .layers import Transition, TriangleMultiplication
from .weights import Weights, key


class PairUpdateBlock:
    """esm ``PairUpdateBlock`` (layers.py:2601): z += tri_mul_out(z); z += tri_mul_in(z); transition.

    ``z`` is [1, N, N, 256]; ``mask`` is [1, N, N] or None (all ones). Weights live under ``prefix``
    (``blocks.{i}`` inside the ``folding_trunk`` group).
    """

    def __init__(self, w: Weights, prefix: str) -> None:
        self.tri_mul_out = TriangleMultiplication(w, key(prefix, "tri_mul_out._engine"), outgoing=True)
        self.tri_mul_in = TriangleMultiplication(w, key(prefix, "tri_mul_in._engine"), outgoing=False)
        self.pair_transition = Transition(w, key(prefix, "pair_transition"))

    def __call__(self, z: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        z = self.tri_mul_out(z, mask, residual=z)
        z = self.tri_mul_in(z, mask, residual=z)
        return self.pair_transition(z)
