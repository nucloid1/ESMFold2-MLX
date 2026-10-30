"""Every random draw a fold makes, behind one interface.

One method per draw site in the torch source; ``SITES`` names each site the way
scripts/torch_model.DrawRecorder records it, ``<enclosing function>:<torch function>``:

- ``pair_init(shape, std)``                 model.py ``_init_pair_state``: normal truncated at
                                            +-3 std, std = sqrt(2 / (5 * c)).
- ``lm_dropout_keep(loop, shape, p)``       the per-loop ``F.dropout`` of the LM pair in
                                            ``_run_one_loop``: a bool keep mask (P(keep) = 1 - p).
- ``msa_column_keep(shape, rate)``          layers.py ``maybe_apply_msa_column_masking``.
- ``msa_row_indices(loop, depth, max_depth)`` layers.py ``maybe_subsample_msa``.
- ``diffusion_init(shape)``, ``rotation_quat(step, n)``, ``translation(step, shape)``,
  ``step_noise(step, shape)``               the sampler (``DiffusionStructureHead.sample``).

``ReplayDraws`` hands back what a torch run recorded, in call order, so an MLX run consumes the same
randomness; ``MLXDraws`` is the default for real folds. Sites whose task has not landed yet raise
``NotImplementedError`` in ``MLXDraws``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Protocol

import mlx.core as mx
import numpy as np

SITES = {
    "pair_init": "_init_pair_state:trunc_normal_",
    "lm_dropout_keep": "_run_one_loop:dropout",
    "msa_column_keep": "maybe_apply_msa_column_masking:rand",
    "msa_row_indices": "maybe_subsample_msa:randperm",
    "diffusion_init": "sample:randn",
    "rotation_quat": "_random_rotations:randn",
    "translation": "_center_random_augmentation:randn_like",
    "step_noise": "sample:randn_like",
}


def rows_from_perm(perm: mx.array, max_depth: int) -> mx.array:
    """``maybe_subsample_msa``: the query row plus ``max_depth - 1`` permuted rows, in sorted order."""
    picked = perm[: max_depth - 1].astype(mx.int32) + 1
    return mx.sort(mx.concatenate([mx.zeros((1,), dtype=mx.int32), picked]))


class Draws(Protocol):
    def pair_init(self, shape: tuple, std: float) -> mx.array: ...
    def lm_dropout_keep(self, loop: int, shape: tuple, p: float) -> mx.array: ...
    def msa_column_keep(self, shape: tuple, rate: float) -> mx.array: ...
    def msa_row_indices(self, loop: int, depth: int, max_depth: int) -> mx.array: ...
    def diffusion_init(self, shape: tuple) -> mx.array: ...
    def rotation_quat(self, step: int, n: int) -> mx.array: ...
    def translation(self, step: int, shape: tuple) -> mx.array: ...
    def step_noise(self, step: int, shape: tuple) -> mx.array: ...


class ReplayDraws:
    """Recorded torch draws (keys ``<site>/<call index>``), returned in call order per site."""

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self._arrays = arrays
        self._next = defaultdict(int)

    def _take(self, method: str, shape: tuple | None = None) -> mx.array:
        site = SITES[method]
        index = self._next[site]
        name = f"{site}/{index}"
        if name not in self._arrays:
            raise IndexError(f"no recorded draw {name!r}: the torch run made only {index} call(s) at site {site!r}")
        value = np.asarray(self._arrays[name])
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"recorded {name!r} has shape {value.shape}, asked for {tuple(shape)}")
        self._next[site] = index + 1
        return mx.array(value)

    def pair_init(self, shape, std):
        return self._take("pair_init", shape)

    def lm_dropout_keep(self, loop, shape, p):
        return self._take("lm_dropout_keep", shape)

    def msa_column_keep(self, shape, rate):
        return self._take("msa_column_keep", shape) >= rate  # torch: rand(B, L) >= rate

    def msa_row_indices(self, loop, depth, max_depth):
        return rows_from_perm(self._take("msa_row_indices", (depth - 1,)), max_depth)

    def diffusion_init(self, shape):
        return self._take("diffusion_init", shape)

    def rotation_quat(self, step, n):
        return self._take("rotation_quat", (n, 4))

    def translation(self, step, shape):
        return self._take("translation", shape)

    def step_noise(self, step, shape):
        return self._take("step_noise", shape)


class MLXDraws:
    """MLX's own generator: same seed, same draws, in the same call order."""

    def __init__(self, seed: int) -> None:
        self._key = mx.random.key(seed)

    def _split(self) -> mx.array:
        self._key, sub = mx.random.split(self._key)
        return sub

    def pair_init(self, shape, std):
        return mx.random.truncated_normal(-3.0, 3.0, shape, key=self._split()) * std

    def lm_dropout_keep(self, loop, shape, p):
        return mx.random.bernoulli(1.0 - p, shape, key=self._split())

    def msa_column_keep(self, shape, rate):
        return mx.random.uniform(shape=shape, key=self._split()) >= rate

    def msa_row_indices(self, loop, depth, max_depth):
        return rows_from_perm(mx.random.permutation(depth - 1, key=self._split()), max_depth)

    def diffusion_init(self, shape):
        raise NotImplementedError("sampler draws land with the diffusion sampler (Task 9)")

    def rotation_quat(self, step, n):
        raise NotImplementedError("sampler draws land with the diffusion sampler (Task 9)")

    def translation(self, step, shape):
        raise NotImplementedError("sampler draws land with the diffusion sampler (Task 9)")

    def step_noise(self, step, shape):
        raise NotImplementedError("sampler draws land with the diffusion sampler (Task 9)")
