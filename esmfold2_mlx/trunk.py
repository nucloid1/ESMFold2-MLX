"""The pair trunk: blocks, the language-model shim, and the looped Parcae trunk (model.py:878-1006).

``ParcaeTrunk`` runs ``num_loops + 1`` steps of

    lm_i     = lm_z * keep / (1 - p)                    (per-loop LM dropout, when enabled)
    inject   = msa_encoder(z_init, MSA) + lm_encoder(lm_i)   (the encoder output replaces z_init:
                                                          config overwrite=true; every fold has an MSA,
                                                          query-only in single-sequence mode)
    z        = a * z + LN(inject) @ B.T                 (a, B from the discretised dynamics)
    z        = folding_trunk(z)                         (48 PairUpdateBlocks)

from a truncated-normal initial state, then ``parcae_readout`` and the 2-block ``parcae_coda``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import mlx.core as mx

from .config import ModelConfig
from .layers import LayerNorm, Linear, Transition, TriangleMultiplication
from .msa import MSAEncoder, auto_chunks
from .rng import Draws
from .weights import Weights, key, load_group


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


class FoldingTrunk:
    """A stack of ``PairUpdateBlock`` under ``<prefix>.blocks.{i}`` (esm ``FoldingTrunk``)."""

    def __init__(self, w: Weights, prefix: str, n_layers: int) -> None:
        self.blocks = [PairUpdateBlock(w, key(prefix, f"blocks.{i}")) for i in range(n_layers)]

    def __call__(self, z: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        for block in self.blocks:
            z = block(z, mask)
        return z


def _gelu(x: mx.array) -> mx.array:
    """nn.GELU() (exact erf form)."""
    return 0.5 * x * (1.0 + mx.erf(x * (1.0 / math.sqrt(2.0))))


class LanguageModelShim:
    """esm ``LanguageModelShim`` (layers.py:2141): ESMC hidden states [1, N, 81, 2560] -> pair [1, N, N, 256].

    Softmax-weighted sum over the 81 layers of ``Linear(LN(h))``, then ``SingleToPair`` (outer product
    and difference through an MLP) and a final LayerNorm.
    """

    def __init__(self, w: Weights) -> None:
        self.in_norm = LayerNorm.load(w, "base_z_linear.0")
        self.in_proj = Linear.load(w, "base_z_linear.1")
        self.combine = w["base_z_combine"]
        self.down = Linear.load(w, "base_z_mlp.0.downproject")
        self.mlp_in = Linear.load(w, "base_z_mlp.0.output_mlp.0")
        self.mlp_out = Linear.load(w, "base_z_mlp.0.output_mlp.2")
        self.out_norm = LayerNorm.load(w, "base_z_mlp.1")

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = self.in_proj(self.in_norm(hidden_states.astype(mx.float32)))  # [1, N, 81, d_z]
        weights = mx.softmax(self.combine, axis=0)
        x = self.down((weights[None, :] @ h).squeeze(-2))  # [1, N, d_z]
        a, b = x[:, :, None, :], x[:, None, :, :]
        pair = mx.concatenate([a * b, a - b], axis=-1)  # [1, N, N, 2 d_z]
        return self.out_norm(self.mlp_out(_gelu(self.mlp_in(pair))))


def _softplus(x: mx.array) -> mx.array:
    """F.softplus with torch's default threshold 20."""
    return mx.where(x > 20.0, x, mx.log1p(mx.exp(x)))


class ParcaeTrunk:
    """The recycling-free looped trunk of ESMFold2 (model.py ``_run_one_loop`` plus readout and coda)."""

    GROUPS = ("folding_trunk", "lm_encoder", "language_model", "parcae_input_norm", "parcae_readout",
              "parcae_log_a", "parcae_log_delta", "parcae_b_cont", "parcae_coda")

    def __init__(self, weights_dir: str, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.groups = {name: load_group(weights_dir, name) for name in self.GROUPS}
        g = self.groups
        self.trunk = FoldingTrunk(g["folding_trunk"], "", cfg.n_trunk_layers)
        self.lm_encoder = FoldingTrunk(g["lm_encoder"], "", cfg.n_lm_encoder_layers)
        self.shim = LanguageModelShim(g["language_model"])
        self.input_norm = LayerNorm.load(g["parcae_input_norm"], "")
        self.readout = Linear.load(g["parcae_readout"], "")
        self.coda = FoldingTrunk(g["parcae_coda"], "", cfg.n_coda_layers)
        self.msa_encoder = MSAEncoder(weights_dir, cfg)
        delta = _softplus(g["parcae_log_delta"]["value"])
        self.a = mx.exp(-delta * mx.exp(g["parcae_log_a"]["value"]))  # [c], broadcast over the last axis
        self.b_t = (delta[:, None] * g["parcae_b_cont"]["value"]).T  # F.linear(x, b) == x @ b.T
        for name, w in g.items():
            w.assert_all_used()

    def step(self, z: mx.array, z_init: mx.array, lm_i: Optional[mx.array], mask: mx.array,
             msa_pair: Optional[mx.array] = None) -> mx.array:
        """One loop step from the carried ``z``; ``lm_i`` is the (dropped-out) LM pair or None."""
        inject = z_init if msa_pair is None else msa_pair  # the MSA encoder overwrites (config overwrite=true)
        if lm_i is not None:
            inject = inject + self.lm_encoder(lm_i, mask)
        z = self.a * z + self.input_norm(inject) @ self.b_t
        return self.trunk(z, mask)

    def lm_dropout_p(self, lm_dropout: Optional[float]) -> float:
        """esm ``_lm_dropout_context``: None or 0 keeps the checkpoint's setting, a value overrides it."""
        lm_cfg = self.cfg.lm_encoder
        if lm_dropout:
            return float(lm_dropout)
        return float(lm_cfg.lm_dropout) if lm_cfg.per_loop_lm_dropout else 0.0

    @staticmethod
    def _column_masked(msa_inputs: dict, draws: Draws, rate: float) -> dict:
        """``maybe_apply_msa_column_masking``, once per fold: drop a fraction ``rate`` of columns in
        the non-query rows. A no-op at depth 1 or rate 0."""
        mask = msa_inputs["msa_attention_mask"].astype(mx.bool_)
        B, M, L = mask.shape
        if M <= 1 or rate <= 0.0:
            return msa_inputs
        keep = draws.msa_column_keep((B, L), rate)  # [B, L]
        rows = mx.concatenate([mx.ones((B, 1, L), dtype=mx.bool_), mx.broadcast_to(keep[:, None, :], (B, M - 1, L))], axis=1)
        return {**msa_inputs, "msa_attention_mask": mask & rows}

    def __call__(self, x_inputs: Optional[mx.array], z_init: mx.array, lm_hidden_states: Optional[mx.array],
                 pair_mask: mx.array, num_loops: int, draws: Draws, msa_inputs: Optional[dict] = None,
                 lm_dropout: Optional[float] = None, msa_max_depth: Optional[int] = None,
                 msa_column_mask_rate: Optional[float] = None, opm_chunk: Optional[int] = None,
                 on_loop: Optional[Callable[[int, mx.array], None]] = None) -> mx.array:
        """``msa_inputs``: the feature dict's ``msa``, ``msa_attention_mask``, ``has_deletion``,
        ``deletion_value`` ([B, M, L]); the esm featurizer always supplies them (query-only in
        single-sequence mode). ``None`` for the MSA options means the checkpoint's value."""
        lm_z = None if lm_hidden_states is None else self.shim(lm_hidden_states)
        p = self.lm_dropout_p(lm_dropout) if lm_z is not None else 0.0
        z = draws.pair_init(tuple(z_init.shape), math.sqrt(2.0 / (5.0 * z_init.shape[-1])))
        mcfg = self.cfg.msa_encoder
        max_depth = mcfg.max_depth if msa_max_depth is None else msa_max_depth
        rate = mcfg.column_mask_rate if msa_column_mask_rate is None else msa_column_mask_rate
        if msa_inputs is not None:
            msa_inputs = self._column_masked(msa_inputs, draws, rate)
            depth = msa_inputs["msa"].shape[1]
            subsample = max_depth is not None and depth > 1 and depth > max_depth
            if opm_chunk is None:
                opm_chunk = auto_chunks(z_init.shape[1], mcfg.outer_hidden_size).opm_rows
        fixed_msa_pair = None
        for loop in range(num_loops + 1):
            lm_i = lm_z
            if lm_z is not None and p > 0.0:
                keep = draws.lm_dropout_keep(loop, tuple(lm_z.shape), p)
                lm_i = lm_z * (keep.astype(mx.float32) / (1.0 - p))
            msa_pair = None
            if msa_inputs is not None:
                if subsample:  # a fresh row subset every loop
                    rows = draws.msa_row_indices(loop, depth, max_depth)
                    sub = {k: v[:, rows] for k, v in msa_inputs.items()}
                    msa_pair = self.msa_encoder(z_init, x_inputs, *MSAEncoder.prepare(**sub), opm_chunk=opm_chunk)
                else:  # identical inputs every loop: torch recomputes it, the result is the same
                    if fixed_msa_pair is None:
                        fixed_msa_pair = self.msa_encoder(z_init, x_inputs, *MSAEncoder.prepare(**msa_inputs), opm_chunk=opm_chunk)
                        mx.eval(fixed_msa_pair)
                    msa_pair = fixed_msa_pair
            z = self.step(z, z_init, lm_i, pair_mask, msa_pair)
            mx.eval(z)
            if on_loop is not None:
                on_loop(loop, z)
        z = self.coda(self.readout(z), pair_mask)
        mx.eval(z)
        return z
