"""The MSA encoder (model.py:1427-1538, layers.py:2731-2828), fp32.

It runs in every fold: in single-sequence mode the esm featurizer supplies a query-only MSA, and
with ``msa_encoder.overwrite`` the encoder's output replaces ``z_init`` as the loop's injection.

    m    = embed([one_hot(msa, 33), has_deletion, deletion_value]) + project_inputs(x_inputs)
    per block:  pair += OuterProductMean(m)
                m    += PairWeightedAveraging(m, pair); m += PairTransition(m)    (not in the last block)
                pair += tri_mul_out(pair); pair += tri_mul_in(pair); pair += PairTransition(pair)

Layouts follow torch: MSA tensors are [B, L, M, ...] inside the encoder ("pre-transposed");
``MSAEncoder.prepare`` turns the feature layout [B, M, L] into it.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .config import ModelConfig
from .layers import LayerNorm, Linear, Transition, TriangleMultiplication
from .weights import Weights, key, load_group

NUM_RES_TYPES = 33
_MASKED_BIAS = -1e5


class OuterProductMean:
    """``Wout(vec(sum_m a[i,m,:] (x) b[j,m,:])) / n_valid[i,j]`` (the default ordering: the bias is divided too)."""

    def __init__(self, w: Weights, prefix: str) -> None:
        self.norm = LayerNorm.load(w, key(prefix, "norm"))
        self.W = Linear.load(w, key(prefix, "W"))
        self.Wout = Linear.load(w, key(prefix, "Wout"))
        self.hidden = self.W.weight.shape[0] // 2

    def __call__(self, m: mx.array, mask: mx.array, chunk: Optional[int] = None) -> mx.array:
        B, L, M, _ = m.shape
        c = self.hidden
        x = self.W(self.norm(m)) * mask[..., None]
        a, b = x[..., :c], x[..., c:]
        n_valid = mx.maximum(mask @ mask.swapaxes(-1, -2), 1.0)[..., None]  # [B, L, L, 1]
        a_t = a.transpose(0, 1, 3, 2).reshape(B, L * c, M)  # rows (i, c)
        b_t = b.transpose(0, 2, 1, 3).reshape(B, M, L * c)  # cols (j, d)
        rows = L if chunk is None else chunk
        parts = []
        for start in range(0, L, rows):
            stop = min(start + rows, L)
            outer = (a_t[:, start * c: stop * c] @ b_t).reshape(B, stop - start, c, L, c)
            outer = outer.transpose(0, 1, 3, 2, 4).reshape(B, stop - start, L, c * c)
            parts.append(self.Wout(outer) / n_valid[:, start:stop])
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


class MSAPairWeightedAveraging:
    """AF3 Algorithm 10: per head, rows of the MSA averaged over j with softmax_j(pair bias), gated."""

    def __init__(self, w: Weights, prefix: str, n_heads: int, head_width: int) -> None:
        self.h, self.dh = n_heads, head_width
        self.norm_single = LayerNorm.load(w, key(prefix, "norm_single"))
        self.bias_norm = LayerNorm.load(w, key(prefix, "compute_bias.0"))
        self.bias_proj = Linear.load(w, key(prefix, "compute_bias.1"))
        self.Wv = Linear.load(w, key(prefix, "Wv"))
        self.Wgate = Linear.load(w, key(prefix, "Wgate"))
        self.Wout = Linear.load(w, key(prefix, "Wout"))

    def __call__(self, m: mx.array, pair: mx.array, pair_mask: mx.array) -> mx.array:
        B, L, M, _ = m.shape
        h, dh = self.h, self.dh
        normed = self.norm_single(m)
        bias = self.bias_proj(self.bias_norm(pair))  # [B, L, L, h]
        bias = mx.where(pair_mask.astype(mx.bool_)[..., None], bias, _MASKED_BIAS)
        attn = mx.softmax(bias, axis=-2).transpose(0, 3, 1, 2)  # [B, h, i, j]
        v = self.Wv(normed).reshape(B, L, M, h, dh).transpose(0, 3, 1, 2, 4).reshape(B, h, L, M * dh)
        averaged = (attn @ v).reshape(B, h, L, M, dh).transpose(0, 2, 3, 1, 4)  # [B, i, M, h, dh]
        gate = mx.sigmoid(self.Wgate(normed)).reshape(B, L, M, h, dh)
        return self.Wout((averaged * gate).reshape(B, L, M, h * dh))


class MSAEncoderBlock:
    def __init__(self, w: Weights, prefix: str, final: bool, n_heads: int, head_width: int) -> None:
        self.final = final
        self.opm = OuterProductMean(w, key(prefix, "outer_product_mean"))
        if not final:
            self.pwa = MSAPairWeightedAveraging(w, key(prefix, "msa_pair_weighted_averaging"), n_heads, head_width)
            self.msa_transition = Transition(w, key(prefix, "msa_transition"), residual=False)
        self.tri_mul_out = TriangleMultiplication(w, key(prefix, "tri_mul_out._engine"), outgoing=True)
        self.tri_mul_in = TriangleMultiplication(w, key(prefix, "tri_mul_in._engine"), outgoing=False)
        self.pair_transition = Transition(w, key(prefix, "pair_transition"), residual=False)

    def __call__(self, m: mx.array, pair: mx.array, msa_mask: mx.array, pair_mask: mx.array,
                 opm_chunk: Optional[int] = None) -> tuple[mx.array, mx.array]:
        pair = pair + self.opm(m, msa_mask, chunk=opm_chunk)
        if not self.final:
            m = m + self.pwa(m, pair, pair_mask)
            m = m + self.msa_transition(m)
        pm = pair_mask.astype(mx.float32)
        pair = self.tri_mul_out(pair, pm, residual=pair)
        pair = self.tri_mul_in(pair, pm, residual=pair)
        return m, pair + self.pair_transition(pair)


class MSAEncoder:
    def __init__(self, weights_dir: str, cfg: ModelConfig) -> None:
        mcfg = cfg.msa_encoder
        self.weights = load_group(weights_dir, "msa_encoder")
        w = self.weights
        self.embed = Linear.load(w, "embed")
        self.project_inputs = Linear.load(w, "project_inputs")
        n = mcfg.num_hidden_layers
        self.blocks = [MSAEncoderBlock(w, f"blocks.{i}", final=(i == n - 1), n_heads=mcfg.num_attention_heads,
                                       head_width=mcfg.head_width) for i in range(n)]
        w.assert_all_used()

    @staticmethod
    def prepare(msa: mx.array, msa_attention_mask: mx.array, has_deletion: mx.array,
                deletion_value: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Feature layout [B, M, L] -> encoder inputs [B, L, M, ...] as ``_run_one_loop`` builds them."""
        mask = msa_attention_mask.transpose(0, 2, 1).astype(mx.float32)
        one_hot = (msa.transpose(0, 2, 1)[..., None] == mx.arange(NUM_RES_TYPES)).astype(mx.float32)
        return (one_hot * mask[..., None], has_deletion.transpose(0, 2, 1).astype(mx.float32),
                deletion_value.transpose(0, 2, 1).astype(mx.float32), mask)

    def encode(self, x_pair: mx.array, x_inputs: mx.array, msa_oh: mx.array, has_deletion: mx.array,
               deletion_value: mx.array, msa_mask: mx.array, opm_chunk: Optional[int] = None) -> mx.array:
        features = mx.concatenate([msa_oh, has_deletion[..., None], deletion_value[..., None]], axis=-1)
        m = self.embed(features) + self.project_inputs(x_inputs)[:, :, None, :]
        tok = msa_mask[:, :, 0] > 0
        pair_mask = tok[:, :, None] & tok[:, None, :]
        for block in self.blocks:
            m, x_pair = block(m, x_pair, msa_mask, pair_mask, opm_chunk=opm_chunk)
        return x_pair

    __call__ = encode
