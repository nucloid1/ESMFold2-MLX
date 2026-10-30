"""Atom-level sliding-window attention with 3D RoPE (layers.py:443-1011): the atom encoder of the
inputs embedder and of the diffusion module, and the diffusion module's atom decoder.

Torch's own casts are mirrored, because they are part of the model on every device: the RoPE cos/sin
tables are rounded to bf16 (``build_3d_rope``), and on the non-flash path (CPU and MPS) the attention
runs on bf16 q, k, v (``SWA3DRoPEAttention``). Here q, k, v are rounded to bf16, the attention is
accumulated in fp32, and the output is rounded to bf16, as a float-accumulating bf16 kernel does.
Everything else is fp32.

Attention is banded (blocks of ``block`` queries against ``block + 2 * half_window`` keys), never a
dense atom-by-atom matrix, when the valid atoms form a prefix (padding at the end, as the featurizer
lays them out); otherwise it falls back to torch's rank-based dense mask.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .config import AtomEncoderConfig
from .layers import LayerNorm, Linear
from .weights import Weights, key

F32_EPS = float(np.finfo(np.float32).eps)  # F.rms_norm's default eps for fp32 input
MAX_CHARS, CHAR_VOCAB = 4, 64


def _bf16(x: mx.array) -> mx.array:
    return x.astype(mx.bfloat16).astype(mx.float32)


def rms_norm(x: mx.array) -> mx.array:
    """F.rms_norm without weight."""
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + F32_EPS)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def build_rope(ref_pos: mx.array, ref_space_uid: mx.array, head_dim: int, n_spatial: int, n_uid: int,
               spatial_base: float, uid_base: float) -> tuple[mx.array, mx.array]:
    """``build_3d_rope``: cos/sin [B, N, head_dim / 2], bf16-rounded like torch's, held as fp32."""
    B, N = ref_pos.shape[:2]
    half = head_dim // 2
    spatial_inv = 1.0 / (spatial_base ** (mx.arange(n_spatial, dtype=mx.float32) / n_spatial))
    uid_inv = 1.0 / (uid_base ** (mx.arange(n_uid, dtype=mx.float32) / n_uid))
    spatial = (ref_pos.astype(mx.float32)[..., None] * spatial_inv).reshape(B, N, 3 * n_spatial)
    uid = ref_space_uid.astype(mx.float32)[..., None] * uid_inv
    freqs = mx.concatenate([spatial, uid], axis=-1)
    if freqs.shape[-1] < half:
        freqs = mx.concatenate([freqs, mx.zeros((B, N, half - freqs.shape[-1]))], axis=-1)
    return _bf16(mx.cos(freqs)), _bf16(mx.sin(freqs))


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """``apply_rotary_emb_3d``: x [B, N, H, D], cos/sin [B, N, D/2] repeated to D."""
    ro = cos.shape[-1] * 2
    c = mx.concatenate([cos, cos], axis=-1)[:, :, None, :]
    s = mx.concatenate([sin, sin], axis=-1)[:, :, None, :]
    xr = x[..., :ro]
    half = ro // 2
    rotated = mx.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    out = xr * c + rotated * s
    return out if ro == x.shape[-1] else mx.concatenate([out, x[..., ro:]], axis=-1)


def _valid_is_prefix(valid: mx.array) -> bool:
    v = np.array(valid)
    counts = v.sum(axis=1)
    return all(v[b, : counts[b]].all() for b in range(v.shape[0]))


def _softmax_masked(scores: mx.array, allowed: mx.array) -> mx.array:
    return mx.softmax(mx.where(allowed, scores, -mx.inf), axis=-1)


def swa_attention(q: mx.array, k: mx.array, v: mx.array, valid: mx.array, half_window: int, block: int = 64,
                  bf16: bool = True) -> mx.array:
    """Sliding-window attention: |rank_i - rank_j| <= half_window over valid atoms, every row may
    attend to itself, padded rows output zero. q, k, v [B, N, H, D] -> [B, N, H, D]."""
    B, N, H, D = q.shape
    scale = D ** -0.5
    if bf16:
        q, k, v = _bf16(q), _bf16(k), _bf16(v)
    valid = valid.astype(mx.bool_)
    if _valid_is_prefix(valid):
        hw = half_window
        nb = -(-N // block)
        Np = nb * block
        W = block + 2 * hw
        qb = mx.pad(q, [(0, 0), (0, Np - N), (0, 0), (0, 0)]).reshape(B, nb, block, H, D).transpose(0, 1, 3, 2, 4)
        kp = mx.pad(k, [(0, 0), (hw, Np - N + hw), (0, 0), (0, 0)])
        vp = mx.pad(v, [(0, 0), (hw, Np - N + hw), (0, 0), (0, 0)])
        validp = mx.pad(valid, [(0, 0), (hw, Np - N + hw)])
        idx = mx.arange(nb)[:, None] * block + mx.arange(W)[None, :]  # [nb, W] into the padded key axis
        kw = kp[:, idx].transpose(0, 1, 3, 4, 2)  # [B, nb, H, D, W]
        vw = vp[:, idx].transpose(0, 1, 3, 2, 4)  # [B, nb, H, W, D]
        scores = (qb @ kw) * scale  # [B, nb, H, block, W]
        qpos = mx.arange(nb)[:, None] * block + mx.arange(block)[None, :]  # [nb, block]
        kpos = idx - hw  # [nb, W]
        within = mx.abs(qpos[:, :, None] - kpos[:, None, :]) <= hw
        qvalid = mx.pad(valid, [(0, 0), (0, Np - N)]).reshape(B, nb, block)
        allowed = within[None] & validp[:, idx][:, :, None, :] & qvalid[:, :, :, None]
        allowed = allowed | (qpos[:, :, None] == kpos[:, None, :])[None]
        out = _softmax_masked(scores, allowed[:, :, None]) @ vw  # [B, nb, H, block, D]
        out = out.transpose(0, 1, 3, 2, 4).reshape(B, Np, H, D)[:, :N]
    else:
        rank = mx.cumsum(valid.astype(mx.int32), axis=1) - 1
        allowed = (mx.abs(rank[:, :, None] - rank[:, None, :]) <= half_window) & valid[:, None, :] & valid[:, :, None]
        allowed = allowed | mx.eye(N, dtype=mx.bool_)[None]
        scores = (q.transpose(0, 2, 1, 3) @ k.transpose(0, 2, 3, 1)) * scale  # [B, H, N, N]
        out = (_softmax_masked(scores, allowed[:, None]) @ v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    out = out * valid[:, :, None, None]
    return _bf16(out) if bf16 else out


class SWAttention:
    """``SWA3DRoPEAttention``: Wqkv, qk RMS-norm, RoPE, bf16 sliding-window attention, sigmoid gate, out_proj."""

    def __init__(self, w: Weights, prefix: str, n_heads: int, half_window: int) -> None:
        self.Wqkv = Linear.load(w, key(prefix, "Wqkv"))
        self.out_proj = Linear.load(w, key(prefix, "out_proj"))
        self.gate_proj = Linear.load(w, key(prefix, "gate_proj"))
        self.h, self.half_window = n_heads, half_window

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, valid: mx.array) -> mx.array:
        B, N, d = x.shape
        H = self.h
        qkv = self.Wqkv(x).reshape(B, N, 3, H, d // H)
        q, k, v = rms_norm(qkv[:, :, 0]), rms_norm(qkv[:, :, 1]), qkv[:, :, 2]
        out = swa_attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, valid, self.half_window)
        return self.out_proj(out.reshape(B, N, d) * mx.sigmoid(self.gate_proj(x)))


class SWAAtomBlock:
    """adaLN-Zero (RMS-norm, shift/scale/gate from SiLU(c)) around attention and a SwiGLU FFN."""

    def __init__(self, w: Weights, prefix: str, n_heads: int, half_window: int) -> None:
        self.adaln = Linear.load(w, key(prefix, "adaln_modulation.1"))
        self.attn = SWAttention(w, key(prefix, "attn"), n_heads, half_window)
        self.w_up = Linear.load(w, key(prefix, "ffn.w_up"))
        self.w_down = Linear.load(w, key(prefix, "ffn.w_down"))

    def __call__(self, x: mx.array, c: mx.array, cos: mx.array, sin: mx.array, valid: mx.array) -> mx.array:
        mod = self.adaln(_silu(c))
        if mod.ndim == 2:
            mod = mod[:, None]
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mx.split(mod, 6, axis=-1)
        x = x + gate_a * self.attn(rms_norm(x) * (1 + scale_a) + shift_a, cos, sin, valid)
        x1, x2 = mx.split(self.w_up(rms_norm(x) * (1 + scale_f) + shift_f), 2, axis=-1)
        return x + gate_f * self.w_down(_silu(x1) * x2)


def scatter_mean(x: mx.array, atom_to_token: mx.array, n_tokens: int, atom_mask: mx.array) -> mx.array:
    """``scatter_atom_to_token``: mean of x over each token's valid atoms; tokens without atoms get 0."""
    B, A, d = x.shape
    idx = mx.where(atom_mask.astype(mx.bool_), atom_to_token.astype(mx.int32), n_tokens)
    rows = mx.broadcast_to(mx.arange(B)[:, None], idx.shape)
    sums = mx.zeros((B, n_tokens + 1, d)).at[rows, idx].add(x)
    counts = mx.zeros((B, n_tokens + 1, 1)).at[rows, idx].add(mx.ones((B, A, 1)))
    return (mx.where(counts > 0, sums / mx.maximum(counts, 1.0), 0.0))[:, :n_tokens]


class AtomEncoder:
    """``EsmFold2AtomEncoder``. ``structure_prediction`` adds ``coords_linear`` (diffusion module)."""

    def __init__(self, w: Weights, prefix: str, acfg: AtomEncoderConfig, window: int, structure_prediction: bool) -> None:
        self.atom_linear = Linear.load(w, key(prefix, "atom_linear"))
        self.atom_norm = LayerNorm.load(w, key(prefix, "atom_norm"))
        self.coords_linear = Linear.load(w, key(prefix, "coords_linear")) if structure_prediction else None
        half = window // 2
        self.blocks = [SWAAtomBlock(w, key(prefix, f"atom_transformer.blocks.{i}"), acfg.num_attention_heads, half)
                       for i in range(acfg.num_hidden_layers)]
        self.to_token = Linear.load(w, key(prefix, "atom_to_token_linear"))
        self.rope = dict(head_dim=acfg.hidden_size // acfg.num_attention_heads, n_spatial=acfg.num_spatial_rope_pairs_per_axis,
                         n_uid=acfg.num_uid_rope_pairs, spatial_base=acfg.spatial_rope_base_frequency,
                         uid_base=acfg.uid_rope_base_frequency)

    def static(self, ref_pos, atom_mask, ref_space_uid, ref_charge, ref_element_oh, chars_oh):
        """Step-invariant parts: c (the atom conditioning) and the RoPE tables."""
        B, N = ref_pos.shape[:2]
        feats = mx.concatenate([ref_pos.astype(mx.float32), ref_charge.astype(mx.float32)[..., None],
                                atom_mask.astype(mx.float32)[..., None], ref_element_oh,
                                chars_oh.reshape(B, N, MAX_CHARS * CHAR_VOCAB)], axis=-1)
        cos, sin = build_rope(ref_pos, ref_space_uid, **self.rope)
        return self.atom_norm(self.atom_linear(feats)), cos, sin

    def transformer(self, q, c, cos, sin, valid):
        for block in self.blocks:
            q = block(q, c, cos, sin, valid)
        return q

    def __call__(self, ref_pos, atom_mask, ref_space_uid, ref_charge, ref_element_oh, chars_oh, atom_to_token):
        """Inputs-embedder use (no coordinates): returns (a [B, L, d_token/2], q, c)."""
        c, cos, sin = self.static(ref_pos, atom_mask, ref_space_uid, ref_charge, ref_element_oh, chars_oh)
        q = self.transformer(c, c, cos, sin, atom_mask)
        n_tokens = int(mx.max(atom_to_token).item()) + 1
        a = scatter_mean(mx.maximum(self.to_token(q), 0.0), atom_to_token, n_tokens, atom_mask)
        return a, q, c

    def encode_coords(self, static, r_noisy: mx.array, atom_mask: mx.array, atom_to_token: mx.array, n_tokens: int):
        """Diffusion-module use: ``r_noisy`` [S, A, 3] (S = batch x samples) enters through
        ``coords_linear([r, 0])``. Returns (a [S, L, d_token], q, c, (cos, sin, valid)) for the decoder."""
        c_base, cos, sin = static
        S = r_noisy.shape[0]
        rep = (lambda x: x) if S == c_base.shape[0] else (lambda x: mx.repeat(x, S // x.shape[0], axis=0))
        c = rep(c_base)
        q = c + self.coords_linear(mx.concatenate([r_noisy, mx.zeros_like(r_noisy)], axis=-1))
        params = (rep(cos), rep(sin), rep(atom_mask))
        q = self.transformer(q, c, *params)
        a = scatter_mean(mx.maximum(self.to_token(q), 0.0), rep(atom_to_token), n_tokens, params[2])
        return a, q, c, params


class AtomDecoder:
    """``EsmFold2AtomDecoder``: token features broadcast to atoms, the atom transformer, LN, xyz."""

    def __init__(self, w: Weights, prefix: str, acfg: AtomEncoderConfig, window: int) -> None:
        self.token_to_atom = Linear.load(w, key(prefix, "token_to_atom_linear"))
        self.blocks = [SWAAtomBlock(w, key(prefix, f"atom_transformer.blocks.{i}"), acfg.num_attention_heads, window // 2)
                       for i in range(acfg.num_hidden_layers)]
        self.norm = LayerNorm.load(w, key(prefix, "norm"))
        self.output_linear = Linear.load(w, key(prefix, "output_linear"))

    def __call__(self, a: mx.array, q: mx.array, c: mx.array, params: tuple, atom_to_token: mx.array) -> mx.array:
        idx = atom_to_token.astype(mx.int32)
        if idx.shape[0] != a.shape[0]:
            idx = mx.repeat(idx, a.shape[0] // idx.shape[0], axis=0)
        q = q + mx.take_along_axis(self.token_to_atom(a), idx[..., None], axis=1)
        for block in self.blocks:
            q = block(q, c, *params)
        return self.output_linear(self.norm(q))
