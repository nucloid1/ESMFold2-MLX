"""MLX layers mirroring ESMFold2's pair stack (esm.models.esmfold2.layers), fp32 throughout.

Each class loads one torch module's weights unchanged and reproduces its output exactly, residual
included where the torch module includes it:

- ``LayerNorm``                nn.LayerNorm, eps 1e-5.
- ``Linear``                   nn.Linear: ``y = x @ W.T (+ b)``, W stored [out, in].
- ``Transition``               layers.py:2503: ``x + w3(silu(x1) * x2)`` with ``[x1, x2] = w12(LN(x))``.
- ``TriangleMultiplication``   layers.py:2352 ``TriangleMultiplicativeBlock``: returns the update
                               ``mixed * sigmoid(proj_gate(LN(z)))`` before the residual add.

The elementwise passes (sigmoid routing, SwiGLU, gate and residual) are ``mx.compile``d with
``shapeless=True`` so a new N does not retrace; the matmuls and the triangle contraction are native.
Shapes follow torch exactly (leading batch dimension kept) so references compare element-wise.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .weights import Weights, key

_EPS = 1e-5


@mx.compile
def _route(signal: mx.array, gate: mx.array, mask: mx.array) -> mx.array:
    return signal * mx.sigmoid(gate) * mask


@mx.compile
def _route_unmasked(signal: mx.array, gate: mx.array) -> mx.array:
    return signal * mx.sigmoid(gate)


@mx.compile
def _gated(mixed: mx.array, gate_logits: mx.array) -> mx.array:
    return mixed * mx.sigmoid(gate_logits)


@mx.compile
def _gated_residual(mixed: mx.array, gate_logits: mx.array, residual: mx.array) -> mx.array:
    return residual + mixed * mx.sigmoid(gate_logits)


@mx.compile
def _swiglu(x1: mx.array, x2: mx.array) -> mx.array:
    return x1 * mx.sigmoid(x1) * x2


@mx.compile
def _add(a: mx.array, b: mx.array) -> mx.array:
    return a + b


class LayerNorm:
    """Normalize the last axis; scale and offset as stored (both present in every ESMFold2 norm)."""

    def __init__(self, weight: mx.array, bias: Optional[mx.array] = None, eps: float = _EPS) -> None:
        self.weight, self.bias, self.eps = weight, bias, eps

    def __call__(self, x: mx.array) -> mx.array:
        if self.bias is not None:
            return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + self.eps) * self.weight

    @classmethod
    def load(cls, w: Weights, prefix: str, eps: float = _EPS) -> "LayerNorm":
        return cls(w[key(prefix, "weight")], w.get(key(prefix, "bias")), eps)


class Linear:
    """``y = x @ W.T (+ b)``; torch stores W as [out, in]."""

    def __init__(self, weight: mx.array, bias: Optional[mx.array] = None) -> None:
        self.weight, self.bias = weight, bias
        self._wt = weight.T

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self._wt
        return y if self.bias is None else y + self.bias

    @classmethod
    def load(cls, w: Weights, prefix: str) -> "Linear":
        return cls(w[key(prefix, "weight")], w.get(key(prefix, "bias")))


class Transition:
    """LayerNorm + SwiGLU feed-forward. ``residual=True`` mirrors esm ``Transition`` (returns
    ``x + ffn(LN(x))``); ``residual=False`` mirrors ``PairTransition`` (model.py:81, returns the update)."""

    def __init__(self, w: Weights, prefix: str, residual: bool = True) -> None:
        self.norm = LayerNorm.load(w, key(prefix, "norm"))
        self.w12 = Linear.load(w, key(prefix, "ffn.w12"))
        self.w3 = Linear.load(w, key(prefix, "ffn.w3"))
        self.hidden = self.w3.weight.shape[1]
        self.residual = residual

    def __call__(self, x: mx.array) -> mx.array:
        x12 = self.w12(self.norm(x))
        update = self.w3(_swiglu(x12[..., : self.hidden], x12[..., self.hidden:]))
        return _add(x, update) if self.residual else update


class TriangleMultiplication:
    """Triangle multiplicative update (outgoing: ``bikd,bjkd->bijd``; incoming: ``bkid,bkjd->bijd``).

    ``__call__(z, mask)`` returns the update; ``residual=z`` returns ``z + update`` in one fused pass.
    """

    def __init__(self, w: Weights, prefix: str, outgoing: bool) -> None:
        self.outgoing = outgoing
        self.norm_start = LayerNorm.load(w, key(prefix, "norm_start"))
        self.norm_mix = LayerNorm.load(w, key(prefix, "norm_mix"))
        self.proj_bundle = Linear.load(w, key(prefix, "proj_bundle"))
        self.proj_emit = Linear.load(w, key(prefix, "proj_emit"))
        self.proj_gate = Linear.load(w, key(prefix, "proj_gate"))

    def _contract(self, left: mx.array, right: mx.array) -> mx.array:
        """Batched over channels: [B,N,N,c] x [B,N,N,c] -> [B,N,N,c] with a matmul over k."""
        if self.outgoing:
            a = left.transpose(0, 3, 1, 2)   # [B,c,i,k]
            b = right.transpose(0, 3, 2, 1)  # [B,c,k,j]
        else:
            a = left.transpose(0, 3, 2, 1)   # [B,c,i,k]
            b = right.transpose(0, 3, 1, 2)  # [B,c,k,j]
        return (a @ b).transpose(0, 2, 3, 1)  # [B,i,j,c]

    def __call__(self, z: mx.array, mask: Optional[mx.array] = None, residual: Optional[mx.array] = None) -> mx.array:
        normalized = self.norm_start(z)
        bundled = self.proj_bundle(normalized)
        half = bundled.shape[-1] // 2
        signal, gate = bundled[..., :half], bundled[..., half:]
        routed = _route_unmasked(signal, gate) if mask is None else _route(signal, gate, mask[..., None])
        c = half // 2
        mixed = self.proj_emit(self.norm_mix(self._contract(routed[..., :c], routed[..., c:])))
        gate_logits = self.proj_gate(normalized)
        return _gated(mixed, gate_logits) if residual is None else _gated_residual(mixed, gate_logits, residual)
