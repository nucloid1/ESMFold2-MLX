"""The diffusion module and sampler (layers.py:1019-1929), fp32 apart from the atom attention's
mirrored bf16 casts.

Per denoiser call (``DiffusionModule.denoise``):

    s       = conditioning(t_hat, s_inputs)                 (Fourier noise embedding, transitions)
    a, q, c = atom_encoder(x_noisy / sqrt(t^2 + sd^2))     (coords enter through coords_linear)
    a      += s_to_token(LN(s));  a = token_transformer(a, s, pair bias from z);  a = LN(a)
    r       = atom_decoder(a, q, c)
    x_den   = sd^2 / (sd^2 + t^2) * x_noisy + sd t / sqrt(sd^2 + t^2) * r

The pair conditioning z, the twelve per-block pair biases and the atom conditioning/RoPE tables do not
depend on the noise level, so ``prepare`` computes them once per fold (torch caches z and the atom
conditioning, and recomputes the pair biases every call; the values are identical).

The sampler (``DiffusionSampler.sample``) is AF3 Algorithm 18 as esm implements it: a Karras schedule
capped at ``max_inference_sigma``, per step a random centring rotation and translation, noise,
a denoiser call, Kabsch alignment of the noisy coordinates onto the denoised ones, and an Euler step.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from .atoms import AtomDecoder, AtomEncoder
from .config import ModelConfig
from .layers import LayerNorm, Linear
from .rng import Draws
from .weights import Weights, key, load_group

_F32_MIN = float(np.finfo(np.float32).min)


def _silu(x):
    return x * mx.sigmoid(x)


def _rep(x: mx.array, n: int) -> mx.array:
    return x if n == 1 or x.shape[0] == n else mx.repeat(x, n // x.shape[0], axis=0)


class AdaptiveLayerNorm:
    def __init__(self, w: Weights, prefix: str) -> None:
        self.s_scale = w[key(prefix, "s_scale")]
        self.s_gate = Linear.load(w, key(prefix, "s_gate"))
        self.s_shift = Linear.load(w, key(prefix, "s_shift"))

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        a_norm = mx.fast.layer_norm(a, None, None, 1e-5)
        s_norm = mx.fast.layer_norm(s, self.s_scale, None, 1e-5)
        return mx.sigmoid(self.s_gate(s_norm)) * a_norm + self.s_shift(s_norm)


class TransitionLayer:
    """norm -> silu(a_proj) * b_proj -> out_proj (layers.py:344)."""

    def __init__(self, w: Weights, prefix: str) -> None:
        self.norm = LayerNorm.load(w, key(prefix, "norm"))
        self.a = Linear.load(w, key(prefix, "a_proj"))
        self.b = Linear.load(w, key(prefix, "b_proj"))
        self.out = Linear.load(w, key(prefix, "out_proj"))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        return self.out(_silu(self.a(x)) * self.b(x))


class DiffusionConditioning:
    def __init__(self, w: Weights, prefix: str, sigma_data: float) -> None:
        self.sigma = float(sigma_data)
        self.z_input_norm = LayerNorm.load(w, key(prefix, "z_input_norm"))
        self.z_proj = Linear.load(w, key(prefix, "z_proj"))
        self.z_transitions = [TransitionLayer(w, key(prefix, f"z_transitions.{i}")) for i in range(2)]
        self.s_input_norm = LayerNorm.load(w, key(prefix, "s_input_norm"))
        self.s_proj = Linear.load(w, key(prefix, "s_proj"))
        self.fourier_w = w[key(prefix, "fourier.w")]
        self.fourier_b = w[key(prefix, "fourier.b")]
        self.noise_norm = LayerNorm.load(w, key(prefix, "noise_norm"))
        self.noise_proj = Linear.load(w, key(prefix, "noise_proj"))
        self.s_transitions = [TransitionLayer(w, key(prefix, f"s_transitions.{i}")) for i in range(2)]

    def pair(self, z_trunk: mx.array, relative_position_encoding: mx.array) -> mx.array:
        z = self.z_proj(self.z_input_norm(mx.concatenate([z_trunk, relative_position_encoding], axis=-1)))
        for block in self.z_transitions:
            z = z + block(z)
        return z

    def single(self, t: mx.array, s_inputs: mx.array, n: int) -> mx.array:
        s = self.s_proj(self.s_input_norm(_rep(s_inputs, n)))
        t_noise = 0.25 * mx.log(mx.maximum(t / self.sigma, 1e-20))
        fourier = mx.cos(2.0 * math.pi * (t_noise[:, None] * self.fourier_w[None, :] + self.fourier_b[None, :]))
        s = s + self.noise_proj(self.noise_norm(fourier))[:, None, :]
        for block in self.s_transitions:
            s = s + block(s)
        return s


class AttentionPairBias:
    """Gated multi-head attention with an additive pair bias (the non-fused path esm takes on Mac)."""

    def __init__(self, w: Weights, prefix: str, n_heads: int) -> None:
        self.adaln = AdaptiveLayerNorm(w, key(prefix, "adaln"))
        self.out_gate = Linear.load(w, key(prefix, "out_gate"))
        self.q_proj = Linear.load(w, key(prefix, "q_proj"))
        self.kv_proj = Linear.load(w, key(prefix, "kv_proj"))
        self.g_proj = Linear.load(w, key(prefix, "g_proj"))
        self.out_proj = Linear.load(w, key(prefix, "out_proj"))
        self.pair_norm = LayerNorm.load(w, key(prefix, "pair_norm"))
        self.pair_bias_proj = Linear.load(w, key(prefix, "pair_bias_proj"))
        self.h = n_heads

    def pair_bias(self, z: mx.array) -> mx.array:
        """[B, L, L, H] -> [B, H, L, L]; the same for every sample and every noise level."""
        return self.pair_bias_proj(self.pair_norm(z)).transpose(0, 3, 1, 2)

    def __call__(self, a: mx.array, s: mx.array, bias: mx.array, key_mask: Optional[mx.array]) -> mx.array:
        S, L, d = a.shape
        H = self.h
        D = d // H
        x = self.adaln(a, s)
        q = self.q_proj(x).reshape(S, L, H, D).transpose(0, 2, 1, 3)
        k, v = mx.split(self.kv_proj(x), 2, axis=-1)
        k = k.reshape(S, L, H, D).transpose(0, 2, 3, 1)
        v = v.reshape(S, L, H, D).transpose(0, 2, 1, 3)
        logits = (q @ k) * (D ** -0.5) + bias  # [S, H, i, j]
        if key_mask is not None:
            logits = logits + mx.where(key_mask.astype(mx.bool_)[:, None, None, :], 0.0, _F32_MIN)
        ctx = (mx.softmax(logits, axis=-1) @ v).transpose(0, 2, 1, 3)  # [S, i, H, D]
        gate = mx.sigmoid(self.g_proj(x)).reshape(S, L, H, D)
        return mx.sigmoid(self.out_gate(s)) * self.out_proj((gate * ctx).reshape(S, L, d))


class ConditionedTransition:
    def __init__(self, w: Weights, prefix: str) -> None:
        self.adaln = AdaptiveLayerNorm(w, key(prefix, "adaln"))
        self.output_gate = Linear.load(w, key(prefix, "output_gate"))
        self.lin_swish = Linear.load(w, key(prefix, "lin_swish"))
        self.lin_out = Linear.load(w, key(prefix, "lin_out"))

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        sa, sb = mx.split(self.lin_swish(self.adaln(a, s)), 2, axis=-1)
        return mx.sigmoid(self.output_gate(s)) * self.lin_out(_silu(sa) * sb)


class DiffusionTransformer:
    def __init__(self, w: Weights, prefix: str, n_blocks: int, n_heads: int) -> None:
        self.attn = [AttentionPairBias(w, key(prefix, f"attn_blocks.{i}"), n_heads) for i in range(n_blocks)]
        self.trans = [ConditionedTransition(w, key(prefix, f"transition_blocks.{i}")) for i in range(n_blocks)]

    def pair_biases(self, z: mx.array) -> list[mx.array]:
        return [block.pair_bias(z) for block in self.attn]

    def __call__(self, a, s, biases, key_mask):
        for attn, trans, bias in zip(self.attn, self.trans, biases):
            a = a + attn(a, s, bias, key_mask)
            a = a + trans(a, s)
        return a


class DiffusionModule:
    PREFIX = "diffusion_module"

    def __init__(self, weights_dir: str, cfg: ModelConfig) -> None:
        self.weights = load_group(weights_dir, "structure_head")
        w, p = self.weights, self.PREFIX
        dm = cfg.structure_head.diffusion_module
        self.sigma = float(dm.sigma_data)
        self.conditioning = DiffusionConditioning(w, key(p, "conditioning"), dm.sigma_data)
        self.atom_encoder = AtomEncoder(w, key(p, "atom_encoder"), dm.atom_encoder, window=cfg.sliding_window,
                                        structure_prediction=True)
        self.atom_decoder = AtomDecoder(w, key(p, "atom_decoder"), dm.atom_encoder, window=cfg.sliding_window)
        self.s_to_token = Linear.load(w, key(p, "s_to_token"))
        self.s_step_norm = LayerNorm.load(w, key(p, "s_step_norm"))
        self.token_norm = LayerNorm.load(w, key(p, "token_norm"))
        self.transformer = DiffusionTransformer(w, key(p, "token_transformer"), dm.token_num_blocks, dm.token_num_heads)
        w.assert_all_used()

    def prepare(self, z_trunk, s_inputs, relative_position_encoding, ref_pos, ref_charge, ref_mask, ref_element,
                ref_atom_name_chars, ref_space_uid, tok_idx, token_attention_mask=None, **_unused) -> dict:
        """Everything that does not depend on the noise level, once per fold."""
        z = self.conditioning.pair(z_trunk, relative_position_encoding)
        st = {
            "s_inputs": s_inputs,
            "biases": self.transformer.pair_biases(z),
            "atom_static": self.atom_encoder.static(ref_pos, ref_mask, ref_space_uid, ref_charge, ref_element, ref_atom_name_chars),
            "ref_mask": ref_mask,
            "tok_idx": tok_idx,
            "n_tokens": int(mx.max(tok_idx).item()) + 1,
            "token_mask": token_attention_mask,
        }
        mx.eval(st["biases"], *st["atom_static"])
        return st

    def denoise(self, st: dict, x_noisy: mx.array, t: mx.array) -> mx.array:
        n = x_noisy.shape[0]
        sd = self.sigma
        s = self.conditioning.single(t, st["s_inputs"], n)
        r_noisy = x_noisy / mx.sqrt(t * t + sd * sd)[:, None, None]
        a, q, c, params = self.atom_encoder.encode_coords(st["atom_static"], r_noisy, st["ref_mask"], st["tok_idx"], st["n_tokens"])
        a = a + self.s_to_token(self.s_step_norm(s))
        mask = st["token_mask"]
        a = self.token_norm(self.transformer(a, s, st["biases"], None if mask is None else _rep(mask, n)))
        r_update = self.atom_decoder(a, q, c, params, st["tok_idx"])
        t2, sd2 = t * t, sd * sd
        return (sd2 / (sd2 + t2))[:, None, None] * x_noisy + ((sd * t) / mx.sqrt(sd2 + t2))[:, None, None] * r_update

    def __call__(self, x_noisy, t_hat, num_diffusion_samples=1, **inputs) -> mx.array:
        """torch ``DiffusionModule.forward`` with its keyword names; returns x_denoised."""
        t = t_hat.reshape(-1)
        if t.size == 1:
            t = mx.broadcast_to(t, (x_noisy.shape[0],))
        return self.denoise(self.prepare(**inputs), x_noisy, t)


def noise_schedule(cfg: ModelConfig, steps: Optional[int], max_sigma: Optional[float]) -> np.ndarray:
    """``inference_noise_schedule`` then the ``max_inference_sigma`` cap, in torch's float32 arithmetic."""
    sh = cfg.structure_head
    sd = np.float32(sh.diffusion_module.sigma_data)
    steps = sh.inference_num_steps if steps is None else int(steps)
    if steps == 1:
        sched = np.array([sh.inference_s_max * sh.diffusion_module.sigma_data, 0.0], dtype=np.float32)
    else:
        inv_p = 1.0 / float(sh.inference_p)
        lo, hi = float(sh.inference_s_min) ** inv_p, float(sh.inference_s_max) ** inv_p
        k = np.arange(steps, dtype=np.float32)
        base = np.float32(hi) + (k / np.float32(steps - 1)) * np.float32(lo - hi)
        sched = np.concatenate([sd * base ** np.float32(sh.inference_p), np.zeros(1, np.float32)]).astype(np.float32)
    if max_sigma is not None:
        sched = np.concatenate([np.array([max_sigma], np.float32), sched[sched <= np.float32(max_sigma)]])
    return sched


def rotations_from_quaternions(q: mx.array) -> mx.array:
    """``_random_rotations`` from its normal draws: [n, 4] -> [n, 3, 3]."""
    scale = mx.sqrt(mx.sum(q * q, axis=1))
    q = q / mx.where(q[:, 0] < 0, -scale, scale)[:, None]
    r, i, j, k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    two_s = 2.0 / mx.sum(q * q, axis=-1)
    return mx.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], axis=-1).reshape(-1, 3, 3)


def _det3(m: mx.array) -> mx.array:
    return (m[:, 0, 0] * (m[:, 1, 1] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 1])
            - m[:, 0, 1] * (m[:, 1, 0] * m[:, 2, 2] - m[:, 1, 2] * m[:, 2, 0])
            + m[:, 0, 2] * (m[:, 1, 0] * m[:, 2, 1] - m[:, 1, 1] * m[:, 2, 0]))


def weighted_rigid_align(x: mx.array, x_gt: mx.array, w: mx.array, mask: mx.array) -> mx.array:
    """Kabsch: align x onto x_gt with weights (``_weighted_rigid_align``). The 3x3 SVDs run on the CPU
    stream (MLX's linear algebra is CPU-only); they are tiny."""
    w = (mask * w)[..., None]
    denom = mx.maximum(mx.sum(w, axis=-2, keepdims=True), 1e-8)
    mu = mx.sum(x * w, axis=-2, keepdims=True) / denom
    mu_gt = mx.sum(x_gt * w, axis=-2, keepdims=True) / denom
    x_c, g_c = x - mu, x_gt - mu_gt
    h = (w * g_c).swapaxes(-1, -2) @ x_c  # [B, 3, 3]
    u, _, vh = mx.linalg.svd(h, stream=mx.cpu)
    d = _det3(u @ vh)
    signs = mx.stack([mx.ones_like(d), mx.ones_like(d), d], axis=-1)  # diag(1, 1, det)
    r = (u * signs[:, None, :]) @ vh
    return x_c @ r.swapaxes(-1, -2) + mu_gt


class DiffusionSampler:
    def __init__(self, weights_dir: str, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.module = DiffusionModule(weights_dir, cfg)

    def sample(self, z_trunk, s_inputs, relative_position_encoding, ref_pos, ref_charge, ref_mask, ref_element,
               ref_atom_name_chars, ref_space_uid, tok_idx, token_attention_mask, num_diffusion_samples: int,
               draws: Draws, num_sampling_steps: Optional[int] = None, noise_scale: float = 1.003,
               step_scale: float = 1.5, max_inference_sigma: Optional[float] = 256.0,
               on_step: Optional[Callable[[int, mx.array], None]] = None, **_unused) -> mx.array:
        """torch ``DiffusionStructureHead.sample`` with its keyword names; returns [S, A, 3]."""
        sh = self.cfg.structure_head
        n = s_inputs.shape[0] * num_diffusion_samples
        A = tok_idx.shape[1]
        st = self.module.prepare(z_trunk, s_inputs, relative_position_encoding, ref_pos, ref_charge, ref_mask, ref_element,
                                 ref_atom_name_chars, ref_space_uid, tok_idx, token_attention_mask)
        sched = noise_schedule(self.cfg, num_sampling_steps, max_inference_sigma)
        gammas = [sh.gamma_0 if s > sh.gamma_min else 0.0 for s in sched]
        x = mx.array(sched[0]) * draws.diffusion_init((n, A, 3))
        atom_mask = _rep(ref_mask, n).astype(mx.float32)
        prev = None
        for step in range(len(sched) - 1):
            x, prev = self._augment(x, atom_mask, prev, draws, step)
            sigma_prev = float(sched[step])
            t_hat = sigma_prev * (1.0 + gammas[step + 1])
            eps = noise_scale * max(t_hat ** 2 - sigma_prev ** 2, 0.0) ** 0.5
            x_noisy = x + eps * draws.step_noise(step, (n, A, 3))
            x_den = self.module.denoise(st, x_noisy, mx.full((n,), t_hat))
            x_noisy = weighted_rigid_align(x_noisy, x_den, atom_mask, atom_mask)
            x = x_noisy + (step_scale * (float(sched[step + 1]) - t_hat)) * ((x_noisy - x_den) / t_hat)
            prev = x_den
            mx.eval(x, prev)
            if on_step is not None:
                on_step(step, x_den)
        return x

    @staticmethod
    def _augment(x, atom_mask, prev, draws: Draws, step: int):
        """Algorithm 19: centre on the masked atoms, random rotation, random translation."""
        m = atom_mask[..., None]
        mean = mx.sum(x * m, axis=1, keepdims=True) / mx.maximum(mx.sum(m, axis=1, keepdims=True), 1.0)
        x = x - mean
        prev = None if prev is None else prev - mean
        r = rotations_from_quaternions(draws.rotation_quat(step, x.shape[0]))
        x = x @ r
        prev = None if prev is None else prev @ r
        t = draws.translation(step, (x.shape[0], 1, 3))
        return x + t, (None if prev is None else prev + t)
