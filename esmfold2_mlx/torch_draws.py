"""--match-torch-rng: reproduce torch's random stream for a seed without running the model.

The torch forward, under the rescore worker's seed context, makes its random draws in this order (all
on the forward's device, so ``device`` picks the generator: CPU and MPS streams differ):

    pair init          nn.init.trunc_normal_ on [1, L, L, c]           (model.py _init_pair_state)
    MSA column mask    torch.rand(B, L), once, if depth > 1 and rate > 0
    per loop step      F.dropout of the LM pair (if per-loop dropout is on), then
                       torch.randperm(depth - 1) (if depth > max_depth)
    sampler            torch.randn(S, A, 3); per step torch.randn(S, 4), torch.randn(S, 1, 3), torch.randn(S, A, 3)

Calling the same functions with the same shapes in the same order draws the same numbers. The result
is a dict keyed like ``scripts/torch_model.DrawRecorder`` records, consumed by ``rng.ReplayDraws``.
Parent process only (imports torch).
"""

import torch  # first

import math

import numpy as np

from .config import FoldOptions, ModelConfig
from .schedule import noise_schedule


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().copy()


def draw_torch_stream(features: dict, options: FoldOptions, cfg: ModelConfig, seed: int, device: str = "mps",
                      has_lm: bool = True) -> dict[str, np.ndarray]:
    from esm.models.esmfold2.processor import _seed_context

    L = int(np.asarray(features["res_type"]).shape[1])
    A = int(np.asarray(features["atom_attention_mask"]).shape[1])
    msa = features.get("msa")
    M = 0 if msa is None else int(np.asarray(msa).shape[1])
    c = cfg.d_pair
    loops = (cfg.num_loops if options.num_loops is None else options.num_loops) + 1
    p = float(options.lm_dropout) if options.lm_dropout else (cfg.lm_encoder.lm_dropout if cfg.lm_encoder.per_loop_lm_dropout else 0.0)
    max_depth = cfg.msa_encoder.max_depth if options.msa_max_depth is None else options.msa_max_depth
    rate = cfg.msa_encoder.column_mask_rate if options.msa_column_mask_rate is None else options.msa_column_mask_rate
    S = options.num_diffusion_samples
    steps = len(noise_schedule(cfg, options.num_sampling_steps, options.max_inference_sigma)) - 1
    out: dict = {}
    with torch.no_grad(), _seed_context(seed):
        std = math.sqrt(2.0 / (5.0 * c))
        state = torch.empty((1, L, L, c), dtype=torch.float32, device=device)
        out["_init_pair_state:trunc_normal_/0"] = _np(torch.nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std))
        if M > 1 and rate > 0:
            out["maybe_apply_msa_column_masking:rand/0"] = _np(torch.rand(1, L, device=device))
        subsample = max_depth is not None and M > 1 and M > max_depth
        for loop in range(loops):
            if has_lm and p > 0:
                kept = torch.nn.functional.dropout(torch.ones((1, L, L, c), device=device), p=p, training=True)
                out[f"_run_one_loop:dropout/{loop}"] = _np(kept != 0)
            if subsample:
                out[f"maybe_subsample_msa:randperm/{loop}"] = _np(torch.randperm(M - 1, device=device))
        out["sample:randn/0"] = _np(torch.randn(S, A, 3, device=device, dtype=torch.float32))
        for step in range(steps):
            out[f"_random_rotations:randn/{step}"] = _np(torch.randn((S, 4), dtype=torch.float32, device=device))
            out[f"_center_random_augmentation:randn_like/{step}"] = _np(torch.randn((S, 1, 3), dtype=torch.float32, device=device))
            out[f"sample:randn_like/{step}"] = _np(torch.randn((S, A, 3), dtype=torch.float32, device=device))
    return out
