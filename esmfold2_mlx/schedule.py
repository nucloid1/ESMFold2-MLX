"""The sampler's noise schedule, numpy only (shared by the MLX sampler and the torch-side parent)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import ModelConfig


def noise_schedule(cfg: ModelConfig, steps: Optional[int], max_sigma: Optional[float]) -> np.ndarray:
    """``inference_noise_schedule`` then the ``max_inference_sigma`` cap, in torch's float32 arithmetic.
    The cap filters the schedule (esm's docstring says it inflates it to keep the step count; the code
    does not), so the checkpoint's 14 steps give 10 denoiser calls."""
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
