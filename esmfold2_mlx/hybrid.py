"""``install``: make an esm ``EsmFold2Model`` run its post-ESMC forward in the MLX worker.

Anything that already drives esm's processor (or calls the model directly) keeps working; ESMC, if the
model has it attached and no ``lm_hidden_states`` are given, still runs in torch, and the rest runs in
MLX. The returned dict has the torch forward's keys, as torch tensors on the model's device.
"""

from __future__ import annotations

from .config import FoldOptions
from .worker import RemoteFold


def install(model, weights_dir: str, cap_gib: float | None = None) -> RemoteFold:
    import torch

    remote = RemoteFold(weights_dir, cap_gib=cap_gib)
    cfg = model.config

    def forward(*, lm_hidden_states=None, input_ids=None, num_loops=None, num_diffusion_samples=None,
                num_sampling_steps=None, lm_mask_pct=None, msa_max_depth=None, msa_column_mask_rate=None,
                noise_scale=None, step_scale=None, max_inference_sigma=256.0, disto_cond=None, disto_cond_mask=None,
                include_embeddings=False, **features):
        if include_embeddings:
            raise NotImplementedError("include_embeddings is not supported by the MLX forward")
        if disto_cond_mask is not None and bool(disto_cond_mask.any()):
            raise NotImplementedError("distogram conditioning is not implemented for ESMFold2; fold without it")
        if lm_hidden_states is None and input_ids is not None and getattr(model, "esmc", None) is not None:
            lm_hidden_states = model._compute_lm_hidden_states(
                input_ids, features["asym_id"], features["residue_index"], features["mol_type"],
                features["token_attention_mask"], lm_mask_pct=cfg.lm_mask_pct if lm_mask_pct is None else lm_mask_pct)
        lm = None if lm_hidden_states is None else lm_hidden_states.float().cpu().numpy()
        arrays = {k: v.detach().cpu().numpy() for k, v in features.items() if isinstance(v, torch.Tensor)}
        sh = cfg.structure_head
        options = FoldOptions(
            num_loops=num_loops, num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples or cfg.num_diffusion_samples, msa_max_depth=msa_max_depth,
            msa_column_mask_rate=cfg.msa_encoder.column_mask_rate if msa_column_mask_rate is None else msa_column_mask_rate,
            noise_scale=sh.noise_scale if noise_scale is None else noise_scale,
            step_scale=sh.step_scale if step_scale is None else step_scale, max_inference_sigma=max_inference_sigma,
            lm_dropout=cfg.lm_encoder.lm_dropout if cfg.lm_encoder.per_loop_lm_dropout else None)
        out = remote(arrays, lm, options)
        device = next(model.parameters()).device
        return {k: torch.from_numpy(v).to(device) for k, v in out.items()}

    model.forward = forward
    return remote
