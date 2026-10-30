"""Model configuration (read from the converter's ``config.json``, so runtime never imports esm) and
per-call fold options.

The dataclasses mirror the checkpoint's resolved ``config.json``, HF boilerplate removed. Fields with
no default are required: a converted config that lacks one fails loudly instead of running with a
guessed value.
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing
from dataclasses import dataclass


@dataclass(frozen=True)
class AtomEncoderConfig:
    hidden_size: int
    output_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    intermediate_size: int
    spatial_rope_base_frequency: float
    num_spatial_rope_pairs_per_axis: int
    num_uid_rope_pairs: int
    uid_rope_base_frequency: float
    expansion_ratio: int | None = None  # only the trunk's atom encoder carries these two
    token_hidden_size: int | None = None


@dataclass(frozen=True)
class DiffusionModuleConfig:
    sigma_data: float
    hidden_size: int
    fourier_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    intermediate_size: int
    pair_intermediate_size: int
    atom_encoder: AtomEncoderConfig
    token_hidden_size: int
    token_num_blocks: int
    token_num_heads: int
    transition_multiplier: int
    c_z: int
    c_s_inputs: int
    relpos_r_max: int
    relpos_s_max: int


@dataclass(frozen=True)
class StructureHeadConfig:
    diffusion_module: DiffusionModuleConfig
    num_distogram_bins: int
    num_diffusion_samples: int
    gamma_0: float
    gamma_min: float
    noise_scale: float
    step_scale: float
    inference_num_steps: int
    inference_s_max: float
    inference_s_min: float
    inference_p: float
    inference_sigma_cap: float


@dataclass(frozen=True)
class ConfidenceHeadConfig:
    num_hidden_layers: int
    num_plddt_bins: int
    num_pde_bins: int
    num_pae_bins: int
    min_dist: float
    max_dist: float
    distogram_bins: int
    eps: float
    num_attention_heads: int


@dataclass(frozen=True)
class MSAEncoderConfig:
    hidden_size: int
    outer_hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_width: int
    intermediate_size: int
    max_depth: int
    column_mask_rate: float
    outer_product_chunk_size: int | None = None


@dataclass(frozen=True)
class LMEncoderConfig:
    num_hidden_layers: int
    lm_dropout: float
    per_loop_lm_dropout: bool


@dataclass(frozen=True)
class ModelConfig:
    d_single: int  # checkpoint: hidden_size
    d_pair: int  # pairwise_hidden_size
    d_single_inputs: int  # single_inputs_size
    n_trunk_layers: int  # folding_trunk_num_hidden_layers
    n_coda_layers: int  # parcae_num_coda_layers
    n_lm_encoder_layers: int  # lm_encoder.num_hidden_layers
    num_loops: int
    n_sampling_steps: int  # structure_head.inference_num_steps
    sliding_window: int
    chunk_size: int
    pair_transition_intermediate_size: int
    num_relative_residx_bins: int
    num_relative_chain_bins: int
    num_res_types: int
    max_atomic_number: int
    char_vocab_size: int
    max_chars: int
    max_atoms_per_token: int
    atom_feature_dim: int
    lm_d_model: int
    lm_num_layers: int
    atom_encoder: AtomEncoderConfig
    structure_head: StructureHeadConfig
    confidence_head: ConfidenceHeadConfig
    msa_encoder: MSAEncoderConfig
    lm_encoder: LMEncoderConfig


@dataclass
class FoldOptions:
    """Per-call options; ``None`` means "the checkpoint's value"."""

    num_loops: int | None = None
    num_sampling_steps: int | None = None
    num_diffusion_samples: int = 1
    msa_max_depth: int | None = None
    msa_column_mask_rate: float = 0.1
    noise_scale: float = 1.003
    step_scale: float = 1.5
    max_inference_sigma: float = 256.0
    seed: int = 0
    lm_dropout: float | None = None  # esm _lm_dropout_context: None/0 keep the checkpoint's 0.25 per loop


def _build(cls, data: dict, where: str):
    """Instantiate dataclass ``cls`` from ``data``, recursing into nested dataclass fields."""
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name in data:
            value = data[field.name]
            if dataclasses.is_dataclass(hints[field.name]):
                value = _build(hints[field.name], value, f"{where}.{field.name}")
            kwargs[field.name] = value
        elif field.default is dataclasses.MISSING:
            raise KeyError(f"config is missing {where}.{field.name}")
    return cls(**kwargs)


def load_config(weights_dir: str) -> ModelConfig:
    with open(os.path.join(weights_dir, "config.json")) as handle:
        raw = json.load(handle)
    renamed = {
        "d_single": raw["hidden_size"],
        "d_pair": raw["pairwise_hidden_size"],
        "d_single_inputs": raw["single_inputs_size"],
        "n_trunk_layers": raw["folding_trunk_num_hidden_layers"],
        "n_coda_layers": raw["parcae_num_coda_layers"],
        "n_lm_encoder_layers": raw["lm_encoder"]["num_hidden_layers"],
        "n_sampling_steps": raw["structure_head"]["inference_num_steps"],
    }
    return _build(ModelConfig, {**raw, **renamed}, "config")
