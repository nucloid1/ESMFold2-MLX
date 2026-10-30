"""Inputs embedding and pair initialisation: the part of ``EsmFold2Model.forward`` before the loop
(model.py:1076-1221), fp32 apart from the atom encoder's mirrored bf16 casts (see atoms.py).

    x_inputs = [atom encoding (384), one_hot(res_type) (33), MSA profile (33), deletion mean (1)]
    z_init   = z_init_1(x)_i + z_init_2(x)_j + rel_pos(i, j) + token_bonds(i, j)
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .atoms import CHAR_VOCAB, AtomEncoder
from .config import ModelConfig
from .layers import Linear
from .weights import Weights, load_group

NUM_RES_TYPES = 33
MAX_ATOMIC_NUMBER = 128


def _one_hot(x: mx.array, n: int) -> mx.array:
    return (x.astype(mx.int32)[..., None] == mx.arange(n)).astype(mx.float32)


@dataclass
class Embedded:
    x_inputs: mx.array  # [B, L, 451]
    z_init: mx.array  # [B, L, L, 256]
    relative_position_encoding: mx.array
    token_bonds_encoding: mx.array
    pair_mask: mx.array  # [B, L, L] float
    ref_element_oh: mx.array | None = None  # masked atom one-hots and token map, for the diffusion module
    chars_oh: mx.array | None = None
    atom_to_token: mx.array | None = None


class RelativePositionEncoding:
    """``ResIdxAsymIdSymIdEntityIdEncoding``: one-hot relative residue, token and chain offsets."""

    def __init__(self, w: Weights, cfg: ModelConfig) -> None:
        self.embed = Linear.load(w, "embed")
        self.r, self.c = cfg.num_relative_residx_bins, cfg.num_relative_chain_bins

    def __call__(self, residue_index, asym_id, sym_id, entity_id, token_index) -> mx.array:
        r, c = self.r, self.c
        same_chain = asym_id[:, :, None] == asym_id[:, None, :]
        same_residue = residue_index[:, :, None] == residue_index[:, None, :]
        same_entity = entity_id[:, :, None] == entity_id[:, None, :]
        d_res = mx.clip(residue_index[:, :, None] - residue_index[:, None, :] + r, 0, 2 * r)
        d_res = mx.where(same_chain, d_res, 2 * r + 1)
        d_tok = mx.clip(token_index[:, :, None] - token_index[:, None, :] + r, 0, 2 * r)
        d_tok = mx.where(same_chain & same_residue, d_tok, 2 * r + 1)
        d_chain = mx.clip(sym_id[:, :, None] - sym_id[:, None, :] + c, 0, 2 * c)
        d_chain = mx.where(same_chain, 2 * c + 1, d_chain)
        feats = mx.concatenate([_one_hot(d_res, 2 * r + 2), _one_hot(d_tok, 2 * r + 2),
                                same_entity.astype(mx.float32)[..., None], _one_hot(d_chain, 2 * c + 2)], axis=-1)
        return self.embed(feats)


class Embedder:
    GROUPS = ("inputs_embedder", "z_init_1", "z_init_2", "rel_pos", "token_bonds")

    def __init__(self, weights_dir: str, cfg: ModelConfig) -> None:
        self.groups = {name: load_group(weights_dir, name) for name in self.GROUPS}
        g = self.groups
        self.atom_encoder = AtomEncoder(g["inputs_embedder"], "atom_attention_encoder", cfg.atom_encoder,
                                        window=cfg.sliding_window, structure_prediction=False)
        self.z_init_1 = Linear.load(g["z_init_1"], "")
        self.z_init_2 = Linear.load(g["z_init_2"], "")
        self.rel_pos = RelativePositionEncoding(g["rel_pos"], cfg)
        self.token_bonds = Linear.load(g["token_bonds"], "")
        for w in g.values():
            w.assert_all_used()

    @staticmethod
    def atom_one_hots(f: dict):
        """The masked atom one-hots and token map the forward builds (model.py:1099-1111)."""
        atom_mask = f["atom_attention_mask"].astype(mx.float32)
        element = _one_hot(f["ref_element"], MAX_ATOMIC_NUMBER) * atom_mask[..., None]
        chars = _one_hot(f["ref_atom_name_chars"], CHAR_VOCAB) * atom_mask[..., None, None]
        atom_to_token = f["atom_to_token"].astype(mx.int32) * f["atom_attention_mask"].astype(mx.int32)
        return element, chars, atom_to_token

    @staticmethod
    def atom_inputs(encoder: AtomEncoder, f: dict) -> mx.array:
        element, chars, atom_to_token = Embedder.atom_one_hots(f)
        a, _, _ = encoder(f["ref_pos"], f["atom_attention_mask"], f["ref_space_uid"], f["ref_charge"], element, chars, atom_to_token)
        return a

    def pair_init(self, x_inputs: mx.array, f: dict) -> Embedded:
        rel = self.rel_pos(f["residue_index"], f["asym_id"], f["sym_id"], f["entity_id"], f["token_index"])
        bonds = self.token_bonds(f["token_bonds"].astype(mx.float32))
        z_init = self.z_init_1(x_inputs)[:, :, None, :] + self.z_init_2(x_inputs)[:, None, :, :]
        z_init = z_init + rel + bonds
        tok = f["token_attention_mask"].astype(mx.float32)
        return Embedded(x_inputs, z_init, rel, bonds, tok[:, :, None] * tok[:, None, :])

    def __call__(self, f: dict) -> Embedded:
        tok = f["token_attention_mask"].astype(mx.float32)
        res_type = _one_hot(f["res_type"], NUM_RES_TYPES) * tok[..., None]
        if "msa" in f and f["msa"] is not None:
            msa_oh = _one_hot(f["msa"], NUM_RES_TYPES)
            if "msa_attention_mask" in f and f["msa_attention_mask"] is not None:
                m = f["msa_attention_mask"].astype(mx.float32)[..., None]
                count = mx.maximum(mx.sum(m, axis=1), 1.0)
                profile = mx.sum(msa_oh * m, axis=1) / count
            else:
                profile = mx.mean(msa_oh, axis=1)
        else:
            profile = res_type
        deletion_mean = f.get("deletion_mean")
        if deletion_mean is None:
            deletion_mean = mx.zeros(tok.shape)
        element, chars, atom_to_token = self.atom_one_hots(f)
        a, _, _ = self.atom_encoder(f["ref_pos"], f["atom_attention_mask"], f["ref_space_uid"], f["ref_charge"],
                                    element, chars, atom_to_token)
        x_inputs = mx.concatenate([a, res_type, profile, deletion_mean.astype(mx.float32)[..., None]], axis=-1)
        out = self.pair_init(x_inputs, f)
        out.ref_element_oh, out.chars_oh, out.atom_to_token = element, chars, atom_to_token
        mx.eval(out.x_inputs, out.z_init)
        return out
