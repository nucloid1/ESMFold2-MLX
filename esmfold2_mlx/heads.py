"""Distogram head (model.py:1322) and confidence head (model.py:104-403), fp32.

The confidence head runs once per diffusion sample (the pair is repeated per sample):

    pair  = z_norm(z) + rel_pos + token_bonds + s->z terms, repeated per sample
          + embed(bin of the representative-atom distance)          ``pair_front``
    pair  = pair + FoldingTrunk(pair)       (the trunk's full output is added, as torch does)
    single = RowAttentionPooling(pair)
    pLDDT / resolved from per-atom-slot weight tables; PAE / PDE from the pair;
    pTM / ipTM / per-chain-pair ipTM from the PAE distribution.
"""

from __future__ import annotations

import mlx.core as mx

from .config import ModelConfig
from .layers import LayerNorm, Linear
from .trunk import FoldingTrunk
from .weights import Weights, load_group

_EPS = 1e-6  # model.py _EPS
_NONPOLYMER_ID = 4
CONFIDENCE_KEYS = ("plddt_logits", "plddt", "plddt_per_atom", "plddt_ca", "complex_plddt", "complex_iplddt",
                   "pae_logits", "pae", "pde_logits", "pde", "resolved_logits", "ptm", "iptm", "pair_chains_iptm")


class DistogramHead:
    def __init__(self, w: Weights) -> None:
        self.linear = Linear.load(w, "")

    def __call__(self, z: mx.array) -> mx.array:
        return self.linear(z + z.swapaxes(1, 2))


def categorical_mean(logits: mx.array, start: float, end: float) -> mx.array:
    """Expected value over ``n`` evenly spaced bins on [start, end] (layers.py ``_categorical_mean``)."""
    n = logits.shape[-1]
    edges = mx.linspace(start, end, n + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return (mx.softmax(logits, axis=-1) @ centers[:, None]).squeeze(-1)


def intra_token_index(atom_to_token: mx.array) -> mx.array:
    """Position of each atom within its token (atoms of a token are contiguous): layers.py ``_compute_intra_token_idx``."""
    same_as_prev = mx.concatenate([mx.zeros((atom_to_token.shape[0], 1), dtype=mx.bool_),
                                   atom_to_token[:, 1:] == atom_to_token[:, :-1]], axis=1)
    count = mx.cumsum(mx.ones_like(atom_to_token), axis=-1)
    start = mx.cummax(mx.where(same_as_prev, 0, count), axis=-1)
    return count - start


def _gather_rows(x: mx.array, idx: mx.array) -> mx.array:
    """x [B, L, d], idx [B, A] -> [B, A, d] (torch.gather along dim 1)."""
    return mx.take_along_axis(x, idx[..., None].astype(mx.int32), axis=1)


def _distances(coords: mx.array) -> mx.array:
    """Exact pairwise Euclidean distances (torch.cdist with donot_use_mm_for_euclid_dist)."""
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    return mx.sqrt(mx.sum(diff * diff, axis=-1))


class RowAttentionPooling:
    def __init__(self, w: Weights, prefix: str) -> None:
        self.attn_proj = Linear.load(w, f"{prefix}.attn_proj")
        self.out_proj = Linear.load(w, f"{prefix}.out_proj")

    def __call__(self, z: mx.array, mask: mx.array) -> mx.array:
        scores = self.attn_proj(z).squeeze(-1)  # [B, L, L]
        scores = scores + mx.where(mask[:, None, :].astype(mx.bool_), 0.0, -1e9)
        weights = mx.softmax(scores, axis=-1)
        pooled = (weights[:, :, None, :] @ z).squeeze(2)  # sum_m w[b,n,m] z[b,n,m,:]
        return self.out_proj(pooled)


class ConfidenceHead:
    def __init__(self, weights_dir: str, cfg: ModelConfig) -> None:
        self.weights = load_group(weights_dir, "confidence_head")
        w = self.weights
        self.boundaries = w["boundaries"]
        self.dist_embed = w["dist_bin_pairwise_embed.weight"]
        self.s_inputs_norm = LayerNorm.load(w, "s_inputs_norm")
        self.z_norm = LayerNorm.load(w, "z_norm")
        self.s_to_z = Linear.load(w, "s_to_z")
        self.s_to_z_t = Linear.load(w, "s_to_z_transpose")
        self.prod_in1 = Linear.load(w, "s_to_z_prod_in1")
        self.prod_in2 = Linear.load(w, "s_to_z_prod_in2")
        self.prod_out = Linear.load(w, "s_to_z_prod_out")
        self.pooling = RowAttentionPooling(w, "row_attention_pooling")
        self.folding_trunk = FoldingTrunk(w, "folding_trunk", cfg.confidence_head.num_hidden_layers)
        self.plddt_ln = LayerNorm.load(w, "plddt_ln")
        self.plddt_weight = w["plddt_weight"]  # [23, d_single, n_plddt_bins]
        self.pae_ln, self.pae_head = LayerNorm.load(w, "pae_ln"), Linear.load(w, "pae_head")
        self.pde_ln, self.pde_head = LayerNorm.load(w, "pde_ln"), Linear.load(w, "pde_head")
        self.resolved_ln = LayerNorm.load(w, "resolved_ln")
        self.resolved_weight = w["resolved_weight"]  # [23, d_single, 2]
        w.assert_all_used()

    @staticmethod
    def _repeat(x: mx.array, n: int) -> mx.array:
        return x if n == 1 else mx.repeat(x, n, axis=0)

    @staticmethod
    def _flatten_samples(x: mx.array) -> mx.array:
        return x.reshape(-1, *x.shape[2:]) if x.ndim == 4 else x

    def pair_front(self, s_inputs, z, x_pred, distogram_atom_idx, token_attention_mask, num_diffusion_samples=1,
                   relative_position_encoding=None, token_bonds_encoding=None, **_unused) -> mx.array:
        s = self.s_inputs_norm(s_inputs)
        z_base = self.z_norm(z)
        if relative_position_encoding is not None:
            z_base = z_base + relative_position_encoding
        if token_bonds_encoding is not None:
            z_base = z_base + token_bonds_encoding
        z_base = z_base + self.s_to_z(s)[:, :, None, :]
        z_base = z_base + self.s_to_z_t(s)[:, None, :, :]
        z_base = z_base + self.prod_out(self.prod_in1(s)[:, :, None, :] * self.prod_in2(s)[:, None, :, :])
        n = num_diffusion_samples
        pair = self._repeat(z_base, n)
        rep = _gather_rows(self._flatten_samples(x_pred), self._repeat(distogram_atom_idx, n))
        bins = mx.sum(_distances(rep)[..., None] > self.boundaries, axis=-1)
        return pair + self.dist_embed[bins]

    def trunk(self, pair: mx.array, pair_mask: mx.array) -> mx.array:
        return self.folding_trunk(pair, pair_mask)

    def __call__(self, s_inputs, z, x_pred, distogram_atom_idx, token_attention_mask, atom_to_token,
                 atom_attention_mask, asym_id, mol_type, num_diffusion_samples=1, relative_position_encoding=None,
                 token_bonds_encoding=None) -> dict[str, mx.array]:
        n = num_diffusion_samples
        pair = self.pair_front(s_inputs, z, x_pred, distogram_atom_idx, token_attention_mask, n,
                               relative_position_encoding, token_bonds_encoding)
        mask = self._repeat(token_attention_mask, n).astype(mx.float32)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        pair = pair + self.folding_trunk(pair, pair_mask)
        single = self.pooling(pair, mask)
        pae_logits = self.pae_head(self.pae_ln(pair))
        pde_logits = self.pde_head(self.pde_ln(pair))
        out = self._finish(single, pae_logits, pde_logits, x_pred, distogram_atom_idx, mask, atom_to_token,
                           atom_attention_mask, asym_id, mol_type, n)
        mx.eval(out)
        return out

    def _finish(self, single, pae_logits, pde_logits, x_pred, distogram_atom_idx, mask, atom_to_token,
                atom_attention_mask, asym_id, mol_type, n) -> dict[str, mx.array]:
        x = self._flatten_samples(x_pred)
        a2t = self._repeat(atom_to_token, n).astype(mx.int32)
        atom_mask = self._repeat(atom_attention_mask, n).astype(mx.float32)
        rep_idx = self._repeat(distogram_atom_idx, n).astype(mx.int32)
        dist = _distances(_gather_rows(x, rep_idx))
        S, L = single.shape[0], single.shape[1]

        s_atoms = _gather_rows(single, a2t)  # [S, A, d_single]
        intra = mx.minimum(intra_token_index(a2t), self.plddt_weight.shape[0] - 1)
        plddt_logits = (self.plddt_ln(s_atoms)[..., None, :] @ self.plddt_weight[intra]).squeeze(-2)
        plddt_atom = categorical_mean(plddt_logits, 0.0, 1.0)

        rows = mx.broadcast_to(mx.arange(S)[:, None], a2t.shape)
        plddt_sum = mx.zeros((S, L)).at[rows, a2t].add(plddt_atom * atom_mask)
        atom_count = mx.zeros((S, L)).at[rows, a2t].add(atom_mask)
        plddt = plddt_sum / mx.maximum(atom_count, 1e-6)
        complex_plddt = mx.sum(plddt_atom * atom_mask, axis=-1) / (mx.sum(atom_mask, axis=-1) + _EPS)

        asym = self._repeat(asym_id, n)
        is_ligand = (self._repeat(mol_type, n) == _NONPOLYMER_ID).astype(mx.float32)
        inter_chain = (asym[:, :, None] != asym[:, None, :]).astype(mx.float32)
        interface = mx.max((dist < 8).astype(mx.float32) * inter_chain * (1.0 - is_ligand)[..., None], axis=-1)
        weight = mx.where(is_ligand > 0, 2.0, interface)
        atom_w = atom_mask * _gather_rows(weight[..., None], a2t).squeeze(-1)
        complex_iplddt = mx.sum(plddt_atom * atom_w, axis=-1) / (mx.sum(atom_w, axis=-1) + _EPS)
        plddt_ca = mx.take_along_axis(plddt_atom, rep_idx, axis=1)

        resolved_logits = (self.resolved_ln(s_atoms)[..., None, :] @ self.resolved_weight[intra]).squeeze(-2)

        n_bins = pae_logits.shape[-1]
        width = 32.0 / n_bins
        centers = mx.arange(0.5 * width, 32.0, width)
        n_res = mx.sum(mask, axis=-1, keepdims=True)
        d0 = 1.24 * (mx.maximum(n_res, 19.0) - 15) ** (1.0 / 3.0) - 1.8
        tm_per_bin = 1.0 / (1.0 + (centers / d0) ** 2)  # [S, n_bins]
        tm = mx.sum(mx.softmax(pae_logits, axis=-1) * tm_per_bin[:, None, None, :], axis=-1)  # [S, L, L]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        ptm = mx.max(mx.sum(tm * pair_mask, axis=-1) / (mx.sum(pair_mask, axis=-1) + _EPS), axis=-1)
        inter = inter_chain * pair_mask
        iptm = mx.max(mx.sum(tm * inter, axis=-1) / (mx.sum(inter, axis=-1) + _EPS), axis=-1)

        n_chains = int(mx.max(asym).item()) + 1 if S > 0 else 1
        table = [[mx.zeros((S,)) for _ in range(n_chains)] for _ in range(n_chains)]
        for c1 in range(n_chains):
            chain_c1 = (asym == c1).astype(mx.float32) * mask
            if float(mx.sum(chain_c1).item()) == 0:
                continue
            col = chain_c1[:, None, :]
            avg = mx.sum(tm * col, axis=-1) / (mx.sum(col, axis=-1) + _EPS)  # [S, L] mean over chain-c1 columns
            for c2 in range(n_chains):
                chain_c2 = (asym == c2).astype(mx.float32) * mask
                rowvals = mx.where(chain_c2 == 0, -mx.inf, avg)
                table[c1][c2] = mx.maximum(mx.max(rowvals, axis=-1), 0.0)
        pair_chains_iptm = mx.stack([mx.stack(row, axis=-1) for row in table], axis=-2)  # [S, c1, c2]

        return {
            "plddt_logits": plddt_logits, "plddt": plddt, "plddt_per_atom": plddt_atom, "plddt_ca": plddt_ca,
            "complex_plddt": complex_plddt, "complex_iplddt": complex_iplddt,
            "pae_logits": pae_logits, "pae": categorical_mean(pae_logits, 0.0, 32.0),
            "pde_logits": pde_logits, "pde": categorical_mean(pde_logits, 0.0, 32.0),
            "resolved_logits": resolved_logits, "ptm": ptm, "iptm": iptm, "pair_chains_iptm": pair_chains_iptm,
        }
