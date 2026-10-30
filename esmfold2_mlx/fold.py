"""``MLXFold``: everything ``EsmFold2Model.forward`` does after ESMC (model.py:1008-1401), in MLX.

    embed -> looped trunk (LM shim, MSA encoder, 21 steps, coda) -> distogram head
          -> diffusion sampler -> confidence head

Inputs are the esm builder's feature dict (numpy or MLX) and the ESMC hidden states; outputs are
numpy arrays under exactly the keys the torch forward returns. Random draws come from ``draws``
(MLX's own generator by default; recorded torch draws for parity runs) in the order torch makes them.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

from .config import FoldOptions, load_config
from .diffusion import DiffusionSampler
from .embed import Embedder
from .heads import ConfidenceHead, DistogramHead
from .rng import Draws, MLXDraws
from .trunk import ParcaeTrunk
from .weights import load_group

MSA_KEYS = ("msa", "msa_attention_mask", "has_deletion", "deletion_value")


class MLXFold:
    def __init__(self, weights_dir: str) -> None:
        self.cfg = load_config(weights_dir)
        self.embedder = Embedder(weights_dir, self.cfg)
        self.trunk = ParcaeTrunk(weights_dir, self.cfg)
        disto = load_group(weights_dir, "distogram_head")
        self.distogram = DistogramHead(disto)
        disto.assert_all_used()
        self.sampler = DiffusionSampler(weights_dir, self.cfg)
        self.confidence = ConfidenceHead(weights_dir, self.cfg)

    def fold(self, features: dict, lm_hidden_states, options: FoldOptions, draws: Optional[Draws] = None) -> dict[str, np.ndarray]:
        draws = MLXDraws(options.seed) if draws is None else draws
        f = {k: (v if isinstance(v, mx.array) else mx.array(np.asarray(v))) for k, v in features.items() if v is not None}
        if f["res_type"].shape[0] != 1:
            raise NotImplementedError("MLXFold folds one complex at a time (feature batch size 1), as the esm processor does")
        lm = None if lm_hidden_states is None else (lm_hidden_states if isinstance(lm_hidden_states, mx.array)
                                                    else mx.array(np.asarray(lm_hidden_states)))
        if "disto_cond_mask" in f and bool(mx.any(f["disto_cond_mask"]).item()):
            raise NotImplementedError("distogram conditioning is not implemented for ESMFold2 (as in torch); fold without it")

        emb = self.embedder(f)
        msa = {k: f[k] for k in MSA_KEYS} if "msa" in f else None
        loops = self.cfg.num_loops if options.num_loops is None else options.num_loops
        z = self.trunk(emb.x_inputs, emb.z_init, lm, emb.pair_mask, num_loops=loops, draws=draws, msa_inputs=msa,
                       lm_dropout=options.lm_dropout, msa_max_depth=options.msa_max_depth,
                       msa_column_mask_rate=options.msa_column_mask_rate)
        del lm, msa
        distogram = self.distogram(z)
        mx.eval(distogram)
        mx.clear_cache()

        tok_mask = f["token_attention_mask"]
        coords = self.sampler.sample(
            z_trunk=z, s_inputs=emb.x_inputs, relative_position_encoding=emb.relative_position_encoding,
            ref_pos=f["ref_pos"], ref_charge=f["ref_charge"], ref_mask=f["atom_attention_mask"], ref_element=emb.ref_element_oh,
            ref_atom_name_chars=emb.chars_oh, ref_space_uid=f["ref_space_uid"], tok_idx=emb.atom_to_token,
            token_attention_mask=tok_mask, num_diffusion_samples=options.num_diffusion_samples, draws=draws,
            num_sampling_steps=options.num_sampling_steps, noise_scale=options.noise_scale, step_scale=options.step_scale,
            max_inference_sigma=options.max_inference_sigma)
        mx.clear_cache()

        confidence = self.confidence(
            s_inputs=emb.x_inputs, z=z, x_pred=coords, distogram_atom_idx=f["distogram_atom_idx"], token_attention_mask=tok_mask,
            atom_to_token=emb.atom_to_token, atom_attention_mask=f["atom_attention_mask"], asym_id=f["asym_id"],
            mol_type=f["mol_type"], num_diffusion_samples=options.num_diffusion_samples,
            relative_position_encoding=emb.relative_position_encoding, token_bonds_encoding=emb.token_bonds_encoding)
        atom_mask = f["atom_attention_mask"]
        out = {"distogram_logits": distogram, "sample_atom_coords": coords, **confidence,
               "atom_pad_mask": atom_mask[None] if atom_mask.ndim == 1 else atom_mask,
               "residue_index": f["residue_index"], "entity_id": f["entity_id"]}
        result = {k: np.array(v) for k, v in out.items()}
        mx.clear_cache()
        return result
