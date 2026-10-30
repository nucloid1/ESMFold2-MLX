"""The MSA encoder against torch-CPU captures.

Every fold runs it: the esm featurizer supplies a query-only MSA in single-sequence mode, and with
``msa_encoder.overwrite`` the encoder's output replaces ``z_init`` as the per-loop injection. These
depth-1 references come from the ``trunk`` capture (stage file ``msa_encoder``: loop 0, block 0's
outer-product mean and pair-weighted averaging, blocks 0 and 3, the whole encoder).
"""

import mlx.core as mx
import numpy as np
import pytest

from helpers import assert_live, load_ref, rel_err

from esmfold2_mlx.config import load_config
from esmfold2_mlx.msa import MSAEncoder, MSAEncoderBlock, MSAPairWeightedAveraging, OuterProductMean
from esmfold2_mlx.weights import load_group

CASES = ["m49", "vegf1"]
GATE = 1e-5


def _w(weights_dir):
    return load_group(weights_dir, "msa_encoder")


def _a(ref, name):
    return mx.array(ref[name])


@pytest.mark.parametrize("case", CASES)
def test_opm_matches(weights_dir, case):
    ref = load_ref(case, "msa_encoder")
    assert_live(ref["opm0_out"])
    got = OuterProductMean(_w(weights_dir), "blocks.0.outer_product_mean")(_a(ref, "opm0_m"), _a(ref, "opm0_mask"))
    assert rel_err(got, ref["opm0_out"]) < GATE


@pytest.mark.parametrize("case", CASES)
def test_pair_weighted_averaging_matches(weights_dir, case):
    ref = load_ref(case, "msa_encoder")
    assert_live(ref["pwa0_out"])
    op = MSAPairWeightedAveraging(_w(weights_dir), "blocks.0.msa_pair_weighted_averaging", n_heads=8, head_width=16)
    got = op(_a(ref, "pwa0_m"), _a(ref, "pwa0_pair"), _a(ref, "pwa0_pair_mask"))
    assert rel_err(got, ref["pwa0_out"]) < GATE


@pytest.mark.parametrize("block,final", [(0, False), (3, True)])
@pytest.mark.parametrize("case", CASES)
def test_msa_block_matches(weights_dir, case, block, final):
    ref = load_ref(case, "msa_encoder")
    assert_live(ref[f"blk{block}_pair_out"])
    op = MSAEncoderBlock(_w(weights_dir), f"blocks.{block}", final=final, n_heads=8, head_width=16)
    m, pair = op(_a(ref, f"blk{block}_m"), _a(ref, f"blk{block}_pair"), _a(ref, f"blk{block}_msa_mask"), _a(ref, f"blk{block}_pair_mask"))
    assert rel_err(pair, ref[f"blk{block}_pair_out"]) < GATE
    if not final:
        assert_live(ref[f"blk{block}_m_out"])
        assert rel_err(m, ref[f"blk{block}_m_out"]) < GATE


@pytest.mark.parametrize("case", CASES)
def test_msa_encoder_matches(weights_dir, case):
    ref = load_ref(case, "msa_encoder")
    assert_live(ref["out"])
    encoder = MSAEncoder(weights_dir, load_config(weights_dir))
    got = encoder.encode(_a(ref, "x_pair"), _a(ref, "x_inputs"), _a(ref, "msa_oh"), _a(ref, "has_deletion"),
                         _a(ref, "deletion_value"), _a(ref, "msa_attention_mask"))
    assert rel_err(got, ref["out"]) < GATE


@pytest.mark.parametrize("case", CASES)
def test_feature_layout_preparation_matches_the_encoder_inputs(weights_dir, case):
    """Features arrive as [B, M, L]; the encoder wants one-hot [B, L, M, 33] with padding zeroed."""
    ref, inputs = load_ref(case, "msa_encoder"), load_ref(case, "trunk_inputs")
    msa_oh, hd, dv, mask = MSAEncoder.prepare(_a(inputs, "msa"), _a(inputs, "msa_attention_mask"),
                                              _a(inputs, "has_deletion"), _a(inputs, "deletion_value"))
    for got, name in ((msa_oh, "msa_oh"), (hd, "has_deletion"), (dv, "deletion_value"), (mask, "msa_attention_mask")):
        np.testing.assert_array_equal(np.array(got), ref[name])


def test_msa_encoder_consumes_every_weight(weights_dir):
    encoder = MSAEncoder(weights_dir, load_config(weights_dir))
    assert encoder.weights.unused() == []
