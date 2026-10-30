"""The pair-trunk block (tri_mul_out, tri_mul_in, pair_transition) against torch-CPU activations.

References come from scripts/capture_refs.py (stage ``pair_block``): the real forward on CPU with no
ESMC hidden states, one loop, hooks on blocks 0, 23 and 47 and on their three sub-ops. The input of
block 23 and 47 is therefore the real propagated pair stream. Real-loop captures with the language
model attached are back-filled once Task 2's corpus exists.
"""

import mlx.core as mx
import numpy as np
import pytest

from helpers import assert_live, load_ref, rel_err

from esmfold2_mlx.layers import Transition, TriangleMultiplication
from esmfold2_mlx.trunk import PairUpdateBlock
from esmfold2_mlx.weights import load_group

CASES = ["m49", "vegf1"]
BLOCKS = [0, 23, 47]
GATE = 1e-5


@pytest.fixture(scope="module")
def trunk_weights(weights_dir):
    return weights_dir


def _fresh(weights_dir):
    return load_group(weights_dir, "folding_trunk")


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("case", CASES)
def test_tri_mul_out_matches(trunk_weights, case, block):
    ref = load_ref(case, "pair_block")
    want = ref[f"b{block}_tri_out_out"]
    assert_live(want)
    op = TriangleMultiplication(_fresh(trunk_weights), f"blocks.{block}.tri_mul_out._engine", outgoing=True)
    got = op(mx.array(ref[f"b{block}_in"]), mx.array(ref["mask"]))
    assert rel_err(got, want) < GATE


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("case", CASES)
def test_tri_mul_in_matches(trunk_weights, case, block):
    ref = load_ref(case, "pair_block")
    want = ref[f"b{block}_tri_in_out"]
    assert_live(want)
    op = TriangleMultiplication(_fresh(trunk_weights), f"blocks.{block}.tri_mul_in._engine", outgoing=False)
    got = op(mx.array(ref[f"b{block}_tri_in_in"]), mx.array(ref["mask"]))
    assert rel_err(got, want) < GATE


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("case", CASES)
def test_transition_matches(trunk_weights, case, block):
    ref = load_ref(case, "pair_block")
    want = ref[f"b{block}_trans_out"]
    assert_live(want)
    op = Transition(_fresh(trunk_weights), f"blocks.{block}.pair_transition")
    got = op(mx.array(ref[f"b{block}_trans_in"]))
    assert rel_err(got, want) < GATE


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("case", CASES)
def test_pair_block_matches(trunk_weights, case, block):
    ref = load_ref(case, "pair_block")
    want = ref[f"b{block}_out"]
    assert_live(want)
    assert_live(ref[f"b{block}_in"])
    got = PairUpdateBlock(_fresh(trunk_weights), f"blocks.{block}")(mx.array(ref[f"b{block}_in"]), mx.array(ref["mask"]))
    assert got.shape == tuple(want.shape)
    assert rel_err(got, want) < GATE


def test_block_consumes_every_weight_under_its_prefix(trunk_weights):
    w = _fresh(trunk_weights)
    PairUpdateBlock(w, "blocks.5")
    w.assert_all_used("blocks.5.")
    assert len(w.unused("blocks.4.")) > 0  # the accounting is per prefix: untouched blocks stay unread


def test_mask_excludes_padded_tokens(trunk_weights):
    rng = np.random.default_rng(0)
    z = (rng.standard_normal((1, 45, 45, 256)) * 0.5).astype(np.float32)
    mask = np.ones((1, 45, 45), np.float32)
    mask[:, 40:, :] = 0
    mask[:, :, 40:] = 0
    block = PairUpdateBlock(_fresh(trunk_weights), "blocks.0")
    padded = np.array(block(mx.array(z), mx.array(mask)))
    plain = np.array(block(mx.array(z[:, :40, :40]), mx.array(mask[:, :40, :40])))
    assert_live(plain)
    assert rel_err(padded[:, :40, :40], plain) < 1e-6


def test_block_without_mask_equals_all_ones_mask(trunk_weights):
    rng = np.random.default_rng(1)
    z = mx.array((rng.standard_normal((1, 16, 16, 256)) * 0.5).astype(np.float32))
    block = PairUpdateBlock(_fresh(trunk_weights), "blocks.0")
    assert rel_err(block(z, None), block(z, mx.ones((1, 16, 16)))) == 0.0
