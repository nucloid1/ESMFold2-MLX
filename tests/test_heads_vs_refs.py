"""Distogram head and confidence head against torch-CPU captures.

References: scripts/capture_refs.py stage ``heads``: the real forward on CPU with synthetic ESMC
states, the checkpoint's 14 sampling steps and TWO diffusion samples (the confidence head repeats
the pair per sample), hooks on the distogram head, the confidence head and its trunk and pooling.
"""

import mlx.core as mx
import numpy as np
import pytest

from helpers import assert_live, load_ref, rel_err

from esmfold2_mlx.config import load_config
from esmfold2_mlx.heads import CONFIDENCE_KEYS, ConfidenceHead, DistogramHead
from esmfold2_mlx.weights import load_group

CASES = ["m49", "vegf1", "tnf60_msa"]
GATE = 1e-5
CONF_INPUTS = ("s_inputs", "z", "x_pred", "distogram_atom_idx", "token_attention_mask", "atom_to_token",
               "atom_attention_mask", "asym_id", "mol_type", "relative_position_encoding", "token_bonds_encoding")


@pytest.fixture(scope="module")
def head(weights_dir):
    return ConfidenceHead(weights_dir, load_config(weights_dir))


def _conf_inputs(ref):
    kw = {k: mx.array(ref[f"conf_{k}"]) for k in CONF_INPUTS}
    kw["num_diffusion_samples"] = int(ref["num_diffusion_samples"])
    return kw


@pytest.mark.parametrize("case", CASES)
def test_distogram_matches(weights_dir, case):
    ref = load_ref(case, "heads")
    assert_live(ref["disto_out"])
    got = DistogramHead(load_group(weights_dir, "distogram_head"))(mx.array(ref["z"]))
    assert got.shape == ref["disto_out"].shape
    assert rel_err(got, ref["disto_out"]) < GATE


@pytest.mark.parametrize("case", CASES)
def test_confidence_pair_front_matches(head, case):
    """z_base, repeated per sample, plus the embedded representative-atom distance bins."""
    ref = load_ref(case, "heads")
    assert_live(ref["conf_trunk_in"])
    assert rel_err(head.pair_front(**_conf_inputs(ref)), ref["conf_trunk_in"]) < GATE


@pytest.mark.parametrize("case", CASES)
def test_confidence_trunk_and_pooling_match(head, case):
    ref = load_ref(case, "heads")
    mask = mx.array(ref["conf_trunk_mask"])
    assert rel_err(head.trunk(mx.array(ref["conf_trunk_in"]), mask), ref["conf_trunk_out"]) < GATE
    assert_live(ref["conf_pool_out"])
    assert rel_err(head.pooling(mx.array(ref["conf_pool_in"]), mx.array(ref["conf_pool_mask"])), ref["conf_pool_out"]) < GATE


@pytest.mark.parametrize("case", CASES)
def test_confidence_head_outputs_match(head, case):
    ref = load_ref(case, "heads")
    got = head(**_conf_inputs(ref))
    errors = {}
    for k in CONFIDENCE_KEYS:
        want = ref[f"conf_out_{k}"]
        assert got[k].shape == want.shape, k
        errors[k] = rel_err(got[k], want) if np.linalg.norm(want) > 0 else float(np.abs(np.array(got[k])).max())
    assert all(e < GATE for e in errors.values()), errors


@pytest.mark.parametrize("case", CASES)
def test_confidence_keys_equal_torch_keys(head, case):
    ref = load_ref(case, "heads")
    torch_keys = {k[len("conf_out_"):] for k in ref if k.startswith("conf_out_")}
    assert set(CONFIDENCE_KEYS) == torch_keys


def test_two_samples_were_captured_and_scored_separately():
    ref = load_ref("vegf1", "heads")
    assert int(ref["num_diffusion_samples"]) == 2
    assert ref["conf_out_iptm"].shape == (2,)
    assert not np.allclose(ref["conf_out_plddt"][0], ref["conf_out_plddt"][1])


def test_confidence_head_is_bitwise_repeatable(head):
    ref = load_ref("vegf1", "heads")
    a, b = head(**_conf_inputs(ref)), head(**_conf_inputs(ref))
    assert all(np.array_equal(np.array(a[k]), np.array(b[k])) for k in CONFIDENCE_KEYS)


def test_samples_are_scored_one_at_a_time_so_peak_memory_does_not_scale_with_samples(head):
    """Checkpoint 2 found the 3-sample confidence head peaking at 22.75 GiB at 499 tokens (the whole
    trunk peaks at 8.9), above torch's chunked peak. Samples are independent here, so scoring them one
    at a time gives the same outputs at about a third of the peak."""
    ref = load_ref("vegf1", "heads")
    kw = _conf_inputs(ref)
    tripled = {**kw, "x_pred": mx.concatenate([kw["x_pred"], kw["x_pred"][:1]], axis=0), "num_diffusion_samples": 3}
    one = head(**{**kw, "x_pred": kw["x_pred"][:1], "num_diffusion_samples": 1})
    seen = []
    original = head.folding_trunk
    head.folding_trunk = lambda pair, mask: seen.append(pair.shape[0]) or original(pair, mask)
    try:
        three = head(**tripled)
    finally:
        head.folding_trunk = original
    assert seen == [1, 1, 1]  # the [S, L, L, c] pair stack never holds more than one sample
    for k in CONFIDENCE_KEYS:  # sample 0 of the batch equals the single-sample run
        assert rel_err(np.array(three[k])[0], np.array(one[k])[0]) < 1e-6, k
        assert np.array(three[k]).shape[0] == 3, k


def test_heads_consume_every_weight(head, weights_dir):
    DistogramHead(w := load_group(weights_dir, "distogram_head"))
    assert w.unused() == []
    assert head.weights.unused() == []
