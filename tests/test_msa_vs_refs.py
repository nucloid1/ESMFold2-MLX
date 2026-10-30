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
from esmfold2_mlx.rng import ReplayDraws
from esmfold2_mlx.weights import load_group

CASES = ["m49", "vegf1", "m49_q", "tnf60_msa", "tnf60_shallow"]
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
    """Features arrive as [B, M, L]; the encoder wants one-hot [B, L, M, 33] with padding zeroed.
    Deeper than max_depth, loop 0 sees the recorded row subset, so this also checks the replayed rows."""
    ref, inputs = load_ref(case, "msa_encoder"), load_ref(case, "trunk_inputs")
    msa = {k: _a(inputs, k) for k in ("msa", "msa_attention_mask", "has_deletion", "deletion_value")}
    depth, max_depth = msa["msa"].shape[1], int(inputs["max_depth"])
    if depth > max_depth:
        rows = ReplayDraws(load_ref(case, "trunk_draws")).msa_row_indices(0, depth, max_depth)
        msa = {k: v[:, rows] for k, v in msa.items()}
    msa_oh, hd, dv, mask = MSAEncoder.prepare(**msa)
    for got, name in ((msa_oh, "msa_oh"), (hd, "has_deletion"), (dv, "deletion_value"), (mask, "msa_attention_mask")):
        np.testing.assert_array_equal(np.array(got), ref[name])


def test_msa_encoder_consumes_every_weight(weights_dir):
    encoder = MSAEncoder(weights_dir, load_config(weights_dir))
    assert encoder.weights.unused() == []


def test_deep_msa_is_real_and_subsampled_every_loop():
    """tnf60_msa: a real 2225-row alignment on the TNF chain, the binder without one (mixed chains)."""
    inputs, draws = load_ref("tnf60_msa", "trunk_inputs"), load_ref("tnf60_msa", "trunk_draws")
    assert inputs["msa"].shape[1] > int(inputs["max_depth"])  # deeper than max_depth: subsampled
    assert sum(k.startswith("maybe_subsample_msa:randperm/") for k in draws) == int(inputs["total_steps"])
    assert sum(k.startswith("maybe_apply_msa_column_masking:rand/") for k in draws) == 1


def test_rows_below_max_depth_are_column_masked_but_never_subsampled():
    inputs, draws = load_ref("tnf60_shallow", "trunk_inputs"), load_ref("tnf60_shallow", "trunk_draws")
    assert 1 < inputs["msa"].shape[1] <= int(inputs["max_depth"])
    assert not any(k.startswith("maybe_subsample_msa:randperm/") for k in draws)
    assert sum(k.startswith("maybe_apply_msa_column_masking:rand/") for k in draws) == 1


def test_query_only_a3m_is_depth_one():
    assert load_ref("m49_q", "trunk_inputs")["msa"].shape[1] == 1


@pytest.mark.parametrize("chunk", [1, 7, 50])
def test_opm_chunking_does_not_change_the_result(weights_dir, chunk):
    ref = load_ref("tnf60_msa", "msa_encoder")
    opm = OuterProductMean(_w(weights_dir), "blocks.0.outer_product_mean")
    whole = opm(_a(ref, "opm0_m"), _a(ref, "opm0_mask"))
    assert rel_err(opm(_a(ref, "opm0_m"), _a(ref, "opm0_mask"), chunk=chunk), whole) < 1e-6


def test_auto_chunks_respects_the_byte_budget():
    from esmfold2_mlx.msa import auto_chunks

    budget = 512 * 2**20
    for n in (43, 250, 499, 600):
        rows = auto_chunks(n, hidden=32, budget_bytes=budget).opm_rows
        assert 1 <= rows <= n
        assert rows == n or rows * n * 32 * 32 * 4 <= budget
    assert auto_chunks(43, hidden=32).opm_rows == 43  # small inputs are not chunked at all


def test_msa_encoder_survives_a_hard_cap_at_realistic_depth(weights_dir, tmp_path):
    """The encoder at 419 tokens with 1024 rows (the real 2225-row alignment subsampled, and a
    7000-row synthetic one) completes under a 12 GiB process cap."""
    import subprocess
    import sys

    from helpers import REPO, ref_path

    assert ref_path("tnf2_msa", "features").exists() or load_ref("tnf2_msa", "features")
    for rows in (0, 7000):
        proc = subprocess.run([sys.executable, str(REPO / "tests" / "msa_memory_probe.py"), weights_dir,
                               str(ref_path("tnf2_msa", "features")), str(rows), "12"],
                              capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "ok" in proc.stdout
