"""The looped pair trunk (language-model shim, lm_encoder, Parcae update, 48 blocks, readout, coda)
against torch-CPU captures.

References: scripts/capture_refs.py stage ``trunk`` runs the real forward on CPU with synthetic ESMC
hidden states (tests/helpers.synthetic_lm_states), the checkpoint's 20 loops (21 steps), recording
the random draws (pair init and the per-loop LM-pair dropout) so the MLX loop can replay them.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from helpers import REPO, assert_live, load_ref, rel_err, synthetic_lm_states

from esmfold2_mlx.config import load_config
from esmfold2_mlx.rng import ReplayDraws
from esmfold2_mlx.trunk import FoldingTrunk, LanguageModelShim, ParcaeTrunk
from esmfold2_mlx.weights import load_group

CASES = ["m49", "vegf1", "m49_q", "tnf60_msa", "tnf60_shallow"]
STAGE_GATE = 1e-5
COMPOSITE_GATE = 1e-3


@pytest.fixture(scope="module")
def trunk(weights_dir):
    return ParcaeTrunk(weights_dir, load_config(weights_dir))


def _lm_states(case):
    n = load_ref(case, "trunk_inputs")["z_init"].shape[1]
    return mx.array(synthetic_lm_states(n))


@pytest.mark.parametrize("case", CASES)
def test_lm_shim_matches(weights_dir, case):
    want = load_ref(case, "lm_shim")["lm_z"]
    assert_live(want)
    shim = LanguageModelShim(load_group(weights_dir, "language_model"))
    assert rel_err(shim(_lm_states(case)), want) < STAGE_GATE


@pytest.mark.parametrize("case", CASES)
def test_lm_encoder_matches(weights_dir, case):
    ref = load_ref(case, "loop_states")
    assert_live(ref["lm_enc0_out"])
    encoder = FoldingTrunk(load_group(weights_dir, "lm_encoder"), "", n_layers=4)
    got = encoder(mx.array(ref["lm_enc0_in"]), mx.array(load_ref(case, "trunk_inputs")["pair_mask"]))
    assert rel_err(got, ref["lm_enc0_out"]) < STAGE_GATE


def _msa_inputs(inputs):
    return {k: mx.array(inputs[k]) for k in ("msa", "msa_attention_mask", "has_deletion", "deletion_value")}


@pytest.mark.parametrize("case", CASES)
def test_single_loop_update_matches(trunk, case):
    """One step from the captured state, fed the captured MSA-encoder output (its own parity is in test_msa_vs_refs)."""
    inputs, ref = load_ref(case, "trunk_inputs"), load_ref(case, "loop_states")
    msa_pair = load_ref(case, "msa_encoder")["out"]
    assert_live(ref["loop0_out"])
    got = trunk.step(
        mx.array(inputs["z"]), mx.array(inputs["z_init"]), mx.array(ref["lm_enc0_in"]), mx.array(inputs["pair_mask"]),
        msa_pair=mx.array(msa_pair),
    )
    assert rel_err(got, ref["loop0_out"]) < STAGE_GATE


@pytest.mark.parametrize("case", CASES)
def test_loop_states_and_trunk_out_track_refs(trunk, case):
    inputs, ref, out = load_ref(case, "trunk_inputs"), load_ref(case, "loop_states"), load_ref(case, "trunk_out")
    draws = ReplayDraws(load_ref(case, "trunk_draws"))
    seen = {}
    got = trunk(
        mx.array(inputs["x_inputs"]), mx.array(inputs["z_init"]), _lm_states(case), mx.array(inputs["pair_mask"]),
        num_loops=int(inputs["total_steps"]) - 1, draws=draws, msa_inputs=_msa_inputs(inputs),
        on_loop=lambda k, z: seen.__setitem__(k, z),
    )
    loops = sorted(int(k[len("loop"):-len("_out")]) for k in ref if k.startswith("loop") and k.endswith("_out"))
    errors = {k: rel_err(seen[k], ref[f"loop{k}_out"]) for k in loops}
    final = rel_err(got, out["z"])
    lines = [f"| {case} | {k} | {e:.2e} |" for k, e in errors.items()] + [f"| {case} | after readout + coda | {final:.2e} |"]
    _append_growth_table(case, lines)
    assert_live(out["z"])
    assert all(e < COMPOSITE_GATE for e in errors.values()), errors
    assert final < COMPOSITE_GATE


def test_trunk_consumes_every_weight(trunk):
    assert trunk.groups and all(w.unused() == [] for w in trunk.groups.values())


def _append_growth_table(case: str, lines: list[str]) -> None:
    path = Path(REPO) / "docs" / "parity" / "trunk_growth.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Trunk loop: MLX vs torch-CPU, relative error per loop\n\n"
        "Synthetic ESMC states, replayed torch draws, 21 loop steps (the checkpoint's 20 loops). Written by "
        "tests/test_trunk_loop_vs_refs.py.\n\n| case | loop | relative error |\n| --- | --- | --- |\n"
    )
    old = path.read_text() if path.exists() else header
    kept = [line for line in old.splitlines() if not line.startswith(f"| {case} |")]
    path.write_text("\n".join(kept + lines) + "\n")
