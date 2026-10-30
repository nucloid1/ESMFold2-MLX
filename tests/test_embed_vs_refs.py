"""Inputs embedder (atom encoder with sliding-window 3D-RoPE attention), relative positions, token
bonds and z_init against torch captures (stage ``embed``, CPU and MPS, plus the shared ``features``).

Torch casts the RoPE tables and the atom attention's q, k, v to bf16 on every device (the non-flash
path), so those stages are only defined to bf16 precision. Their gate is torch's own disagreement
with itself: twice the CPU-vs-MPS relative difference of the same output, never below 1e-5. Stages
without those casts keep the 1e-5 gate.
"""

import mlx.core as mx
import numpy as np
import pytest

from helpers import assert_live, load_ref, rel_err

from esmfold2_mlx.atoms import AtomEncoder, SWAAtomBlock, build_rope, swa_attention
from esmfold2_mlx.config import load_config
from esmfold2_mlx.embed import Embedder, RelativePositionEncoding
from esmfold2_mlx.weights import load_group

CASES = ["m49", "vegf1", "tnf60_msa"]
GATE = 1e-5


def _features(case):
    return {k: mx.array(v) for k, v in load_ref(case, "features").items()}


def _bf16_gate(case, key):
    spread = rel_err(load_ref(case, "embed", "mps")[key], load_ref(case, "embed", "cpu")[key])
    return max(GATE, 2.0 * spread)


@pytest.fixture(scope="module")
def embedder(weights_dir):
    return Embedder(weights_dir, load_config(weights_dir))


@pytest.mark.parametrize("case", CASES)
def test_rel_pos_matches(weights_dir, case):
    ref, f = load_ref(case, "embed"), _features(case)
    assert_live(ref["rel_pos"])
    got = RelativePositionEncoding(load_group(weights_dir, "rel_pos"), load_config(weights_dir))(
        f["residue_index"], f["asym_id"], f["sym_id"], f["entity_id"], f["token_index"])
    assert rel_err(got, ref["rel_pos"]) < GATE


@pytest.mark.parametrize("case", CASES)
def test_z_init_from_torch_x_inputs_matches(embedder, case):
    """z_init_1/2 outer sum + rel_pos + token_bonds, fed torch's own x_inputs: no bf16 involved."""
    ref, f = load_ref(case, "embed"), _features(case)
    assert_live(ref["z_init"])
    got = embedder.pair_init(mx.array(ref["x_inputs"]), f)
    assert rel_err(got.z_init, ref["z_init"]) < GATE
    assert rel_err(got.token_bonds_encoding, ref["token_bonds"]) < GATE


@pytest.mark.parametrize("case", CASES)
def test_rope_tables_match_torch_bf16_tables(case):
    ref, f = load_ref(case, "embed"), _features(case)
    cos, sin = build_rope(f["ref_pos"], f["ref_space_uid"], head_dim=32, n_spatial=2, n_uid=10,
                          spatial_base=20.0, uid_base=10000.0)
    for got, want in ((cos, ref["cos"]), (sin, ref["sin"])):
        diff = np.abs(np.array(got) - want)
        assert diff.max() <= 2.0 ** -8 + 1e-9  # at most one bf16 step, where fp32 cos/sin round differently
        assert (diff > 0).mean() < 0.01


@pytest.mark.parametrize("case", CASES)
def test_atom_block_matches(weights_dir, case):
    ref = load_ref(case, "embed")
    assert_live(ref["blk0_out"])
    block = SWAAtomBlock(load_group(weights_dir, "inputs_embedder"), "atom_attention_encoder.atom_transformer.blocks.0",
                         n_heads=4, half_window=64)
    got = block(mx.array(ref["blk0_x"]), mx.array(ref["blk0_c"]), mx.array(ref["cos"]), mx.array(ref["sin"]),
                mx.array(ref["valid"]))
    assert rel_err(got, ref["blk0_out"]) < _bf16_gate(case, "blk0_out")


@pytest.mark.parametrize("case", CASES)
def test_inputs_embedder_matches(embedder, case):
    ref = load_ref(case, "embed")
    got = embedder(_features(case))
    x, want = np.array(got.x_inputs), ref["x_inputs"]
    assert_live(want[..., :384])
    assert rel_err(x[..., :384], want[..., :384]) < _bf16_gate(case, "x_inputs")  # the atom encoding
    np.testing.assert_allclose(x[..., 384:], want[..., 384:], rtol=0, atol=1e-6)  # aatype, profile, deletion mean
    assert rel_err(got.z_init, ref["z_init"]) < _bf16_gate(case, "z_init")


@pytest.mark.parametrize("case", CASES)
def test_inputs_embedder_matches_torch_on_mps_to_1e5(embedder, case):
    """Torch's CPU bf16 attention kernel differs from its MPS one by ~1e-4; MLX's bf16 emulation
    reproduces MPS (the production device) to ~1e-6, so against MPS the ordinary 1e-5 gate applies."""
    mps = load_ref(case, "embed", "mps")
    got = embedder(_features(case))
    assert rel_err(np.array(got.x_inputs)[..., :384], mps["x_inputs"][..., :384]) < GATE
    assert rel_err(got.z_init, mps["z_init"]) < GATE


def test_bf16_spread_is_small_and_recorded():
    """The gate is only meaningful if torch agrees with itself to bf16 precision."""
    for case in CASES:
        assert _bf16_gate(case, "x_inputs") < 1e-2


@pytest.mark.parametrize("n,half,block", [(40, 64, 64), (130, 64, 64), (200, 16, 32), (77, 8, 16)])
def test_banded_attention_matches_dense(n, half, block):
    rng = np.random.default_rng(n)
    q, k, v = (mx.array(rng.standard_normal((2, n, 4, 32)).astype(np.float32)) for _ in range(3))
    valid = np.ones((2, n), bool)
    valid[1, n - 9:] = False  # padded tail in the second batch row
    got = np.array(swa_attention(q, k, v, mx.array(valid), half, block=block, bf16=False))
    want = _dense_reference(np.array(q), np.array(k), np.array(v), valid, half)
    assert rel_err(got, want) < 1e-6


def _dense_reference(q, k, v, valid, half):
    B, N, H, D = q.shape
    rank = np.cumsum(valid, axis=1) - 1
    allowed = (np.abs(rank[:, :, None] - rank[:, None, :]) <= half) & valid[:, None, :] & valid[:, :, None]
    allowed |= np.eye(N, dtype=bool)[None]
    s = np.einsum("bihd,bjhd->bhij", q, k) / np.sqrt(D)
    s = np.where(allowed[:, None], s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhij,bjhd->bihd", p, v) * valid[:, :, None, None]


def test_padded_atoms_do_not_leak_into_real_atoms(weights_dir):
    """Extra masked atoms (count no longer a multiple of the block) leave real tokens' encoding alone."""
    f = load_ref("m49", "features")
    pad = 29
    padded = dict(f)
    for name in ("ref_pos", "ref_charge", "ref_element", "ref_space_uid", "atom_to_token", "ref_atom_name_chars", "atom_attention_mask"):
        a = f[name]
        padded[name] = np.concatenate([a, np.zeros((a.shape[0], pad, *a.shape[2:]), a.dtype)], axis=1)
    cfg = load_config(weights_dir)
    encoder = AtomEncoder(load_group(weights_dir, "inputs_embedder"), "atom_attention_encoder", cfg.atom_encoder,
                          window=cfg.sliding_window, structure_prediction=False)
    plain = np.array(Embedder.atom_inputs(encoder, {k: mx.array(v) for k, v in f.items()}))
    with_pad = np.array(Embedder.atom_inputs(encoder, {k: mx.array(v) for k, v in padded.items()}))
    assert_live(plain)
    # bf16 attention may round a handful of elements differently when the block partition shifts;
    # a leak from padding would be O(1).
    assert rel_err(with_pad, plain) < 1e-3


def test_embedder_consumes_every_weight(embedder):
    assert embedder.groups and all(w.unused() == [] for w in embedder.groups.values())
