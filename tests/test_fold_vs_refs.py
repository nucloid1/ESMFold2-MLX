"""The whole post-ESMC fold (MLXFold) against torch (stage ``fold``).

CPU: the real forward (synthetic ESMC states, checkpoint loops and steps, 2 samples) with every random
draw recorded. MPS: the same forward with the CPU draws replayed, so CPU, MPS and MLX consume identical
randomness. Blocking here (plan Task 10): output keys, distogram within 1e-3 relative with under 0.5%
of bins flipped, coordinates inside torch's own CPU/MPS spread, and seed behaviour. Confidences are
tabulated against both devices by scripts/parity_report.py into docs/parity/end_to_end.md.
"""

import mlx.core as mx
import numpy as np
import pytest

from helpers import assert_live, kabsch_rmsd, load_ref, rel_err, synthetic_lm_states

from esmfold2_mlx.config import FoldOptions
from esmfold2_mlx.fold import MLXFold
from esmfold2_mlx.rng import MLXDraws, ReplayDraws

CASES = ["m49", "vegf1", "tnf60_msa", "tnf60_shallow"]
_RESULTS: dict = {}


@pytest.fixture(scope="module")
def fold(weights_dir):
    return MLXFold(weights_dir)


def replayed(fold, case):
    """MLX fold of a case with the CPU run's recorded draws (cached across tests)."""
    if case not in _RESULTS:
        f = load_ref(case, "features")
        lm = synthetic_lm_states(f["res_type"].shape[1])
        _RESULTS[case] = fold.fold(f, lm, FoldOptions(num_diffusion_samples=2), draws=ReplayDraws(load_ref(case, "fold_draws")))
    return _RESULTS[case]


def test_fold_output_keys_equal_torch(fold):
    assert set(replayed(fold, "m49")) == set(load_ref("m49", "fold"))


@pytest.mark.parametrize("case", CASES)
def test_distogram_matches_torch(fold, case):
    got, cpu, mps = replayed(fold, case)["distogram_logits"], load_ref(case, "fold")["distogram_logits"], load_ref(case, "fold", "mps")["distogram_logits"]
    assert_live(cpu)
    assert rel_err(got, cpu) < 1e-3
    assert rel_err(got, mps) < 1e-3
    assert (got.argmax(-1) != cpu.argmax(-1)).mean() < 0.005


@pytest.mark.parametrize("case", CASES)
def test_coordinates_within_torch_device_spread(fold, case):
    got = replayed(fold, case)["sample_atom_coords"]
    cpu, mps = load_ref(case, "fold")["sample_atom_coords"], load_ref(case, "fold", "mps")["sample_atom_coords"]
    mask = load_ref(case, "features")["atom_attention_mask"][0].astype(bool)
    for s in range(got.shape[0]):
        spread = kabsch_rmsd(mps[s], cpu[s], mask)
        assert kabsch_rmsd(got[s], mps[s], mask) <= max(0.05, 2 * spread), s
        assert kabsch_rmsd(got[s], cpu[s], mask) <= max(0.5, 2 * spread), s


def _seeded(fold, seed):
    f = load_ref("m49", "features")
    lm = synthetic_lm_states(f["res_type"].shape[1])
    return fold.fold(f, lm, FoldOptions(num_diffusion_samples=1, seed=seed), draws=MLXDraws(seed))["sample_atom_coords"]


def test_same_seed_twice_identical(fold):
    assert rel_err(_seeded(fold, 7), _seeded(fold, 7)) < 1e-6


def test_different_seeds_differ(fold):
    mask = np.ones(load_ref("m49", "features")["atom_attention_mask"].shape[1], bool)
    assert kabsch_rmsd(_seeded(fold, 7)[0], _seeded(fold, 8)[0], mask) > 0.01
