"""The randomness interface: recorded torch draws replayed in order, and MLX's own seeded draws."""

import mlx.core as mx
import numpy as np
import pytest

from esmfold2_mlx.rng import SITES, MLXDraws, ReplayDraws


def test_sites_name_the_torch_call_sites():
    assert SITES["pair_init"] == "_init_pair_state:trunc_normal_"
    assert SITES["lm_dropout_keep"] == "_run_one_loop:dropout"
    assert set(SITES) == {"pair_init", "lm_dropout_keep", "msa_column_keep", "msa_row_indices",
                          "diffusion_init", "rotation_quat", "translation", "step_noise"}


def test_replay_returns_recorded_tensors_in_order():
    keep0 = np.array([[True, False]])
    keep1 = np.array([[False, True]])
    draws = ReplayDraws({"_run_one_loop:dropout/0": keep0, "_run_one_loop:dropout/1": keep1})
    np.testing.assert_array_equal(np.array(draws.lm_dropout_keep(0, (1, 2), 0.25)), keep0)
    np.testing.assert_array_equal(np.array(draws.lm_dropout_keep(1, (1, 2), 0.25)), keep1)


def test_replay_draws_exhausted_raises_naming_the_site():
    draws = ReplayDraws({"_init_pair_state:trunc_normal_/0": np.zeros((1, 2, 2, 3), np.float32)})
    draws.pair_init((1, 2, 2, 3), 0.1)
    with pytest.raises(IndexError, match="_init_pair_state:trunc_normal_"):
        draws.pair_init((1, 2, 2, 3), 0.1)


def test_replay_rejects_a_shape_that_does_not_match_the_recording():
    draws = ReplayDraws({"_init_pair_state:trunc_normal_/0": np.zeros((1, 2, 2, 3), np.float32)})
    with pytest.raises(ValueError, match="shape"):
        draws.pair_init((1, 3, 3, 3), 0.1)


def test_mlx_draws_same_seed_identical_and_different_seed_differs():
    a, b, c = MLXDraws(7), MLXDraws(7), MLXDraws(8)
    shape = (1, 8, 8, 16)
    pa, pb, pc = (np.array(d.pair_init(shape, 0.05)) for d in (a, b, c))
    np.testing.assert_array_equal(pa, pb)
    assert not np.array_equal(pa, pc)
    np.testing.assert_array_equal(np.array(a.lm_dropout_keep(0, shape, 0.25)), np.array(b.lm_dropout_keep(0, shape, 0.25)))


def test_mlx_pair_init_is_truncated_at_three_std():
    std = 0.05
    x = np.array(MLXDraws(0).pair_init((1, 64, 64, 64), std))
    assert np.abs(x).max() <= 3 * std + 1e-7
    assert abs(x.std() - std * 0.986) < 0.002  # a normal truncated at 3 std has std 0.986 * std


def test_mlx_dropout_keep_rate_is_one_minus_p():
    keep = np.array(MLXDraws(0).lm_dropout_keep(0, (1, 128, 128, 32), 0.25))
    assert keep.dtype == np.bool_
    assert abs(keep.mean() - 0.75) < 0.01


def test_mlx_draws_cover_every_site_with_the_requested_shapes():
    d = MLXDraws(0)
    assert d.diffusion_init((2, 5, 3)).shape == (2, 5, 3)
    assert d.rotation_quat(0, 2).shape == (2, 4)
    assert d.translation(0, (2, 1, 3)).shape == (2, 1, 3)
    assert d.step_noise(0, (2, 5, 3)).shape == (2, 5, 3)
    assert d.msa_column_keep((1, 7), 0.1).shape == (1, 7)
    rows = np.array(d.msa_row_indices(0, 10, 4))
    assert rows[0] == 0 and len(rows) == 4 and (np.diff(rows) > 0).all()
