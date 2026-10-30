"""``install``: an esm EsmFold2Model whose post-ESMC forward runs in the MLX worker (integration only:
it loads the torch model in this process)."""

import os

import numpy as np
import pytest

from helpers import load_ref, synthetic_lm_states

pytestmark = pytest.mark.skipif(os.environ.get("ESMFOLD2_MLX_INTEGRATION") != "1",
                                reason="loads the torch model: set ESMFOLD2_MLX_INTEGRATION=1")


def test_install_returns_the_torch_forward_keys(weights_dir):
    import sys

    import torch

    from helpers import REPO

    sys.path.insert(0, str(REPO / "scripts"))
    from cases import CASES
    from torch_model import SAMPLER, featurize, load_trunk_only

    from esmfold2_mlx.hybrid import install

    model = load_trunk_only("cpu")
    features, _ = featurize(CASES["m49"], 0, "cpu")
    lm = torch.from_numpy(synthetic_lm_states(43))
    with torch.no_grad():
        want = model(**features, lm_hidden_states=lm, num_loops=1, num_sampling_steps=2, num_diffusion_samples=1, **SAMPLER)
    remote = install(model, weights_dir)
    try:
        got = model(**features, lm_hidden_states=lm, num_loops=1, num_sampling_steps=2, num_diffusion_samples=1, **SAMPLER)
    finally:
        remote.close()
    assert set(got) == set(want)
    assert all(isinstance(v, torch.Tensor) for v in got.values())
    assert np.isfinite(got["sample_atom_coords"].numpy()).all()
    assert got["distogram_logits"].shape == want["distogram_logits"].shape
