"""Run the MSA encoder once on a real featurized complex under a hard memory cap (child process of
tests/test_msa_vs_refs.py). Usage: msa_memory_probe.py WEIGHTS FEATURES_NPZ ROWS CAP_GIB

ROWS 0 uses the alignment as featurized; ROWS > 0 tiles its rows up to that depth first. Either way
the encoder sees ``max_depth`` subsampled rows, exactly as a fold would.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from esmfold2_mlx.config import load_config  # noqa: E402
from esmfold2_mlx.guard import memory_guard  # noqa: E402
from esmfold2_mlx.msa import MSAEncoder, auto_chunks  # noqa: E402
from esmfold2_mlx.rng import MLXDraws  # noqa: E402


def main(weights: str, features: str, rows: int, cap: float) -> None:
    f = np.load(features)
    msa = {k: f[k] for k in ("msa", "msa_attention_mask", "has_deletion", "deletion_value")}
    if rows:
        reps = -(-rows // msa["msa"].shape[1])
        msa = {k: np.concatenate([v[:, :1]] + [v[:, 1:]] * reps, axis=1)[:, :rows] for k, v in msa.items()}
    cfg = load_config(weights)
    with memory_guard(max_gib=cap) as peak:
        encoder = MSAEncoder(weights, cfg)
        depth, n = msa["msa"].shape[1], msa["msa"].shape[2]
        picked = MLXDraws(0).msa_row_indices(0, depth, cfg.msa_encoder.max_depth) if depth > cfg.msa_encoder.max_depth else mx.arange(depth)
        sub = {k: mx.array(v)[:, picked] for k, v in msa.items()}
        rng = np.random.default_rng(0)
        x_inputs = mx.array(rng.standard_normal((1, n, cfg.d_single_inputs), dtype=np.float32))
        z_init = mx.array(rng.standard_normal((1, n, n, cfg.d_pair), dtype=np.float32))
        out = encoder(z_init, x_inputs, *MSAEncoder.prepare(**sub), opm_chunk=auto_chunks(n, cfg.msa_encoder.outer_hidden_size).opm_rows)
        mx.eval(out)
        assert np.isfinite(np.array(out)).all()
    print(f"ok depth={depth} used={sub['msa'].shape[1]} n={n} mlx_peak_gib={peak['mlx']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
