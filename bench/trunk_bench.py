"""Time the MLX trunk (LM shim, MSA encoder, 21 loop steps of 48 blocks, readout, coda) on a case's
real features with synthetic ESMC states. Prints one JSON line. Clean MLX process, no torch.

    PYTHONPATH=. python bench/trunk_bench.py --case vegf2 [--warmups 2] [--repeats 5] [--cap-gib G]

Timing is value-independent, so synthetic ESMC states are valid for speed (Mol-1-MLX measured this).
With --cap-gib the run happens under the process memory guard (hard-cap survival is the memory test).
"""

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "tests")]

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from esmfold2_mlx.config import load_config  # noqa: E402
from esmfold2_mlx.embed import Embedder  # noqa: E402
from esmfold2_mlx.guard import memory_guard  # noqa: E402
from esmfold2_mlx.rng import MLXDraws  # noqa: E402
from esmfold2_mlx.trunk import ParcaeTrunk  # noqa: E402
from helpers import ref_path, synthetic_lm_states  # noqa: E402
from preflight import preflight  # noqa: E402

MSA_KEYS = ("msa", "msa_attention_mask", "has_deletion", "deletion_value")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cap-gib", type=float, default=None)
    parser.add_argument("--weights", default=str(REPO / "weights"))
    args = parser.parse_args()
    preflight()

    f = {k: mx.array(v) for k, v in np.load(ref_path(args.case, "features")).items()}
    cfg = load_config(args.weights)
    guard = memory_guard(max_gib=args.cap_gib) if args.cap_gib else contextlib.nullcontext({"mlx": None})
    with guard as peak:
        emb = Embedder(args.weights, cfg)(f)
        trunk = ParcaeTrunk(args.weights, cfg)
        n = int(f["res_type"].shape[1])
        lm = mx.array(synthetic_lm_states(n)).astype(mx.bfloat16)  # ESMC hands over bf16
        msa = {k: f[k] for k in MSA_KEYS}
        mx.eval(emb.x_inputs, emb.z_init, lm)

        def run():
            return trunk(emb.x_inputs, emb.z_init, lm, emb.pair_mask, num_loops=cfg.num_loops, draws=MLXDraws(0), msa_inputs=msa)

        times = []
        for _ in range(args.warmups + args.repeats):
            start = time.perf_counter()
            mx.eval(run())
            times.append(time.perf_counter() - start)
    warm = times[args.warmups:] or times
    print(json.dumps({"impl": "mlx", "case": args.case, "n": n, "msa_depth": int(f["msa"].shape[1]),
                      "first": times[0], "times": [round(t, 3) for t in warm], "median": statistics.median(warm),
                      "min": min(warm), "max": max(warm), "mlx_peak_gib": peak.get("mlx")}))


if __name__ == "__main__":
    main()
