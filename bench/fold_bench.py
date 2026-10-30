"""Time the whole MLX fold (MLXFold.fold: embed, trunk, distogram, sampler, confidence) on a case's real
features with synthetic ESMC states. Prints one JSON line. Clean MLX process, no torch.

    PYTHONPATH=. python bench/fold_bench.py --case vegf2 --samples 1 [--warmups 2] [--repeats 5] [--cap-gib G]
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

from esmfold2_mlx.config import FoldOptions  # noqa: E402
from esmfold2_mlx.fold import MLXFold  # noqa: E402
from esmfold2_mlx.guard import memory_guard  # noqa: E402
from helpers import ref_path, synthetic_lm_states  # noqa: E402
from preflight import preflight  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cap-gib", type=float, default=None)
    parser.add_argument("--weights", default=str(REPO / "weights"))
    args = parser.parse_args()
    preflight()
    f = dict(np.load(ref_path(args.case, "features")))
    n = int(f["res_type"].shape[1])
    guard = memory_guard(max_gib=args.cap_gib) if args.cap_gib else contextlib.nullcontext({"mlx": None})
    with guard as peak:
        fold = MLXFold(args.weights)
        lm = mx.array(synthetic_lm_states(n)).astype(mx.bfloat16)  # ESMC hands over bf16
        times = []
        for run in range(args.warmups + args.repeats):
            start = time.perf_counter()
            fold.fold(f, lm, FoldOptions(num_diffusion_samples=args.samples, seed=run))
            times.append(time.perf_counter() - start)
    warm = times[args.warmups:] or times
    print(json.dumps({"impl": "mlx", "case": args.case, "n": n, "samples": args.samples, "first": times[0],
                      "times": [round(t, 3) for t in warm], "median": statistics.median(warm), "min": min(warm),
                      "max": max(warm), "mlx_peak_gib": peak.get("mlx")}))


if __name__ == "__main__":
    main()
