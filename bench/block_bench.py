"""Time one PairUpdateBlock in MLX (fp32, real block weights, random pair input). Prints one JSON line.

    PYTHONPATH=. python bench/block_bench.py --n 250 [--block 0] [--warmups 2] [--repeats 7]

Run in a clean process (no torch): MLX runs slower in a process where torch has done CPU-side work
(Mol-1-MLX finding). Timing is value-independent, so a random input is valid for speed.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from esmfold2_mlx.trunk import PairUpdateBlock  # noqa: E402
from esmfold2_mlx.weights import load_group  # noqa: E402
from preflight import preflight  # noqa: E402


def _time(fn, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        mx.eval(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - start)
    return times


def _summary(times: list[float]) -> dict:
    return {"times": [round(t, 5) for t in times], "median": statistics.median(times), "min": min(times), "max": max(times)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--weights", default=str(REPO / "weights"))
    args = parser.parse_args()
    preflight()

    block = PairUpdateBlock(load_group(args.weights, "folding_trunk"), f"blocks.{args.block}")
    rng = np.random.default_rng(0)
    z = mx.array((rng.standard_normal((1, args.n, args.n, 256)) * 0.5).astype(np.float32))
    mask = mx.ones((1, args.n, args.n))
    mx.eval(z, mask)

    result = {"impl": "mlx", "n": args.n, "block": args.block, **_summary(_time(lambda: block(z, mask), args.warmups, args.repeats))}
    # Per-op medians (each op run alone on the same input) to see where the block's time goes.
    result["ops"] = {
        "tri_mul_out": statistics.median(_time(lambda: block.tri_mul_out(z, mask, residual=z), args.warmups, args.repeats)),
        "tri_mul_in": statistics.median(_time(lambda: block.tri_mul_in(z, mask, residual=z), args.warmups, args.repeats)),
        "pair_transition": statistics.median(_time(lambda: block.pair_transition(z), args.warmups, args.repeats)),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
