"""Where one triangle multiplication's time goes (Task 14's ablation, Mol-1-MLX's method): time the
whole op and each part on its own at a given N. Prints a markdown table.

    PYTHONPATH=. python bench/trimul_ablation.py [--n 419]

A hand-written Metal kernel only makes sense if the arithmetic (the contraction's matmul) dominates;
if layout copies (the transposes around it) or elementwise passes dominate, a kernel is the wrong fix.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from esmfold2_mlx.layers import _route  # noqa: E402
from esmfold2_mlx.trunk import PairUpdateBlock  # noqa: E402
from esmfold2_mlx.weights import load_group  # noqa: E402
from preflight import preflight  # noqa: E402


def timed(fn, repeats=7):
    mx.eval(fn())
    out = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        out.append(time.perf_counter() - start)
    return statistics.median(out) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=419)
    args = parser.parse_args()
    preflight()
    n = args.n
    op = PairUpdateBlock(load_group(str(REPO / "weights"), "folding_trunk"), "blocks.0").tri_mul_out
    rng = np.random.default_rng(0)
    z = mx.array((rng.standard_normal((1, n, n, 256)) * 0.5).astype(np.float32))
    mask = mx.ones((1, n, n))
    normed = op.norm_start(z)
    bundled = op.proj_bundle(normed)
    routed = _route(bundled[..., :512], bundled[..., 512:], mask[..., None])
    left, right = routed[..., :256], routed[..., 256:]
    a_contig = mx.contiguous(left.transpose(0, 3, 1, 2))
    b_contig = mx.contiguous(right.transpose(0, 3, 2, 1))
    contracted = op._contract(left, right)
    mx.eval(normed, bundled, routed, a_contig, b_contig, contracted)
    rows = {
        "whole op (with residual)": timed(lambda: op(z, mask, residual=z)),
        "LayerNorm + 1024-wide projection": timed(lambda: op.proj_bundle(op.norm_start(z))),
        "sigmoid routing and mask (compiled)": timed(lambda: _route(bundled[..., :512], bundled[..., 512:], mask[..., None])),
        "contraction as written (transposes + batched matmul + transpose back)": timed(lambda: op._contract(left, right)),
        "contraction matmul alone (contiguous inputs, no transpose back)": timed(lambda: a_contig @ b_contig),
        "LayerNorm + output projection + output gate": timed(lambda: op.proj_emit(op.norm_mix(contracted)) * mx.sigmoid(op.proj_gate(normed))),
    }
    flops = 2 * 256 * n ** 3
    print(f"# Triangle multiplication at N={n}: where the time goes\n")
    print("| part | ms |\n| --- | --- |")
    for k, v in rows.items():
        print(f"| {k} | {v:.1f} |")
    print(f"\nContraction arithmetic: {flops / 1e9:.1f} GFLOP; the matmul alone runs at "
          f"{flops / (rows['contraction matmul alone (contiguous inputs, no transpose back)'] / 1000) / 1e12:.2f} TFLOP/s.")


if __name__ == "__main__":
    main()
