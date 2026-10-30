"""Time one PairUpdateBlock in torch on MPS, called as production calls it. Prints one JSON line.

    PYTHONPATH=. python bench/torch_block_bench.py --n 250 --chunk {64,none} [--block 0]

The model is the trunk-only copy loaded by ``scripts/torch_model.load_trunk_only`` (a copy of the
rescore worker's loader), eval, no grad. ``--chunk 64`` leaves the checkpoint's default chunk size
untouched (the worker never calls ``set_chunk_size``, so this is production); ``--chunk none`` calls
``model.set_chunk_size(None)``, which the model's docstring calls faster below 600 tokens.
"""

import torch  # first

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from preflight import preflight  # noqa: E402
from torch_model import load_trunk_only  # noqa: E402


def _time(fn, warmups: int, repeats: int) -> list[float]:
    with torch.no_grad():
        for _ in range(warmups):
            fn()
            torch.mps.synchronize()
        times = []
        for _ in range(repeats):
            torch.mps.synchronize()
            start = time.perf_counter()
            fn()
            torch.mps.synchronize()
            times.append(time.perf_counter() - start)
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--chunk", choices=("64", "none"), required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    preflight()

    model = load_trunk_only("mps")
    if args.chunk == "none":
        model.set_chunk_size(None)
    block = model.folding_trunk.blocks[args.block]
    generator = torch.Generator().manual_seed(0)
    z = (torch.randn(1, args.n, args.n, 256, generator=generator) * 0.5).to("mps")
    mask = torch.ones(1, args.n, args.n, device="mps")

    times = _time(lambda: block(z, pair_attention_mask=mask), args.warmups, args.repeats)
    ops = {
        "tri_mul_out": statistics.median(_time(lambda: block.tri_mul_out(z, mask=mask), args.warmups, args.repeats)),
        "tri_mul_in": statistics.median(_time(lambda: block.tri_mul_in(z, mask=mask), args.warmups, args.repeats)),
        "pair_transition": statistics.median(_time(lambda: block.pair_transition(z), args.warmups, args.repeats)),
    }
    print(json.dumps({
        "impl": "torch-mps", "chunk": args.chunk, "n": args.n, "block": args.block,
        "times": [round(t, 5) for t in times], "median": statistics.median(times), "min": min(times), "max": max(times), "ops": ops,
    }))


if __name__ == "__main__":
    main()
