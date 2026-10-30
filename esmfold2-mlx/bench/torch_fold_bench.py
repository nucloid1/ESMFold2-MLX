"""Time the whole torch forward on MPS (everything after ESMC), called as the rescore worker calls it:
trunk-only copy, eval, no grad, default chunk size, checkpoint loops and steps. Prints one JSON line.

    PYTHONPATH=. python bench/torch_fold_bench.py --case vegf2 --samples 1 [--warmups 2] [--repeats 5]
"""

import torch  # first

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "tests")]

from cases import CASES  # noqa: E402
from helpers import synthetic_lm_states  # noqa: E402
from preflight import preflight  # noqa: E402
from torch_model import SAMPLER, featurize, load_trunk_only  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    preflight()
    from esm.models.esmfold2.processor import _seed_context

    model = load_trunk_only("mps")
    features, _ = featurize(CASES[args.case], 0, "mps")
    n = features["res_type"].shape[1]
    lm = torch.from_numpy(synthetic_lm_states(n)).to("mps", torch.bfloat16)
    peak, stop = [0.0], threading.Event()

    def sample_memory():
        while not stop.wait(0.25):
            peak[0] = max(peak[0], torch.mps.driver_allocated_memory() / 1e9)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    times = []
    for run in range(args.warmups + args.repeats):
        torch.mps.synchronize()
        start = time.perf_counter()
        with torch.no_grad(), _seed_context(run):
            out = model(**features, lm_hidden_states=lm, num_diffusion_samples=args.samples, **SAMPLER)
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
        del out
        torch.mps.empty_cache()
    stop.set()
    sampler.join()
    warm = times[args.warmups:] or times
    print(json.dumps({"impl": "torch-mps", "case": args.case, "n": n, "samples": args.samples, "first": times[0],
                      "times": [round(t, 3) for t in warm], "median": statistics.median(warm), "min": min(warm),
                      "max": max(warm), "peak_gb": round(peak[0], 2)}))


if __name__ == "__main__":
    main()
