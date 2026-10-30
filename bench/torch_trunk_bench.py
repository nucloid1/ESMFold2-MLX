"""Time the torch trunk on MPS inside the real forward, called as the rescore worker calls it
(trunk-only copy, eval, no grad, default chunk size, the checkpoint's 20 loops). Prints one JSON line.

    PYTHONPATH=. python bench/torch_trunk_bench.py --case vegf2 [--warmups 2] [--repeats 5]

The trunk time is the LM shim + ``_run_one_loop`` (MSA encoder, lm_encoder, 21 loop steps) +
readout + coda, each segment bracketed by ``torch.mps.synchronize()``; the forward runs one sampler
step and one sample so the rest is short. Peak is the sampled MPS driver allocation over all runs.
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


class Segments:
    """Accumulate synchronised wall time between a start hook and an end hook."""

    def __init__(self) -> None:
        self.total = 0.0
        self._t0 = None

    def start(self, *_):
        torch.mps.synchronize()
        self._t0 = time.perf_counter()

    def stop(self, *_):
        torch.mps.synchronize()
        self.total += time.perf_counter() - self._t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    preflight()
    from esm.models.esmfold2.processor import _seed_context

    model = load_trunk_only("mps")
    features, _ = featurize(CASES[args.case], 0, "mps")
    n = features["res_type"].shape[1]
    lm = torch.from_numpy(synthetic_lm_states(n)).to("mps", torch.bfloat16)  # ESMC hands over bf16
    seg = Segments()
    model.language_model.register_forward_pre_hook(seg.start)
    model.language_model.register_forward_hook(seg.stop)
    model.parcae_readout.register_forward_pre_hook(seg.start)
    model.parcae_coda.register_forward_hook(seg.stop)
    original_loop = model._run_one_loop

    def run_one_loop(**kw):
        seg.start()
        out = original_loop(**kw)
        seg.stop()
        return out

    model._run_one_loop = run_one_loop
    peak = [0.0]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.25):
            peak[0] = max(peak[0], torch.mps.driver_allocated_memory() / 1e9)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    times = []
    for run in range(args.warmups + args.repeats):
        seg.total = 0.0
        with torch.no_grad(), _seed_context(run):
            model(**features, lm_hidden_states=lm, num_sampling_steps=1, num_diffusion_samples=1, **SAMPLER)
        torch.mps.synchronize()
        times.append(seg.total)
        torch.mps.empty_cache()
    stop.set()
    sampler.join()
    warm = times[args.warmups:] or times
    print(json.dumps({"impl": "torch-mps", "case": args.case, "n": n, "msa_depth": int(features["msa"].shape[1]),
                      "first": times[0], "times": [round(t, 3) for t in warm], "median": statistics.median(warm),
                      "min": min(warm), "max": max(warm), "peak_gb": round(peak[0], 2)}))


if __name__ == "__main__":
    main()
