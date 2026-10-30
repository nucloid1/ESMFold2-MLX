"""Where the MLX fold's time goes (Task 14 step 1). Writes docs/profile.md.

    PYTHONPATH=. python bench/profile_fold.py [case ...]     (default: vegf2 tnf2_nomsa)

Phase times come from one warm MLXFold run instrumented phase by phase (synchronised); op-class times
come from timing one instance of each op at the case's size (median of 5, synchronised) multiplied by
how many times a fold runs it. Clean MLX process, no torch.
"""

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
from esmfold2_mlx.schedule import noise_schedule  # noqa: E402
from helpers import ref_path, synthetic_lm_states  # noqa: E402
from preflight import preflight  # noqa: E402

OUT = REPO / "docs" / "profile.md"


def timed(fn, repeats=5, warmups=1):
    for _ in range(warmups):
        mx.eval(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def phases(fold: MLXFold, f: dict, lm) -> dict:
    """One fold, timed per phase by wrapping the components."""
    spent = {}

    def wrap(name, obj, attr):
        original = getattr(obj, attr)

        def run(*a, **kw):
            mx.synchronize()
            start = time.perf_counter()
            out = original(*a, **kw)
            mx.eval(out if isinstance(out, mx.array) else [v for v in (out.values() if isinstance(out, dict) else [out.x_inputs]) if isinstance(v, mx.array)])
            spent[name] = spent.get(name, 0.0) + time.perf_counter() - start
            return out
        setattr(obj, attr, run)
        return original

    saved = [(fold, "embedder", wrap("embed", fold, "embedder")), (fold, "trunk", wrap("trunk (LM shim, MSA encoder, 21 loops, coda)", fold, "trunk")),
             (fold.sampler, "sample", wrap("diffusion sampler", fold.sampler, "sample")),
             (fold, "confidence", wrap("confidence head", fold, "confidence"))]
    start = time.perf_counter()
    fold.fold(f, lm, FoldOptions(num_diffusion_samples=1))
    total = time.perf_counter() - start
    for obj, attr, original in saved:
        setattr(obj, attr, original)
    spent["other (distogram, conversion)"] = total - sum(spent.values())
    return {"total": total, **spent}


def op_classes(fold: MLXFold, n: int, atoms: int, calls: int) -> dict:
    rng = np.random.default_rng(0)
    z = mx.array((rng.standard_normal((1, n, n, 256)) * 0.5).astype(np.float32))
    mask = mx.ones((1, n, n))
    block = fold.trunk.trunk.blocks[0]
    pair_blocks = 21 * 48 + 2 + 21 * 4  # trunk + coda + lm_encoder; the MSA encoder adds 4 more (once per fold at depth <= 1024)
    out = {
        "triangle multiplication (x2 per pair block)": (timed(lambda: block.tri_mul_out(z, mask, residual=z)) * 2, pair_blocks + 4),
        "pair transition": (timed(lambda: block.pair_transition(z)), pair_blocks),
    }
    sm = fold.sampler.module
    tt = sm.transformer
    a = mx.array(rng.standard_normal((1, n, 768)).astype(np.float32))
    s = mx.array(rng.standard_normal((1, n, 768)).astype(np.float32))
    bias = tt.attn[0].pair_bias(z[..., :256]) if z.shape[-1] == 256 else None
    out["token transformer attention (12 per denoiser call)"] = (timed(lambda: tt.attn[0](a, s, bias, None)), 12 * calls)
    out["token transformer transition (12 per denoiser call)"] = (timed(lambda: tt.trans[0](a, s)), 12 * calls)
    q = mx.array(rng.standard_normal((1, atoms, 128)).astype(np.float32))
    cos = mx.ones((1, atoms, 16))
    sin = mx.zeros((1, atoms, 16))
    valid = mx.ones((1, atoms), dtype=mx.bool_)
    ab = sm.atom_encoder.blocks[0]
    out["atom attention block (6 per denoiser call + 3 embedder)"] = (timed(lambda: ab(q, q, cos, sin, valid)), 6 * calls + 3)
    return {k: (t, c, t * c) for k, (t, c) in out.items()}


def main(cases) -> None:
    preflight()
    fold = MLXFold(str(REPO / "weights"))
    calls = len(noise_schedule(fold.cfg, None, 256.0)) - 1
    lines = ["# Where the MLX fold's time goes", "", "One sample, the checkpoint's 20 loops and steps (" + str(calls) + " denoiser calls), synthetic ESMC states. "
             "Phases: one warm fold, synchronised per phase. Op classes: one instance timed at the case's size (median of 5) times "
             "the number of instances a fold runs. Written by bench/profile_fold.py.", ""]
    for case in cases:
        f = dict(np.load(ref_path(case, "features")))
        n, atoms = int(f["res_type"].shape[1]), int(f["atom_attention_mask"].shape[1])
        lm = mx.array(synthetic_lm_states(n)).astype(mx.bfloat16)
        fold.fold(f, lm, FoldOptions())  # warm
        ph = phases(fold, f, lm)
        lines += [f"## {case} ({n} tokens, {atoms} atoms)", "", "| phase | seconds | share |", "| --- | --- | --- |"]
        lines += [f"| {k} | {v:.2f} | {100 * v / ph['total']:.0f}% |" for k, v in ph.items() if k != "total"]
        lines += [f"| total | {ph['total']:.2f} | 100% |", "", "| op class | one instance (ms) | instances per fold | seconds per fold | share of fold |",
                  "| --- | --- | --- | --- | --- |"]
        for k, (t, c, tot) in op_classes(fold, n, atoms, calls).items():
            lines.append(f"| {k} | {t * 1000:.1f} | {c} | {tot:.2f} | {100 * tot / ph['total']:.0f}% |")
        lines.append("")
        print(f"{case}: total {ph['total']:.1f} s", flush=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["vegf2", "tnf2_nomsa"])
