"""Checkpoint 2: the whole fold (everything after ESMC), MLX vs torch/MPS. Writes docs/checkpoints/ckpt2-fold.md.

    PYTHONPATH=. python bench/ckpt2.py

Cases at 1 and 3 diffusion samples; per configuration, one process each, one after another under this
driver's gpu lock: torch timing (2 warmups, 5 timed), MLX timing (same), then MLX once more under a hard
memory cap equal to torch's peak. PASS only if MLX's median is at least 10 percent below torch's for
every configuration and every capped run survived.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "tests")]

from helpers import ref_path  # noqa: E402
from preflight import preflight  # noqa: E402

CASES = (("vegf2", "none"), ("il17", "none"), ("tnf2_nomsa", "none"), ("tnf2_msa", "real, 2225 rows"), ("il17_scfv", "none"))
SAMPLES = (1, 3)
REPS = ("--warmups", "1", "--repeats", "3")  # Checkpoint 1 runs varied by <1%; see the ledger ruling
OUT = REPO / "docs" / "checkpoints" / "ckpt2-fold.md"
GIB = 2**30


def run(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, *cmd], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)}, cwd=REPO)
    if check and proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{proc.stderr[-3000:]}")
    return proc


def last_json(proc) -> dict:
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    preflight()
    rows = []
    for case, msa in CASES:
        if not ref_path(case, "features").exists():
            run("scripts/capture_refs.py", case, "--stages", "features")
        for samples in SAMPLES:
            s = str(samples)
            t = last_json(run("bench/torch_fold_bench.py", "--case", case, "--samples", s, *REPS))
            m = last_json(run("bench/fold_bench.py", "--case", case, "--samples", s, *REPS))
            cap = t["peak_gb"] * 1e9 / GIB
            capped = run("bench/fold_bench.py", "--case", case, "--samples", s, "--warmups", "0", "--repeats", "1",
                         "--cap-gib", f"{cap:.2f}", check=False)
            survived = capped.returncode == 0
            rows.append((case, msa, samples, t, m, cap, survived, last_json(capped).get("mlx_peak_gib") if survived else None))
            print(f"{case} x{samples}: torch {t['median']:.1f}s  mlx {m['median']:.1f}s  x{t['median'] / m['median']:.2f}  "
                  f"capped {cap:.1f} GiB survived={survived}", flush=True)

    def span(r):
        return f"{r['median']:.1f} s ({r['min']:.1f}-{r['max']:.1f})"

    lines = [
        "# Checkpoint 2: the whole fold, MLX vs torch/MPS",
        "",
        "Everything after ESMC (embed, trunk with MSA encoder and 20 loops, distogram, diffusion sampler at the checkpoint's "
        "steps, confidence head) on each case's real features with synthetic ESMC states (bf16). Torch: trunk-only copy loaded "
        "as the rescore worker loads it, eval, no grad, default chunk size, the forward bracketed by `torch.mps.synchronize()`. "
        "MLX: MLXFold in a clean process. 1 warmup, 3 timed runs (Checkpoint 1 runs varied by under 1 percent); `first` is the cold run. Memory: MLX re-run under a hard cap "
        "equal to torch's sampled peak driver allocation.",
        "",
        "| case | tokens | MSA | samples | torch median (min-max) | MLX median (min-max) | speedup | torch first | MLX first | torch peak | MLX under that cap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    passed = True
    for case, msa, samples, t, m, cap, survived, peak in rows:
        passed &= m["median"] <= 0.9 * t["median"] and survived
        lines.append(f"| {case} | {t['n']} | {msa} | {samples} | {span(t)} | {span(m)} | {t['median'] / m['median']:.2f}x | "
                     f"{t['first']:.1f} s | {m['first']:.1f} s | {t['peak_gb']:.1f} GB | {'survived' if survived else 'KILLED'} "
                     f"({cap:.1f} GiB{'' if peak is None else f'; MLX peak {peak:.1f} GiB'}) |")
    lines += ["", "Parity of the same fold: docs/parity/end_to_end.md.", "", f"**{'PASS' if passed else 'FAIL'}**"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
