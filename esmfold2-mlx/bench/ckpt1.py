"""Checkpoint 1: the full trunk, MLX vs torch/MPS, with and without an MSA. Writes docs/checkpoints/ckpt1-trunk.md.

    PYTHONPATH=. python bench/ckpt1.py

Per case, one process each, one after another under this driver's gpu lock: torch timing (2 warmups,
5 timed), MLX timing (same), then MLX once more under a hard memory cap equal to torch's peak. PASS
only if MLX's median is at least 10 percent below torch's for every case and every capped run survived.
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

CASES = (("vegf2", "none (single sequence)"), ("il17", "none (single sequence)"), ("tnf2_nomsa", "none (single sequence)"),
         ("tnf2_msa", "real, 2225 rows on both TNF chains"), ("il17_scfv", "none (single sequence)"))
OUT = REPO / "docs" / "checkpoints" / "ckpt1-trunk.md"
GIB = 2**30


def run(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.run([sys.executable, *cmd], capture_output=True, text=True, env=env, cwd=REPO)
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
        torch_r = last_json(run("bench/torch_trunk_bench.py", "--case", case))
        mlx_r = last_json(run("bench/trunk_bench.py", "--case", case))
        cap = torch_r["peak_gb"] * 1e9 / GIB
        capped = run("bench/trunk_bench.py", "--case", case, "--warmups", "0", "--repeats", "1", "--cap-gib", f"{cap:.2f}", check=False)
        survived = capped.returncode == 0
        peak = last_json(capped).get("mlx_peak_gib") if survived else None
        rows.append((case, msa, torch_r, mlx_r, cap, survived, peak))
        print(f"{case}: torch {torch_r['median']:.2f}s  mlx {mlx_r['median']:.2f}s  "
              f"x{torch_r['median'] / mlx_r['median']:.2f}  capped {cap:.1f} GiB survived={survived}", flush=True)

    def span(r):
        return f"{r['median']:.2f} s ({r['min']:.2f}-{r['max']:.2f})"

    lines = [
        "# Checkpoint 1: the full trunk, MLX vs torch/MPS",
        "",
        "Trunk = LM shim + MSA encoder + 21 loop steps of 48 pair blocks (the checkpoint's 20 loops) + readout + coda, on each "
        "case's real features with synthetic ESMC states (bf16, as ESMC hands them over; timing is value-independent). "
        "Torch: trunk-only copy loaded as the rescore worker loads it, eval, no grad, default chunk size, segments bracketed by "
        "`torch.mps.synchronize()` inside the real forward. MLX: a clean process, no torch. 2 warmups, 5 timed runs each; "
        "`first` is the first (cold) run. Memory: MLX re-run under a hard cap equal to torch's sampled peak driver allocation "
        "for the whole forward.",
        "",
        "| case | tokens | MSA | torch median (min-max) | MLX median (min-max) | speedup | torch first | MLX first | torch peak | MLX under that cap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    passed = True
    for case, msa, t, m, cap, survived, peak in rows:
        speed = t["median"] / m["median"]
        passed &= m["median"] <= 0.9 * t["median"] and survived
        lines.append(f"| {case} | {t['n']} | {msa} | {span(t)} | {span(m)} | {speed:.2f}x | {t['first']:.2f} s | {m['first']:.2f} s | "
                     f"{t['peak_gb']:.1f} GB | {'survived' if survived else 'KILLED'} ({cap:.1f} GiB{'' if peak is None else f'; MLX peak {peak:.1f} GiB'}) |")
    lines += [
        "",
        "Parity of the same trunk: tests/test_trunk_loop_vs_refs.py (torch-CPU references, 21 loops, 43 to 145 tokens, with and "
        "without MSA): 5e-7 to 2.3e-6 per loop, no growth with loop count (docs/parity/trunk_growth.md).",
        "",
        f"**{'PASS' if passed else 'FAIL'}**",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
