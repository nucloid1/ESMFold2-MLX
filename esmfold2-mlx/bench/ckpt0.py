"""Checkpoint 0: one trunk block, MLX against torch/MPS, N of 250 and 400. Writes docs/checkpoints/ckpt0-block.md.

    PYTHONPATH=. python bench/ckpt0.py

Each measurement runs in its own process, one after another, under this driver's gpu lock. The gate
is the production torch call (checkpoint default chunk size 64): PASS only if MLX's median is at least
10 percent below torch's at both sizes. The torch chunk=None numbers are reported next to it.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from preflight import preflight  # noqa: E402

SIZES = (250, 400)
OUT = REPO / "docs" / "checkpoints" / "ckpt0-block.md"


def run(script: str, *extra: str) -> dict:
    cmd = [sys.executable, str(REPO / "bench" / script), *extra]
    env = {"PYTHONPATH": str(REPO)}
    import os

    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env}, cwd=REPO)
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    preflight()
    rows = {}
    for n in SIZES:
        rows[n] = {
            "mlx": run("block_bench.py", "--n", str(n)),
            "torch64": run("torch_block_bench.py", "--n", str(n), "--chunk", "64"),
            "torch_none": run("torch_block_bench.py", "--n", str(n), "--chunk", "none"),
        }
        print(f"N={n}: mlx {rows[n]['mlx']['median']:.3f}s  torch(64) {rows[n]['torch64']['median']:.3f}s  torch(none) {rows[n]['torch_none']['median']:.3f}s", flush=True)

    def fmt(r):
        return f"{r['median'] * 1000:.0f} ms (min {r['min'] * 1000:.0f}, max {r['max'] * 1000:.0f})"

    lines = [
        "# Checkpoint 0: one pair-trunk block, MLX vs torch/MPS",
        "",
        "fp32, block 0 with real weights, random pair input (timing is value-independent), 2 warmups and 7 timed runs per cell, "
        "each cell in its own process. Torch is the trunk-only copy loaded as the rescore worker loads it, eval, no grad. "
        "`torch (64)` is production (checkpoint default chunk size 64); `torch (none)` calls `set_chunk_size(None)`.",
        "",
        "| N | MLX | torch (64, production) | torch (none) | MLX speedup vs 64 | MLX speedup vs none |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    passed = True
    for n, r in rows.items():
        s64 = r["torch64"]["median"] / r["mlx"]["median"]
        snone = r["torch_none"]["median"] / r["mlx"]["median"]
        passed &= r["mlx"]["median"] <= 0.9 * r["torch64"]["median"]
        lines.append(f"| {n} | {fmt(r['mlx'])} | {fmt(r['torch64'])} | {fmt(r['torch_none'])} | {s64:.2f}x | {snone:.2f}x |")
    lines += ["", "Per-op medians (ms), each op alone on the same input:", "", "| N | impl | tri_mul_out | tri_mul_in | pair_transition |", "| --- | --- | --- | --- | --- |"]
    for n, r in rows.items():
        for name, label in (("mlx", "MLX"), ("torch64", "torch (64)"), ("torch_none", "torch (none)")):
            ops = r[name]["ops"]
            lines.append(f"| {n} | {label} | " + " | ".join(f"{ops[k] * 1000:.0f}" for k in ("tri_mul_out", "tri_mul_in", "pair_transition")) + " |")
    beats_none = all(r["mlx"]["median"] <= 0.9 * r["torch_none"]["median"] for r in rows.values())
    lines += [
        "",
        f"MLX beats the production call by at least 10 percent at every size: **{'yes' if passed else 'no'}**. "
        f"MLX also beats torch with chunking off by at least 10 percent at every size: **{'yes' if beats_none else 'no'}**"
        + ("" if beats_none or not passed else " (so part of the production-call speedup comes from torch's chunk setting, not from MLX; Ethan to weigh)") + ".",
        "",
        f"**{'PASS' if passed else 'FAIL'}**",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
