"""End-to-end parity table: MLXFold vs torch CPU and torch MPS on identical inputs and draws.
Writes docs/parity/end_to_end.md. MLX only (no torch in this process); reads the ``fold`` captures.

    PYTHONPATH=. python scripts/parity_report.py [case ...]
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "tests")]

import numpy as np  # noqa: E402

from esmfold2_mlx.config import FoldOptions  # noqa: E402
from esmfold2_mlx.fold import MLXFold  # noqa: E402
from esmfold2_mlx.rng import ReplayDraws  # noqa: E402
from helpers import kabsch_rmsd, load_ref, ref_path, rel_err, synthetic_lm_states  # noqa: E402
from preflight import preflight  # noqa: E402

DEFAULT_CASES = ("m49", "vegf1", "tnf60_msa", "tnf60_shallow")
OUT = REPO / "docs" / "parity" / "end_to_end.md"


def main(cases) -> None:
    preflight()
    fold = MLXFold(str(REPO / "weights"))
    lines = [
        "# End to end: MLXFold vs torch CPU and torch MPS",
        "",
        "Identical inputs (the esm featurizer's features, synthetic ESMC states) and identical randomness (the CPU run's "
        "recorded draws, replayed by MLX and by torch on MPS); checkpoint loops and steps, 2 samples. Coordinates: aligned "
        "RMSD in angstrom. `MPS-CPU` is torch's own device spread. Written by scripts/parity_report.py.",
        "",
        "| case | distogram rel. err. vs CPU / MPS | bins flipped vs CPU | sample | RMSD MLX-CPU | MLX-MPS | MPS-CPU | pTM MLX / CPU / MPS | ipTM MLX / CPU / MPS | pLDDT MLX / CPU / MPS |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        if not ref_path(case, "fold", "mps").exists():
            print(f"skip {case}: no fold captures")
            continue
        f = load_ref(case, "features")
        got = fold.fold(f, synthetic_lm_states(f["res_type"].shape[1]), FoldOptions(num_diffusion_samples=2),
                        draws=ReplayDraws(load_ref(case, "fold_draws")))
        cpu, mps = load_ref(case, "fold"), load_ref(case, "fold", "mps")
        mask = f["atom_attention_mask"][0].astype(bool)
        d_cpu, d_mps = rel_err(got["distogram_logits"], cpu["distogram_logits"]), rel_err(got["distogram_logits"], mps["distogram_logits"])
        flips = (got["distogram_logits"].argmax(-1) != cpu["distogram_logits"].argmax(-1)).mean() * 100
        for s in range(got["sample_atom_coords"].shape[0]):
            c = [kabsch_rmsd(got["sample_atom_coords"][s], cpu["sample_atom_coords"][s], mask),
                 kabsch_rmsd(got["sample_atom_coords"][s], mps["sample_atom_coords"][s], mask),
                 kabsch_rmsd(mps["sample_atom_coords"][s], cpu["sample_atom_coords"][s], mask)]
            trio = lambda k: " / ".join(f"{float(np.ravel(x[k])[s]):.4f}" for x in (got, cpu, mps))  # noqa: E731
            lines.append(f"| {case} | {d_cpu:.2e} / {d_mps:.2e} | {flips:.3f}% | {s} | {c[0]:.4f} | {c[1]:.4f} | {c[2]:.4f} | "
                         f"{trio('ptm')} | {trio('iptm')} | {trio('complex_plddt')} |")
        print(f"{case}: distogram {d_cpu:.2e}/{d_mps:.2e}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_CASES)
