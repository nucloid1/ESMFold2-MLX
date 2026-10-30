"""Real ESMC-6B end to end: torch/MPS vs MLX on real complexes with real ESMC hidden states.

    PYTHONPATH=. python scripts/real_esmc_parity.py [case ...]      (default: vegf2 tnf2_msa)

Per case: ESMC-6B (bf16, on MPS, cached) gives the hidden states; torch on MPS folds twice with the
same seed (its own repeat spread); MLX folds once consuming torch's exact MPS random stream for that
seed (--match-torch-rng). Writes docs/parity/real_esmc.md. Uses the gpu lock for the whole run; the
MLX fold runs in the worker process.
"""

import torch  # first

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "tests")]

import numpy as np  # noqa: E402

from cases import CASES  # noqa: E402
from esmfold2_mlx.config import FoldOptions, load_config  # noqa: E402
from esmfold2_mlx.esmc_step import compute_states  # noqa: E402
from esmfold2_mlx.inputs import Chain, Job  # noqa: E402
from esmfold2_mlx.torch_draws import draw_torch_stream  # noqa: E402
from esmfold2_mlx.worker import RemoteFold  # noqa: E402
from helpers import kabsch_rmsd  # noqa: E402
from preflight import preflight  # noqa: E402
from torch_model import SAMPLER, load_trunk_only  # noqa: E402

SAMPLES, SEED = 2, 0
OUT = REPO / "docs" / "parity" / "real_esmc.md"


def to_job(case) -> Job:
    return Job(case.name, tuple(Chain(ids if isinstance(ids, tuple) else (ids,), seq, a3m) for ids, seq, a3m in case.chains))


def main(names) -> None:
    preflight()
    from esm.models.esmfold2.processor import ESMFold2InputBuilder, _seed_context

    weights = str(REPO / "weights")
    cfg = load_config(weights)
    parents = compute_states([to_job(CASES[n]) for n in names], builder=ESMFold2InputBuilder(), device="mps")
    model = load_trunk_only("mps")
    remote = RemoteFold(weights)
    lines = [
        "# Real ESMC-6B end to end: MLX vs torch on MPS",
        "",
        f"Real ESMC-6B hidden states (bf16), the checkpoint's loops and steps, {SAMPLES} samples, seed {SEED}. Torch on MPS folds "
        "twice with the same seed (`torch A` vs `torch B` is torch's own repeat spread on this device); MLX consumes torch's exact "
        "MPS random stream for the seed (`--match-torch-rng`). Coordinates: aligned RMSD in angstrom. Written by "
        "scripts/real_esmc_parity.py.",
        "",
        "| case | tokens | sample | pTM torch A / B / MLX | ipTM torch A / B / MLX | pLDDT torch A / B / MLX | RMSD MLX-A | RMSD B-A |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    try:
        for job in parents:
            features = {k: v.to("mps") for k, v in job.features.items() if isinstance(v, torch.Tensor)}
            lm = torch.from_numpy(job.lm_hidden_states).to("mps", torch.bfloat16)
            runs = []
            for _ in range(2):
                with torch.no_grad(), _seed_context(SEED):
                    out = model(**features, lm_hidden_states=lm, num_diffusion_samples=SAMPLES, msa_column_mask_rate=0.1, **SAMPLER)
                runs.append({k: v.float().cpu().numpy() for k, v in out.items() if k in ("ptm", "iptm", "complex_plddt", "sample_atom_coords")})
                del out
                torch.mps.empty_cache()
            arrays = {k: v.numpy() for k, v in job.features.items() if isinstance(v, torch.Tensor)}
            options = FoldOptions(num_diffusion_samples=SAMPLES, seed=SEED)
            mlx = remote(arrays, job.lm_hidden_states, options,
                         torch_draws=draw_torch_stream(arrays, options, cfg, seed=SEED, device="mps"))
            mask = arrays["atom_attention_mask"][0].astype(bool)
            for s in range(SAMPLES):
                trio = lambda k: " / ".join(f"{float(np.ravel(r[k])[s]):.4f}" for r in (runs[0], runs[1], mlx))  # noqa: E731
                lines.append(f"| {job.name} | {job.job.tokens()} | {s} | {trio('ptm')} | {trio('iptm')} | {trio('complex_plddt')} | "
                             f"{kabsch_rmsd(mlx['sample_atom_coords'][s], runs[0]['sample_atom_coords'][s], mask):.4f} | "
                             f"{kabsch_rmsd(runs[1]['sample_atom_coords'][s], runs[0]['sample_atom_coords'][s], mask):.4f} |")
            print(lines[-1], flush=True)
    finally:
        remote.close()
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["vegf2", "tnf2_msa"])
