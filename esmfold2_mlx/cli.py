"""``esmfold2-mlx pred``: fold protein complexes with ESMFold2 in MLX.

    bin/esmfold2-mlx pred -i job.json -o out/ [--samples N] [--loops N] [--steps N] [--seed S]
                          [--msa-depth D] [--lm-dropout P] [--weights DIR] [--cap-gib G] [--lm-cache DIR]
                          [--match-torch-rng [--torch-rng-device mps|cpu]]

This process is the torch side: it featurizes with the esm builder, runs ESMC-6B for complexes not in
the hidden-state cache, frees ESMC, then folds every job in one persistent MLX worker and decodes with
the esm builder. Per sample it writes ``out/<job>/sample_<i>/`` with complex.cif, pae_complex.npz,
plddt_complex.npz, tokens_complex.npz and, last, summary.json (the BindCraft-Hunter rescore layout).

Exit status: 0 done, 2 bad input or option, 3 the MLX worker failed (its message names the cause).
The whole run holds the gpu lock: one GPU run at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXIT_INPUT, EXIT_WORKER = 2, 3


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _write_sample(result, features, directory: Path, seconds: float) -> None:
    """One sample in the rescore worker's file layout; summary.json last, so a sample without it is partial."""
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    mask = features["token_attention_mask"][0].cpu().numpy()
    (directory / "complex.cif").write_text(result.complex.to_mmcif())
    np.savez(directory / "pae_complex.npz", pae=result.pae.float().numpy().astype(np.float32))
    np.savez(directory / "plddt_complex.npz", plddt=result.plddt.float().numpy().astype(np.float32))
    np.savez(directory / "tokens_complex.npz", asym_id=features["asym_id"][0].cpu().numpy(),
             residue_index=features["residue_index"][0].cpu().numpy(), mol_type=features["mol_type"][0].cpu().numpy(), token_mask=mask)
    _write_json(directory / "summary.json", {
        "iptm": float(result.iptm) if result.iptm is not None else 0.0, "ptm": float(result.ptm) if result.ptm is not None else 0.0,
        "plddt_mean": float(result.plddt.float().mean()), "tokens": int(mask.sum()), "seconds": round(seconds, 2)})


def pred(args) -> int:
    from .gpu_lock import gpu_lock
    from .inputs import parse_jobs

    with gpu_lock():
        try:
            jobs = parse_jobs(args.input)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            log(f"error: {error}")
            return EXIT_INPUT
        import torch  # the parent is the torch side; torch before the numpy-side modules

        from esm.models.esmfold2.processor import ESMFold2InputBuilder

        from .config import FoldOptions, load_config
        from .esmc_step import compute_states
        from .torch_draws import draw_torch_stream
        from .worker import RemoteFold, WorkerError

        builder = ESMFold2InputBuilder()
        parents = compute_states(jobs, builder=builder, device=args.esmc_device, cache_dir=args.lm_cache, seed=args.seed, log=log)
        remote = RemoteFold(args.weights, cap_gib=args.cap_gib)
        try:
            remote.start()
            log("worker ready")
            options = FoldOptions(num_loops=args.loops, num_sampling_steps=args.steps, num_diffusion_samples=args.samples,
                                  msa_max_depth=args.msa_depth, seed=args.seed, lm_dropout=args.lm_dropout)
            cfg = load_config(args.weights)
            for job in parents:
                features = {k: v.numpy() for k, v in job.features.items() if isinstance(v, torch.Tensor)}
                draws = None
                if args.match_torch_rng:  # torch's own stream for this seed, drawn on the chosen device
                    draws = draw_torch_stream(features, options, cfg, seed=args.seed, device=args.torch_rng_device)
                out = remote(features, job.lm_hidden_states, options, torch_draws=draws)
                import numpy as np

                (Path(args.output) / job.name).mkdir(parents=True, exist_ok=True)
                np.savez(Path(args.output) / job.name / "coords.npz", sample_atom_coords=out["sample_atom_coords"])
                results = builder.decode({k: torch.from_numpy(v) for k, v in out.items()}, job.features, job.chain_infos,
                                         num_diffusion_samples=args.samples)
                results = results if isinstance(results, list) else [results]
                for i, result in enumerate(results):
                    _write_sample(result, job.features, Path(args.output) / job.name / f"sample_{i}", remote.last_seconds / len(results))
                log(f"{job.name}: {len(results)} sample(s), fold {remote.last_seconds:.1f} s")
        except WorkerError as error:
            log(f"error: {error}")
            return EXIT_WORKER
        finally:
            remote.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="esmfold2-mlx", description="ESMFold2 in MLX")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pred", help="fold the jobs in a job file")
    p.add_argument("-i", "--input", required=True, help="job file (JSON, see esmfold2_mlx/inputs.py)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--samples", type=int, default=1, help="diffusion samples per job")
    p.add_argument("--loops", type=int, default=None, help="trunk loops (default: the checkpoint's 20)")
    p.add_argument("--steps", type=int, default=None, help="sampling steps (default: the checkpoint's 14)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--msa-depth", type=int, default=None, help="MSA rows per loop (default: the checkpoint's 1024)")
    p.add_argument("--lm-dropout", type=float, default=None, help="per-loop LM-pair dropout (default: the checkpoint's 0.25)")
    p.add_argument("--weights", default=str(REPO / "weights"))
    p.add_argument("--cap-gib", type=float, default=None, help="stop the MLX worker if its memory passes this")
    p.add_argument("--lm-cache", default=None, help="ESMC hidden-state cache (default ~/.cache/esmfold2_mlx/lm_states)")
    p.add_argument("--esmc-device", default="mps")
    p.add_argument("--match-torch-rng", action="store_true",
                   help="consume torch's random stream for --seed, so a fold can be compared with torch's for the same seed")
    p.add_argument("--torch-rng-device", choices=("mps", "cpu"), default="mps",
                   help="whose generator --match-torch-rng reproduces (CPU and MPS streams differ)")
    args = parser.parse_args(argv)
    if args.lm_cache is None:
        args.lm_cache = str(Path.home() / ".cache" / "esmfold2_mlx" / "lm_states")
    return pred(args)


if __name__ == "__main__":
    sys.exit(main())
