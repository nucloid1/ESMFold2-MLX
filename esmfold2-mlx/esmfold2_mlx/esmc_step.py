"""The torch-side parent step: featurize with the esm builder and run ESMC-6B (bf16) once per distinct
complex, caching its hidden states on disk; ESMC is freed before the MLX worker starts, so its 12 GB
and the MLX side are never resident together. Imports torch: never imported by the MLX worker.

The cache stores the bf16 states as their raw 16-bit patterns (exact, half the size of float32):
``<cache_dir>/<Job.states_key()>.npy``. A float32 ``.npy`` there is accepted too.
"""

import torch  # first: importing numpy-side modules before torch can crash an MPS forward

import gc
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .inputs import Job

DEFAULT_CACHE = Path.home() / ".cache" / "esmfold2_mlx" / "lm_states"
ESMC_REPO = "biohub/ESMC-6B"


@dataclass
class ParentJob:
    name: str
    job: Job
    features: dict  # torch CPU tensors, as the esm builder returns them (decode needs them)
    lm_hidden_states: np.ndarray  # [1, L, 81, 2560] float32
    chain_infos: list


def load_esmc(repo: str, device: str):
    """ESMC-6B built tensor by tensor straight to bf16 (about 12 GB resident; the library loader peaks
    near 37 GB). A copy of BindCraft-Hunter's rescore worker ``load_esmc_lazy``."""
    from accelerate import init_empty_weights
    from safetensors import safe_open

    from esm.models.esmc import EsmcModel
    from esm.models.esmc.checkpoint_layout import published_to_native
    from esm.models.hub import resolve_model_dir

    local = Path(resolve_model_dir(repo, local_files_only=True))
    config = EsmcModel.config_class.from_pretrained(local)
    with init_empty_weights():
        model = EsmcModel(config)
    weight_map = json.loads((local / "model.safetensors.index.json").read_text())["weight_map"]
    groups: dict = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(local / shard), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith("esmc."):
                    rest = key[len("esmc."):]
                    match = re.match(r"layers\.(\d+)\.", rest)
                    groups.setdefault(match.group(1) if match else "root", {})[rest] = handle.get_tensor(key).to(torch.bfloat16)
    native: dict = {}
    for name in list(groups):
        native.update(published_to_native(groups.pop(name)))
    incompatible = model.load_state_dict(native, strict=False, assign=True)
    unexpected = [k for k in incompatible.unexpected_keys if not k.endswith("_extra_state")]
    if unexpected or incompatible.missing_keys:
        raise RuntimeError(f"ESMC checkpoint did not map cleanly: unexpected {unexpected[:3]}, missing {incompatible.missing_keys[:3]}")
    del native
    model._materialize_uninitialized(device="cpu")
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _load_cached(path: Path) -> np.ndarray:
    a = np.load(path)
    return (a.astype(np.uint32) << 16).view(np.float32) if a.dtype == np.uint16 else a.astype(np.float32)


def _save_cached(path: Path, states: torch.Tensor) -> None:
    bits = states.to(torch.bfloat16).cpu().view(torch.int16).numpy().view(np.uint16)
    tmp = path.with_name(path.stem + ".partial.npy")
    np.save(tmp, bits)
    os.replace(tmp, path)


def compute_states(jobs: list, *, builder, esmc_repo: str = ESMC_REPO, device: str = "mps", cache_dir=DEFAULT_CACHE,
                   seed: int = 0, log=lambda m: print(m, file=sys.stderr, flush=True)) -> list:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, esmc = [], None
    for job in jobs:
        features, chain_infos = builder.prepare_input(job.to_spec(), seed=seed, device="cpu")
        path = cache_dir / f"{job.states_key()}.npy"
        if not path.exists():
            if esmc is None:
                log(f"loading {esmc_repo} on {device}")
                esmc = load_esmc(esmc_repo, device)
                log("esmc loaded")
            from esm.models.esmfold2.layers import compute_lm_hidden_states

            args = [features[k].to(device) for k in ("input_ids", "asym_id", "residue_index", "mol_type", "token_attention_mask")]
            with torch.no_grad():
                _save_cached(path, compute_lm_hidden_states(esmc, *args, lm_mask_pct=0.0))
            log(f"{job.name}: ESMC states cached")
        out.append(ParentJob(job.name, job, features, _load_cached(path), chain_infos))
    if esmc is not None:
        del esmc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log("esmc freed")
    return out
