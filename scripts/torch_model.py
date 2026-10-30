"""Torch-side loaders for references and baselines (esm env). Import this module before numpy-side ones.

``load_trunk_only`` is a copy of the rescore worker's loader
(bindcraft_hunter/rescore/worker/worker_models.py) so baselines are timed on exactly the model object
production uses: trunk-only copy, eval, no grad, the checkpoint's default chunk size.
"""

import torch  # first

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import Case  # noqa: E402

DEFAULT_CKPT = Path.home() / ".cache" / "huggingface" / "esmfold2-trunk" / "ESMFold2"
SAMPLER = {"noise_scale": 1.003, "step_scale": 1.5, "max_inference_sigma": 256.0}
SCHEDULE = {"inference_p": 7.0, "inference_s_max": 160.0, "inference_s_min": 0.0004}


def assert_same_sampler(model) -> None:
    head = model.config.structure_head
    found = {"noise_scale": head.noise_scale, "step_scale": head.step_scale, **{k: getattr(head, k) for k in SCHEDULE}}
    wanted = {"noise_scale": SAMPLER["noise_scale"], "step_scale": SAMPLER["step_scale"], **SCHEDULE}
    if any(abs(found[k] - wanted[k]) > 1e-9 for k in wanted):
        raise RuntimeError(f"the checkpoint's sampler config {found} differs from the shared settings {wanted}")


def load_trunk_only(device: str, checkpoint=DEFAULT_CKPT):
    """The ESMFold2 trunk without ESMC (about 1 GB), eval, no grad, kernels off."""
    from accelerate import init_empty_weights
    from safetensors import safe_open

    from esm.models.esmfold2.config import EsmFold2Config, default_module_flags
    from esm.models.esmfold2.hf_checkpoint import hf_state_dict_to_native
    from esm.models.esmfold2.model import EsmFold2Model

    checkpoint = Path(checkpoint)
    config = EsmFold2Config.from_pretrained(checkpoint, **default_module_flags(checkpoint))
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as handle:
            weight_map = {key: "model.safetensors" for key in handle.keys()}
    with init_empty_weights():
        model = EsmFold2Model(config)
    raw = {}
    for shard in sorted({s for k, s in weight_map.items() if not k.startswith("esmc.")}):
        with safe_open(str(checkpoint / shard), framework="pt") as handle:
            for key in handle.keys():
                if not key.startswith("esmc."):
                    raw[key] = handle.get_tensor(key)
    native = hf_state_dict_to_native(raw)
    model.esmc = None  # ESMC is never loaded in this process
    incompatible = model.load_state_dict(native, strict=False, assign=True)
    allow = [re.compile(p) for p in EsmFold2Model._keys_to_ignore_on_load_missing]
    ignore = [re.compile(p) for p in EsmFold2Model._keys_to_ignore_on_load_unexpected]
    missing = [k for k in incompatible.missing_keys if not any(p.search(k) for p in allow)]
    unexpected = [k for k in incompatible.unexpected_keys if not any(p.search(k) for p in ignore)]
    if missing or unexpected:
        raise RuntimeError(f"the checkpoint did not map cleanly onto the trunk-only model: missing {missing[:3]}, unexpected {unexpected[:3]}")
    del raw, native
    model._materialize_uninitialized(device="cpu")
    model.set_kernel_backend(None)
    model = model.to(device).eval().requires_grad_(False)
    assert_same_sampler(model)
    return model


def build_spec(case: Case):
    from esm.models.esmfold2.types import MSA, ProteinInput, StructurePredictionInput

    items = []
    for ids, sequence, a3m in case.chains:
        ids = list(ids) if isinstance(ids, tuple) else ids
        if isinstance(ids, list) and len(ids) == 1:
            ids = ids[0]
        items.append(ProteinInput(id=ids, sequence=sequence, msa=MSA.from_a3m(a3m) if a3m else None))
    return StructurePredictionInput(sequences=items)


def featurize(case: Case, seed: int, device: str):
    """``(features, chain_infos)`` from the esm package's own input builder."""
    from esm.models.esmfold2.processor import ESMFold2InputBuilder

    return ESMFold2InputBuilder().prepare_input(build_spec(case), seed=seed, device=device)
