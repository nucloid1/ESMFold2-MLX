"""Capture torch reference activations for the parity tests.

    PYTHONPATH=. python scripts/capture_refs.py CASE --device cpu --stages pair_block [--seed 0]

Each stage is a function registered in ``CAPTURES`` and writes ``refs/<case>/<device>/<stage>.npz``
(layout in tests/helpers.py). Later tasks add stages. Every capture holds the gpu lock for its whole
run, CPU-only ones included.
"""

import torch  # first

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))

from cases import CASES, Case  # noqa: E402
from helpers import ref_path  # noqa: E402
from preflight import preflight  # noqa: E402
from torch_model import SAMPLER, featurize, load_trunk_only  # noqa: E402

PAIR_BLOCKS = (0, 23, 47)


def _save(case: Case, device: str, stage: str, arrays: dict) -> None:
    path = ref_path(case.name, stage, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"wrote {path} ({len(arrays)} arrays)")


def capture_pair_block(case: Case, device: str, seed: int) -> None:
    """Blocks 0, 23, 47 and their three sub-ops, first loop of the real forward, no ESMC hidden states.

    ``num_loops=1`` keeps the run short; the trunk is called with the real pair stream, so blocks 23
    and 47 see the state propagated through the blocks before them.
    """
    from esm.models.esmfold2.processor import _seed_context

    model = load_trunk_only(device)
    features, _ = featurize(case, seed, device)
    store: dict = {}

    def record(name, mask_kwarg):
        def hook(module, args, kwargs, output):
            if name in store:  # first loop only
                return
            store[name] = (args[0].detach().cpu().numpy(), output.detach().cpu().numpy())
            mask = kwargs.get(mask_kwarg)
            if mask is not None and "mask" not in store:
                store["mask"] = mask.detach().cpu().numpy()
        return hook

    for i in PAIR_BLOCKS:
        block = model.folding_trunk.blocks[i]
        block.register_forward_hook(record(f"b{i}", "pair_attention_mask"), with_kwargs=True)
        block.tri_mul_out.register_forward_hook(record(f"b{i}_tri_out", "mask"), with_kwargs=True)
        block.tri_mul_in.register_forward_hook(record(f"b{i}_tri_in", "mask"), with_kwargs=True)
        block.pair_transition.register_forward_hook(record(f"b{i}_trans", "mask"), with_kwargs=True)

    with torch.no_grad(), _seed_context(seed):
        model(**features, lm_hidden_states=None, num_loops=1, num_sampling_steps=1, num_diffusion_samples=1, **SAMPLER)

    arrays = {"mask": store["mask"]}
    for i in PAIR_BLOCKS:
        arrays[f"b{i}_in"], arrays[f"b{i}_out"] = store[f"b{i}"]
        arrays[f"b{i}_tri_out_in"], arrays[f"b{i}_tri_out_out"] = store[f"b{i}_tri_out"]
        arrays[f"b{i}_tri_in_in"], arrays[f"b{i}_tri_in_out"] = store[f"b{i}_tri_in"]
        arrays[f"b{i}_trans_in"], arrays[f"b{i}_trans_out"] = store[f"b{i}_trans"]
    _save(case, device, "pair_block", arrays)


CAPTURES = {"pair_block": capture_pair_block}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--stages", required=True, help=f"comma separated, from {sorted(CAPTURES)}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    stages = args.stages.split(",")
    unknown = [s for s in stages if s not in CAPTURES]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; known: {sorted(CAPTURES)}")
    preflight()
    for stage in stages:
        CAPTURES[stage](CASES[args.case], args.device, args.seed)


if __name__ == "__main__":
    main()
