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
            store[name] = (args[0].detach().cpu().numpy().copy(), output.detach().cpu().numpy().copy())
            mask = kwargs.get(mask_kwarg)
            if mask is not None and "mask" not in store:
                store["mask"] = mask.detach().cpu().numpy().copy()
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


LOOP_SAMPLE = (0, 1, 2, 5, 10, 15, 20)


def capture_trunk(case: Case, device: str, seed: int) -> None:
    """The looped trunk with synthetic ESMC states (helpers.synthetic_lm_states), checkpoint loops.

    Writes ``lm_shim`` (the shim's output), ``trunk_inputs`` (what ``_run_one_loop`` receives),
    ``loop_states`` (lm_encoder in/out at loop 0 and the pair after sampled loops), ``trunk_out`` (after
    readout and coda) and ``trunk_draws`` (every random draw, for replay). One sampler step only.
    """
    from esm.models.esmfold2.processor import _seed_context

    from helpers import synthetic_lm_states
    from torch_model import DrawRecorder

    model = load_trunk_only(device)
    features, _ = featurize(case, seed, device)
    n = features["res_type"].shape[1]
    lm = torch.from_numpy(synthetic_lm_states(n)).to(device)
    shim, inputs, states, out = {}, {}, {}, {}
    loop_count = [0]

    def on_shim(module, args, output):
        shim["lm_z"] = output.detach().cpu().numpy().copy()

    original_loop = model._run_one_loop

    def run_one_loop(**kw):
        inputs.update({k: kw[k].detach().cpu().numpy().copy() for k in ("z", "z_init", "lm_z", "pair_mask", "a", "b_mat")})
        inputs["total_steps"] = np.array(kw["total_steps"])
        msa_inputs = kw["_msa_inputs"]  # always present: the featurizer supplies a query-only MSA
        for k in ("msa", "msa_attention_mask", "has_deletion", "deletion_value", "x_inputs"):
            inputs[k] = msa_inputs[k].detach().cpu().numpy().copy()
        inputs["max_depth"] = np.array(msa_inputs["max_depth"])
        return original_loop(**kw)

    msa = {}

    def first(name, getter):
        def hook(module, args, kwargs, output):
            if name not in msa:
                msa[name] = True
                for k, v in getter(args, kwargs, output).items():
                    msa[k] = v.detach().cpu().numpy().copy()
        return hook

    enc = model.msa_encoder
    enc.register_forward_hook(first("enc", lambda a, kw, o: {**{k: kw[k] for k in (
        "x_pair", "x_inputs", "msa_oh", "has_deletion", "deletion_value", "msa_attention_mask")}, "out": o}), with_kwargs=True)
    enc.blocks[0].outer_product_mean.register_forward_hook(
        first("opm0", lambda a, kw, o: {"opm0_m": a[0], "opm0_mask": a[1], "opm0_out": o}), with_kwargs=True)
    enc.blocks[0].msa_pair_weighted_averaging.register_forward_hook(
        first("pwa0", lambda a, kw, o: {"pwa0_m": a[0], "pwa0_pair": a[1], "pwa0_pair_mask": a[2], "pwa0_out": o}), with_kwargs=True)
    for i in (0, 3):
        enc.blocks[i].register_forward_hook(first(f"blk{i}", lambda a, kw, o, i=i: {
            f"blk{i}_m": a[0], f"blk{i}_pair": a[1], f"blk{i}_msa_mask": a[2], f"blk{i}_pair_mask": a[3],
            f"blk{i}_m_out": o[0], f"blk{i}_pair_out": o[1]}), with_kwargs=True)

    def on_lm_encoder(module, args, kwargs, output):
        if "lm_enc0_in" not in states:
            states["lm_enc0_in"] = args[0].detach().cpu().numpy().copy()
            states["lm_enc0_out"] = output.detach().cpu().numpy().copy()

    def on_trunk(module, args, kwargs, output):
        k = loop_count[0]
        loop_count[0] += 1
        if k in LOOP_SAMPLE:
            states[f"loop{k}_out"] = output.detach().cpu().numpy().copy()

    def on_coda(module, args, kwargs, output):
        out["z"] = output.detach().cpu().numpy().copy()

    model.language_model.register_forward_hook(on_shim)
    model._run_one_loop = run_one_loop
    model.lm_encoder.register_forward_hook(on_lm_encoder, with_kwargs=True)
    model.folding_trunk.register_forward_hook(on_trunk, with_kwargs=True)
    model.parcae_coda.register_forward_hook(on_coda, with_kwargs=True)

    with torch.no_grad(), _seed_context(seed), DrawRecorder() as recorder:
        model(**features, lm_hidden_states=lm, num_sampling_steps=1, num_diffusion_samples=1, **SAMPLER)

    _save(case, device, "lm_shim", shim)
    _save(case, device, "trunk_inputs", inputs)
    _save(case, device, "loop_states", states)
    _save(case, device, "trunk_out", out)
    _save(case, device, "msa_encoder", {k: v for k, v in msa.items() if v is not True})
    _save(case, device, "trunk_draws", recorder.arrays)
    print(f"{loop_count[0]} trunk loops; draw sites: {sorted({k.split('/')[0] for k in recorder.arrays})}")


def capture_heads(case: Case, device: str, seed: int) -> None:
    """Distogram and confidence heads in the real forward: synthetic ESMC states, the checkpoint's
    sampling steps, TWO diffusion samples."""
    from esm.models.esmfold2.processor import _seed_context

    from helpers import synthetic_lm_states

    model = load_trunk_only(device)
    features, _ = featurize(case, seed, device)
    lm = torch.from_numpy(synthetic_lm_states(features["res_type"].shape[1])).to(device)
    out: dict = {}

    def keep(**arrays):
        for k, v in arrays.items():
            out[k] = v.detach().cpu().numpy().copy() if isinstance(v, torch.Tensor) else np.array(v)

    model.parcae_coda.register_forward_hook(lambda m, a, kw, o: keep(z_coda=o), with_kwargs=True)
    model.distogram_head.register_forward_hook(lambda m, a, o: keep(disto_in=a[0], disto_out=o))

    def on_conf(module, args, kwargs, output):
        keep(num_diffusion_samples=kwargs["num_diffusion_samples"])
        keep(**{f"conf_{k}": v for k, v in kwargs.items() if isinstance(v, torch.Tensor)})
        keep(**{f"conf_out_{k}": v for k, v in output.items()})

    ch = model.confidence_head
    ch.register_forward_hook(on_conf, with_kwargs=True)
    ch.folding_trunk.register_forward_hook(
        lambda m, a, kw, o: keep(conf_trunk_in=a[0], conf_trunk_mask=kw["pair_attention_mask"], conf_trunk_out=o), with_kwargs=True)
    ch.row_attention_pooling.register_forward_hook(lambda m, a, o: keep(conf_pool_in=a[0], conf_pool_mask=a[1], conf_pool_out=o))

    with torch.no_grad(), _seed_context(seed):
        model(**features, lm_hidden_states=lm, num_diffusion_samples=2, **SAMPLER)
    out["z"] = out.pop("z_coda")  # the distogram head's z (float32 coda output)
    _save(case, device, "heads", out)


def _np(t) -> np.ndarray:
    t = t.detach().cpu()
    return (t.float() if t.dtype == torch.bfloat16 else t).numpy().copy()


def capture_embed(case: Case, device: str, seed: int) -> None:
    """Inputs embedder (atom encoder, block 0), rel_pos, token bonds and z_init; no ESMC states, one
    loop step and one sampler step (only the prelude matters). Run on cpu AND mps: the atom attention
    runs in bf16 in torch, and the cpu/mps spread is the parity gate for those stages."""
    from esm.models.esmfold2.processor import _seed_context

    model = load_trunk_only(device)
    features, _ = featurize(case, seed, device)
    out: dict = {}
    model.inputs_embedder.register_forward_hook(lambda m, a, kw, o: out.update(x_inputs=_np(o)), with_kwargs=True)

    def on_encoder(module, args, kwargs, output):
        a, q, c, params, _ = output
        out.update(atom_a=_np(a), atom_q=_np(q), cos=_np(params[0]), sin=_np(params[1]),
                   valid=_np(kwargs["atom_attention_mask"]).astype(bool))

    enc = model.inputs_embedder.atom_attention_encoder
    enc.register_forward_hook(on_encoder, with_kwargs=True)
    enc.atom_transformer.blocks[0].register_forward_hook(
        lambda m, a, o: out.update(blk0_x=_np(a[0]), blk0_c=_np(a[1]), blk0_out=_np(o)))
    model.rel_pos.register_forward_hook(lambda m, a, kw, o: out.update(rel_pos=_np(o)), with_kwargs=True)
    model.token_bonds.register_forward_hook(lambda m, a, o: out.update(token_bonds=_np(o)))
    original_loop = model._run_one_loop

    def run_one_loop(**kw):
        out["z_init"] = _np(kw["z_init"])
        return original_loop(**kw)

    model._run_one_loop = run_one_loop
    with torch.no_grad(), _seed_context(seed):
        model(**features, lm_hidden_states=None, num_loops=0, num_sampling_steps=1, num_diffusion_samples=1, **SAMPLER)
    _save(case, device, "embed", out)


SAMPLE_KEYS = ("z_trunk", "s_inputs", "relative_position_encoding", "ref_pos", "ref_charge", "ref_mask", "ref_element",
               "ref_atom_name_chars", "ref_space_uid", "tok_idx", "asym_id", "residue_index", "entity_id", "token_index",
               "sym_id", "token_attention_mask")
SAMPLER_SITES = ("sample", "_random_rotations", "_center_random_augmentation")


def capture_sampler(case: Case, device: str, seed: int) -> None:
    """cpu: the real forward (synthetic ESMC states, checkpoint steps, 2 samples) records the sampler's
    inputs, every denoiser call's output (first and last call with inputs) and the sampler draws.
    mps: replays the cpu sampler inputs with the cpu draws, and re-runs the cpu-recorded first and
    last denoiser calls, so all devices see identical inputs."""
    from torch_model import DrawRecorder, DrawReplayer

    model = load_trunk_only(device)
    dm = model.structure_head.diffusion_module
    calls: list = []
    t_hats: list = []
    first_last: dict = {}

    def on_dm(module, args, kwargs, output):
        calls.append(_np(output["x_denoised"]))
        t_hats.append(float(kwargs["t_hat"][0]))
        record = {k: _np(v) for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        if len(calls) == 1:
            first_last["first"] = record
        first_last["last"] = record

    dm.register_forward_hook(on_dm, with_kwargs=True)
    out: dict = {}
    if device == "cpu":
        from esm.models.esmfold2.processor import _seed_context

        from helpers import synthetic_lm_states

        features, _ = featurize(case, seed, device)
        lm = torch.from_numpy(synthetic_lm_states(features["res_type"].shape[1]))
        original = model.structure_head.sample

        def sample(**kw):
            result = original(**kw)
            out.update({f"in_{k}": _np(kw[k]) for k in SAMPLE_KEYS})
            out["num_diffusion_samples"] = np.array(kw["num_diffusion_samples"])
            out["out"] = _np(result["sample_atom_coords"])
            return result

        model.structure_head.sample = sample
        with torch.no_grad(), _seed_context(seed), DrawRecorder() as recorder:
            model(**features, lm_hidden_states=lm, num_diffusion_samples=2, **SAMPLER)
        _save(case, device, "sampler_draws", {k: v for k, v in recorder.arrays.items() if k.split(":")[0] in SAMPLER_SITES})
        for i, name in ((0, "first"), (len(calls) - 1, "last")):
            out.update({f"call{i}_{k}": v for k, v in first_last[name].items()})
            out[f"call{i}_out"] = calls[i]
    else:
        cpu = dict(np.load(ref_path(case.name, "sampler", "cpu")))
        draws = dict(np.load(ref_path(case.name, "sampler_draws", "cpu")))
        n = int(cpu["num_diffusion_samples"])
        inputs = {k: torch.from_numpy(cpu[f"in_{k}"]).to(device) for k in SAMPLE_KEYS}
        with torch.no_grad(), DrawReplayer(draws):
            result = model.structure_head.sample(**inputs, s_trunk=None, num_diffusion_samples=n, **SAMPLER)
        out["out"] = _np(result["sample_atom_coords"])
        sampler_calls = len(calls)
        for i in (0, int(cpu["n_calls"]) - 1):  # the cpu-recorded calls, re-run here on identical inputs
            kw = {k[len(f"call{i}_"):]: torch.from_numpy(v).to(device) for k, v in cpu.items()
                  if k.startswith(f"call{i}_") and k != f"call{i}_out"}
            with torch.no_grad():
                out[f"replay{i}_out"] = _np(dm(**kw, s_trunk=None, num_diffusion_samples=n, return_token_repr=True)["x_denoised"])
        calls, t_hats = calls[:sampler_calls], t_hats[:sampler_calls]
    out["denoised"] = np.stack(calls)  # [calls, samples, atoms, 3]
    out["t_hats"] = np.array(t_hats)
    out["n_calls"] = np.array(len(calls))
    _save(case, device, "sampler", out)


def capture_fold(case: Case, device: str, seed: int) -> None:
    """The whole forward (synthetic ESMC states, checkpoint loops and steps, 2 samples). cpu records
    every random draw (``fold_draws``); mps replays those draws, so both devices see the same randomness."""
    from esm.models.esmfold2.processor import _seed_context

    from helpers import synthetic_lm_states
    from torch_model import DrawRecorder, DrawReplayer

    model = load_trunk_only(device)
    features, _ = featurize(case, seed, device)
    lm = torch.from_numpy(synthetic_lm_states(features["res_type"].shape[1])).to(device)
    with torch.no_grad(), _seed_context(seed):
        if device == "cpu":
            with DrawRecorder() as recorder:
                out = model(**features, lm_hidden_states=lm, num_diffusion_samples=2, **SAMPLER)
            _save(case, device, "fold_draws", recorder.arrays)
        else:
            with DrawReplayer(dict(np.load(ref_path(case.name, "fold_draws", "cpu")))):
                out = model(**features, lm_hidden_states=lm, num_diffusion_samples=2, **SAMPLER)
    _save(case, device, "fold", {k: _np(v) for k, v in out.items() if isinstance(v, torch.Tensor)})


def capture_features(case: Case, device: str, seed: int) -> None:
    """The esm builder's feature dict (shared by both devices), numpy."""
    features, _ = featurize(case, seed, "cpu")
    _save(case, device, "features", {k: v.numpy() for k, v in features.items() if isinstance(v, torch.Tensor)})


CAPTURES = {"pair_block": capture_pair_block, "trunk": capture_trunk, "features": capture_features, "heads": capture_heads,
            "embed": capture_embed, "sampler": capture_sampler, "fold": capture_fold}


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
