"""Weights bookkeeping, config, and the converter's key accounting.

The converter tests run scripts/convert_weights.py once (module-scoped fixture, a subprocess of this
pytest session so it runs under the session's gpu lock) on the real trunk-only checkpoint.
"""

import collections
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from esmfold2_mlx.config import FoldOptions, ModelConfig, load_config
from esmfold2_mlx.weights import Weights, key, load_group

REPO = Path(__file__).resolve().parents[1]
SRC = Path.home() / ".cache" / "huggingface" / "esmfold2-trunk" / "ESMFold2"
SRC_FILE = SRC / "model.safetensors"


def _child_env():
    env = dict(os.environ)  # keeps ESMFOLD2_MLX_LOCK_OWNER: the child runs under this session's lock
    env["PYTHONPATH"] = str(REPO)
    return env


# --------------------------------------------------------------------------- Weights / key


def test_assert_all_used_names_unread_keys():
    w = Weights({"a.w": np.ones(2), "b.w": np.ones(2)})
    w["a.w"]
    assert w.unused("a") == []
    assert w.unused() == ["b.w"]
    with pytest.raises(KeyError, match=r"b\.w"):
        w.assert_all_used()
    w.assert_all_used("a")  # nothing under "a" is unread


def test_get_records_reads_and_allow_skips_prefixes():
    w = Weights({"a.w": 1, "b.w": 2, "c.w": 3})
    assert w.get("a.w") == 1
    assert w.get("missing", 7) == 7  # a miss records nothing and raises nothing
    assert w.unused() == ["b.w", "c.w"]
    w.assert_all_used(allow=("b", "c"))
    with pytest.raises(KeyError, match=r"c\.w"):
        w.assert_all_used(allow=("b",))


def test_key_joins_without_leading_dot():
    assert key("", "weight") == "weight"
    assert key("blocks.0", "weight") == "blocks.0.weight"
    assert key("a", "b.c") == "a.b.c"


def test_load_group_returns_tracked_weights(tmp_path):
    w_mat = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_file(
        {"weight": w_mat, "blocks.0.bias": np.ones(3, dtype=np.float32)},
        str(tmp_path / "toy.safetensors"),
    )
    w = load_group(str(tmp_path), "toy")
    assert isinstance(w, Weights)
    assert sorted(w) == ["blocks.0.bias", "weight"]
    assert w.unused() == ["blocks.0.bias", "weight"]
    np.testing.assert_array_equal(np.array(w["weight"]), w_mat)
    assert w.unused() == ["blocks.0.bias"]
    with pytest.raises(KeyError, match=r"blocks\.0\.bias"):
        w.assert_all_used()
    w.get("blocks.0.bias")
    w.assert_all_used()


def test_load_group_missing_file_names_the_converter(tmp_path):
    with pytest.raises(FileNotFoundError, match="convert_weights"):
        load_group(str(tmp_path), "nope")


def test_mlx_package_never_imports_torch():
    code = (
        "import sys\n"
        "import esmfold2_mlx.weights, esmfold2_mlx.config, esmfold2_mlx.gpu_lock\n"
        "bad = [m for m in sys.modules if m == 'torch' or m == 'esm' or m.startswith('esm.')]\n"
        "assert not bad, bad\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], env=_child_env(), cwd=REPO, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


# --------------------------------------------------------------------------- FoldOptions / config


def test_fold_options_defaults():
    o = FoldOptions()
    assert (o.num_loops, o.num_sampling_steps, o.num_diffusion_samples) == (None, None, 1)
    assert (o.msa_max_depth, o.msa_column_mask_rate) == (None, 0.1)
    assert (o.noise_scale, o.step_scale, o.max_inference_sigma, o.seed) == (1.003, 1.5, 256.0, 0)


# --------------------------------------------------------------------------- converter


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    if not SRC_FILE.exists():
        pytest.fail(f"source checkpoint missing: {SRC_FILE}")
    out = tmp_path_factory.mktemp("weights")
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "convert_weights.py"), "--out", str(out)],
        env=_child_env(), cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    manifest = json.loads((out / "manifest.json").read_text())
    yield out, manifest, proc.stdout
    shutil.rmtree(out, ignore_errors=True)  # about 1 GB per conversion


def _header_tensors(path):
    """(name -> (shape, dtype)) from a safetensors header alone, no tensor data read."""
    with safe_open(str(path), "numpy") as f:
        return {k: (tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype()) for k in f.keys()}


def _numel(shape):
    return int(np.prod(shape, dtype=np.int64))


def test_conversion_accounts_for_every_source_tensor(converted):
    out, manifest, stdout = converted
    src = _header_tensors(SRC_FILE)
    assert manifest["source_tensors"] == 1616 == len(src)
    src_elements = sum(_numel(s) for s, _ in src.values())
    assert manifest["source_elements"] == src_elements
    assert src_elements * 4 == 937_903_372  # the checkpoint's recorded fp32 byte count

    # Every source key appears in the map, and nothing else does.
    assert set(manifest["map"]) == set(src)

    # What was actually written, counted from the files on disk.
    on_disk = {}
    for f in sorted(out.glob("*.safetensors")):
        for name, (shape, dtype) in _header_tensors(f).items():
            assert dtype == "F32", (f.name, name, dtype)
            on_disk[(f.name, name)] = shape
    converted_elements = sum(_numel(s) for s in on_disk.values())
    assert converted_elements == src_elements == manifest["converted_elements"]
    assert len(on_disk) == manifest["converted_tensors"]

    # Every written tensor is in the manifest map exactly once, and every mapped tensor was written.
    mapped = collections.defaultdict(set)  # (file, stripped key) -> native keys that land there
    native_sources = collections.defaultdict(list)
    for src_key, entry in manifest["map"].items():
        assert len(entry["native"]) == len(entry["key"])
        for native, stripped in zip(entry["native"], entry["key"]):
            mapped[(entry["file"], stripped)].add(native)
            native_sources[native].append(src_key)
    assert set(mapped) == set(on_disk)
    assert all(len(v) == 1 for v in mapped.values())

    # hf_state_dict_to_native packs q/k/v (and k/v) and splits a fused gate/up projection, so the
    # tensor count changes; the manifest says how, and the identity must hold. Elements never change.
    packs = {n: s for n, s in native_sources.items() if len(s) > 1}
    splits = [k for k, e in manifest["map"].items() if len(e["native"]) == 2]
    change = manifest["tensor_count_changes"]
    assert change["packed_native_tensors"] == len(packs)
    assert change["packed_source_tensors"] == sum(len(s) for s in packs.values())
    assert change["split_source_tensors"] == len(splits)
    assert change["split_native_tensors"] == 2 * len(splits)
    assert (
        1616 - sum(len(s) - 1 for s in packs.values()) + len(splits)
        == manifest["native_tensors"]
        == len(native_sources)
        == len(on_disk)
    )
    assert manifest["native_elements"] == src_elements

    # Group names printed by the converter and recorded in the manifest agree.
    assert set(manifest["groups"]) == {e["group"] for e in manifest["map"].values()}
    for group in manifest["groups"]:
        assert group in stdout
    for bare in ("parcae_b_cont", "parcae_log_delta", "parcae_log_a"):
        assert manifest["groups"][bare]["bare"] is True
        assert (out / f"{bare}.safetensors").exists()


REF_SCRIPT = """\
import sys, json, torch, numpy as np
from safetensors.torch import load_file
from esm.models.esmfold2.hf_checkpoint import hf_state_dict_to_native
src, keys_json, out = sys.argv[1:4]
native = hf_state_dict_to_native(load_file(src))
keys = json.load(open(keys_json))
np.savez(out, **{f"k{i}": native[k].contiguous().numpy() for i, k in enumerate(keys)})
"""


def test_converted_tensors_equal_native_mapped_source(converted, tmp_path):
    out, manifest, _ = converted
    where = {}  # native key -> (group, stripped key, transform)
    for entry in manifest["map"].values():
        for native, stripped in zip(entry["native"], entry["key"]):
            where[native] = (entry["group"], stripped, entry["transform"])
    assert len(where) == manifest["native_tensors"]

    # 50 random native keys, plus every packed or split tensor, every bare parameter and the first
    # key of every group, so none of the unusual paths is left to chance.
    forced = {n for n, (_, _, t) in where.items() if t in ("pack", "split")}
    forced |= {n for n, (g, s, _) in where.items() if manifest["groups"][g]["bare"]}
    first_of_group = {}
    for n in sorted(where):
        first_of_group.setdefault(where[n][0], n)
    forced |= set(first_of_group.values())
    assert any(where[n][2] == "pack" for n in forced) and any(where[n][2] == "split" for n in forced)
    rest = sorted(set(where) - forced)
    sample = sorted(forced | set(random.Random(0).sample(rest, 50)))
    assert len(sample) >= 50

    keys_json = tmp_path / "keys.json"
    keys_json.write_text(json.dumps(sample))
    ref_npz = tmp_path / "ref.npz"
    res = subprocess.run(
        [sys.executable, "-c", REF_SCRIPT, str(SRC_FILE), str(keys_json), str(ref_npz)],
        env=_child_env(), cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert res.returncode == 0, res.stderr[-4000:]
    ref = np.load(ref_npz)

    loaded = {}
    for i, native in enumerate(sample):
        group, stripped, _ = where[native]
        if group not in loaded:
            loaded[group] = load_group(str(out), group)
        got = np.array(loaded[group][stripped])
        want = ref[f"k{i}"]
        assert got.dtype == np.float32 and want.dtype == np.float32, native
        assert got.shape == want.shape, native
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), native


def test_converted_groups_use_stripped_keys_and_bare_parameters_use_value(converted):
    out, manifest, _ = converted
    w = load_group(str(out), "folding_trunk")
    assert "blocks.0.tri_mul_out._engine.proj_bundle.weight" in w
    assert not any(k.startswith("folding_trunk.") for k in w)
    assert w["blocks.0.tri_mul_out._engine.proj_bundle.weight"].shape == (1024, 256)
    bare = load_group(str(out), "parcae_log_a")
    assert list(bare) == ["value"]
    assert bare["value"].shape == (256,)
    head = load_group(str(out), "distogram_head")
    assert sorted(head) == ["bias", "weight"]
    assert head["weight"].shape == (64, 256)


def test_config_json_carries_the_checkpoints_resolved_values(converted):
    out, manifest, _ = converted
    cfg = load_config(str(out))
    assert isinstance(cfg, ModelConfig)
    assert (cfg.d_single, cfg.d_pair, cfg.d_single_inputs) == (384, 256, 451)
    assert (cfg.n_trunk_layers, cfg.n_coda_layers, cfg.n_lm_encoder_layers) == (48, 2, 4)
    assert (cfg.num_loops, cfg.n_sampling_steps) == (20, 14)
    assert cfg.msa_encoder.max_depth == 1024 and cfg.msa_encoder.column_mask_rate == 0.1
    assert (cfg.msa_encoder.num_hidden_layers, cfg.msa_encoder.num_attention_heads) == (4, 8)
    assert cfg.msa_encoder.head_width == 16  # config.json says 16; the dataclass default (32) is not used
    assert cfg.sliding_window == 128
    assert cfg.atom_encoder.token_hidden_size == 768  # wins over output_dim (384)
    assert cfg.structure_head.diffusion_module.token_num_blocks == 12
    assert cfg.structure_head.inference_p == 7.0 and cfg.structure_head.inference_s_max == 160.0
    assert cfg.structure_head.noise_scale == 1.003 and cfg.structure_head.step_scale == 1.5
    assert cfg.confidence_head.num_hidden_layers == 4 and cfg.confidence_head.distogram_bins == 39
    # Dimensions agree with the tensors that were written.
    assert load_group(str(out), "z_init_1")["weight"].shape == (cfg.d_pair, cfg.d_single_inputs)
    blocks = {k.split(".")[1] for k in load_group(str(out), "folding_trunk")}
    assert len(blocks) == cfg.n_trunk_layers
