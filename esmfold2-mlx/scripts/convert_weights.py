"""Convert the trunk-only ESMFold2 checkpoint into per-group fp32 MLX weight files.

Runs in the esm env (torch and esm are imported here and nowhere in the MLX package).

    PYTHONPATH=. python scripts/convert_weights.py [--src DIR] [--out DIR]

The source is HuggingFace-layout; ``esm...hf_checkpoint.hf_state_dict_to_native`` maps it onto the
native layout (renames, packs q/k/v and k/v projections, splits a fused gate/up projection). The
native tensors are then written one file per top-level module or parameter name: the leading
``<group>.`` is stripped from the keys inside a group file, and a bare top-level parameter (for
example ``parcae_log_a``) is stored in its own file under the single key ``value``.

``manifest.json`` records, for every source key, the native keys it produced, the group file and the
stripped keys, so any tensor can be traced and compared bitwise to the esm package's own mapping.
"""

import torch  # first: see the ESMFold2Score note about importing numpy-side modules before torch

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path.home() / ".cache" / "huggingface" / "esmfold2-trunk" / "ESMFold2"
DEFAULT_OUT = REPO / "weights"

_CONFIG_BOILERPLATE = {
    "architectures", "output_hidden_states", "return_dict", "dtype", "chunk_size_feed_forward",
    "is_encoder_decoder", "id2label", "label2id", "problem_type", "_name_or_path", "model_type",
    "output_attentions", "transformers_version", "auto_map",
}


def _strip_config(node):
    if isinstance(node, dict):
        return {k: _strip_config(v) for k, v in node.items() if k not in _CONFIG_BOILERPLATE}
    return node


def _provenance(source_keys, to_native_key) -> dict:
    """source key -> (native keys, transform), mirroring hf_state_dict_to_native's three cases."""
    by_native = defaultdict(list)
    for src in source_keys:
        by_native[to_native_key(src)].append(src)
    out = {}
    for src in source_keys:
        native = to_native_key(src)
        if len(by_native[native]) > 1:
            out[src] = ([native], "pack")
        elif native.endswith(".a_proj.weight"):
            out[src] = ([native, native.replace(".a_proj.", ".b_proj.")], "split")
        else:
            out[src] = ([native], "rename")
    return out


def convert(src_dir: str, out_dir: str) -> dict:
    from esm.models.esmfold2.hf_checkpoint import hf_state_dict_to_native, is_hf_layout, to_native_key

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    raw = load_file(str(src_dir / "model.safetensors"))
    if any(k.startswith("esmc.") for k in raw):
        raise SystemExit("the source still bundles esmc.* tensors; point --src at the trunk-only copy")
    if not is_hf_layout(raw):
        raise SystemExit("the source is not in the HuggingFace layout this converter expects")
    bad = {k: t.dtype for k, t in raw.items() if t.dtype != torch.float32}
    if bad:
        raise SystemExit(f"fp32 only: {len(bad)} source tensors are not float32, e.g. {next(iter(bad.items()))}")
    source_elements = sum(t.numel() for t in raw.values())

    provenance = _provenance(list(raw), to_native_key)
    native = hf_state_dict_to_native(raw)
    expected = {n for natives, _ in provenance.values() for n in natives}
    if set(native) != expected:
        raise SystemExit(
            f"provenance disagrees with hf_state_dict_to_native: only in provenance "
            f"{sorted(expected - set(native))[:3]}, only in output {sorted(set(native) - expected)[:3]}"
        )
    native_elements = sum(t.numel() for t in native.values())
    if native_elements != source_elements:
        raise SystemExit(f"element count changed in mapping: {source_elements} -> {native_elements}")

    def split(native_key):
        group = native_key.split(".")[0]
        bare = "." not in native_key
        return group, bare, "value" if bare else native_key[len(group) + 1:]

    tensors = defaultdict(dict)
    groups = {}
    for native_key, tensor in native.items():
        group, bare, stripped = split(native_key)
        tensors[group][stripped] = np.ascontiguousarray(tensor.numpy())
        groups.setdefault(group, {"file": f"{group}.safetensors", "tensors": 0, "bare": bare})
        groups[group]["tensors"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for group, content in sorted(tensors.items()):
        save_file(content, str(out_dir / f"{group}.safetensors"))

    mapping = {}
    for src, (natives, transform) in provenance.items():
        parts = [split(n) for n in natives]
        mapping[src] = {
            "native": natives, "key": [p[2] for p in parts], "file": groups[parts[0][0]]["file"],
            "group": parts[0][0], "transform": transform,
        }
    packs = defaultdict(list)
    for src, entry in mapping.items():
        if entry["transform"] == "pack":
            packs[entry["native"][0]].append(src)
    splits = [s for s, e in mapping.items() if e["transform"] == "split"]
    manifest = {
        "source": str(src_dir),
        "source_tensors": len(raw), "source_elements": source_elements,
        "native_tensors": len(native), "native_elements": native_elements,
        "converted_tensors": sum(g["tensors"] for g in groups.values()), "converted_elements": native_elements,
        "tensor_count_changes": {
            "packed_native_tensors": len(packs),
            "packed_source_tensors": sum(len(v) for v in packs.values()),
            "split_source_tensors": len(splits),
            "split_native_tensors": 2 * len(splits),
        },
        "groups": groups,
        "map": mapping,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    config = json.loads((src_dir / "config.json").read_text())
    (out_dir / "config.json").write_text(json.dumps(_strip_config(config), indent=1))
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    manifest = convert(args.src, args.out)
    print(f"source tensors {manifest['source_tensors']} -> native tensors {manifest['native_tensors']} "
          f"({manifest['source_elements']} elements, unchanged); packed "
          f"{manifest['tensor_count_changes']['packed_source_tensors']} source tensors into "
          f"{manifest['tensor_count_changes']['packed_native_tensors']}, split "
          f"{manifest['tensor_count_changes']['split_source_tensors']} into "
          f"{manifest['tensor_count_changes']['split_native_tensors']}")
    print("groups (file stem, tensors):")
    for group, info in sorted(manifest["groups"].items()):
        print(f"  {group:28s} {info['tensors']:5d}{'  (bare parameter, key: value)' if info['bare'] else ''}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
