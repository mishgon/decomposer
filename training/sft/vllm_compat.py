"""Build a zero-copy vLLM compatibility view of a Gemma-4 SFT export.

Transformers omits K/V projection modules and ``k_norm`` parameters for the
final KV-sharing layers because those layers reuse earlier KV states. vLLM
0.24 understands the shared KV cache, but its Gemma-4 loader still requires an
unused ``k_norm.weight`` for every layer. This module adds identity weights for
only those missing norms while symlinking the original model shard.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_SOURCE_SHARD = "model-00001-of-00002.safetensors"
_COMPAT_SHARD = "model-00002-of-00002.safetensors"
_INDEX = "model.safetensors.index.json"
_MANIFEST = "vllm_compat_manifest.json"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_key_for_layer(
    keys: set[str], layer_index: int, suffix: str
) -> str:
    needle = f".layers.{layer_index}.self_attn.{suffix}"
    matches = [key for key in keys if key.endswith(needle)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source key ending in {needle!r}, found {matches}"
        )
    return matches[0]


def _compatibility_tensors(
    source_model: Path,
    text_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    num_layers = int(text_config["num_hidden_layers"])
    num_shared = int(text_config.get("num_kv_shared_layers", 0))
    first_shared = num_layers - num_shared
    layer_types = list(text_config["layer_types"])
    if num_shared < 1 or first_shared < 1 or len(layer_types) != num_layers:
        raise ValueError(
            "Gemma-4 vLLM compatibility requires a valid positive "
            "text_config.num_kv_shared_layers"
        )

    with safe_open(source_model, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        output: dict[str, torch.Tensor] = {}
        previous_layer_types = layer_types[:first_shared]
        for layer_index in range(first_shared, num_layers):
            layer_type = layer_types[layer_index]
            try:
                reversed_offset = previous_layer_types[::-1].index(layer_type)
            except ValueError as error:
                raise ValueError(
                    f"No non-shared {layer_type!r} layer precedes layer {layer_index}"
                ) from error
            source_layer = first_shared - 1 - reversed_offset
            source_key = _source_key_for_layer(
                keys, source_layer, "k_norm.weight"
            )
            target_key = source_key.replace(
                f".layers.{source_layer}.", f".layers.{layer_index}."
            )
            if target_key in keys:
                raise ValueError(
                    f"Compatibility tensor already exists in source: {target_key}"
                )
            output[target_key] = torch.ones_like(tensors.get_tensor(source_key))
    return output


def create_vllm_compat_export(source: Path, output: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    source_model = source / "model.safetensors"
    config_path = source / "config.json"
    tokenizer_path = source / "tokenizer_config.json"
    for required in (source_model, config_path, tokenizer_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if (source / _INDEX).exists():
        raise ValueError("Only a monolithic source model.safetensors is supported")

    if output.exists():
        manifest_path = output / _MANIFEST
        if not manifest_path.is_file():
            raise FileExistsError(f"Incomplete compatibility export: {output}")
        manifest = json.loads(manifest_path.read_text())
        if Path(manifest["source"]).resolve() != source:
            raise FileExistsError(
                f"Compatibility export points at another source: {output}"
            )
        return manifest

    config = json.loads(config_path.read_text())
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"Gemma-4 text_config is missing from {config_path}")
    compat_tensors = _compatibility_tensors(source_model, text_config)

    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    try:
        for path in source.iterdir():
            if path.name == "model.safetensors" or path.name == _INDEX:
                continue
            os.symlink(os.path.relpath(path, temporary), temporary / path.name)

        os.symlink(
            os.path.relpath(source_model, temporary), temporary / _SOURCE_SHARD
        )
        save_file(
            compat_tensors,
            temporary / _COMPAT_SHARD,
            metadata={"format": "pt"},
        )

        with safe_open(source_model, framework="pt", device="cpu") as tensors:
            weight_map = {key: _SOURCE_SHARD for key in tensors.keys()}
        weight_map.update({key: _COMPAT_SHARD for key in compat_tensors})
        _write_json(
            temporary / _INDEX,
            {
                "metadata": {
                    "total_size": source_model.stat().st_size
                    + (temporary / _COMPAT_SHARD).stat().st_size
                },
                "weight_map": dict(sorted(weight_map.items())),
            },
        )
        manifest = {
            "schema_version": 1,
            "source": str(source),
            "source_model": str(source_model),
            "source_model_size": source_model.stat().st_size,
            "compatibility_tensors": sorted(compat_tensors),
            "compatibility_tensor_count": len(compat_tensors),
        }
        _write_json(temporary / _MANIFEST, manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = create_vllm_compat_export(args.source, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
