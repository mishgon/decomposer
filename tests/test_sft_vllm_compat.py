from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from training.sft.vllm_compat import create_vllm_compat_export


def test_vllm_compat_adds_identity_norms_for_shared_kv_layers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "final"
    source.mkdir()
    config = {
        "text_config": {
            "num_hidden_layers": 4,
            "num_kv_shared_layers": 2,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        }
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer_config.json").write_text("{}")
    save_file(
        {
            "model.language_model.layers.0.self_attn.k_norm.weight": torch.full(
                (2,), 3.0
            ),
            "model.language_model.layers.1.self_attn.k_norm.weight": torch.full(
                (4,), 4.0
            ),
            "model.language_model.layers.2.self_attn.q_norm.weight": torch.zeros(2),
        },
        source / "model.safetensors",
    )

    output = tmp_path / "final-vllm"
    manifest = create_vllm_compat_export(source, output)

    assert manifest["compatibility_tensor_count"] == 2
    assert (output / "model-00001-of-00002.safetensors").is_symlink()
    index = json.loads((output / "model.safetensors.index.json").read_text())
    layer_2 = "model.language_model.layers.2.self_attn.k_norm.weight"
    layer_3 = "model.language_model.layers.3.self_attn.k_norm.weight"
    assert index["weight_map"][layer_2] == "model-00002-of-00002.safetensors"
    assert index["weight_map"][layer_3] == "model-00002-of-00002.safetensors"

    with safe_open(
        output / "model-00002-of-00002.safetensors",
        framework="pt",
        device="cpu",
    ) as tensors:
        assert torch.equal(tensors.get_tensor(layer_2), torch.ones(2))
        assert torch.equal(tensors.get_tensor(layer_3), torch.ones(4))

    assert create_vllm_compat_export(source, output) == manifest
