from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import GenerationConfig

from sft.train import _build_lora_config, _export_lora

LORA = {"r": 4, "alpha": 8, "dropout": 0.0}


def _tiny_qwen35():
    """A randomly initialised Qwen3.5 with 3 linear-attention layers, 1 full-attention layer and a vision tower."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
        },
    )
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(config)


def test_build_lora_config_is_none_for_full_sft() -> None:
    assert _build_lora_config(None, training_config={"fsdp": True}, model_type="gemma4") is None


@pytest.mark.parametrize(
    ("lora", "training_config", "model_type", "match"),
    [
        ({"r": 4, "alpha": 8}, {}, "qwen3_5", "exactly r, alpha and dropout"),
        ({**LORA, "rank": 4}, {}, "qwen3_5", "exactly r, alpha and dropout"),
        (LORA, {"fsdp": True}, "qwen3_5", "use DDP"),
        (LORA, {}, "gemma4", "Qwen3.5 only"),
        ({**LORA, "r": 0}, {}, "qwen3_5", "lora.r"),
        ({**LORA, "dropout": 1.0}, {}, "qwen3_5", "lora.dropout"),
    ],
)
def test_build_lora_config_rejects_invalid_sections(lora, training_config, model_type, match) -> None:
    with pytest.raises(ValueError, match=match):
        _build_lora_config(lora, training_config=training_config, model_type=model_type)


def test_lora_adapts_only_language_model_projections() -> None:
    peft = pytest.importorskip("peft")
    config = _build_lora_config(LORA, training_config={"fsdp": False}, model_type="qwen3_5")
    assert (config.r, config.lora_alpha, config.lora_dropout) == (4, 8, 0.0)
    model = peft.get_peft_model(_tiny_qwen35(), config)

    adapted = {
        name.removeprefix("base_model.model.")
        for name, module in model.named_modules()
        if isinstance(module, peft.tuners.lora.LoraLayer)
    }
    layers = "model.language_model.layers"
    expected = {
        f"{layers}.{layer}.linear_attn.{name}"
        for layer in range(3)
        for name in ("in_proj_qkv", "in_proj_z", "out_proj")
    }
    expected |= {f"{layers}.3.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
    expected |= {
        f"{layers}.{layer}.mlp.{name}"
        for layer in range(4)
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    assert adapted == expected
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(".lora_" in name for name in trainable)


class _Tokenizer:
    def save_pretrained(self, path) -> None:
        (Path(path) / "tokenizer_config.json").write_text("{}")


def test_export_lora_saves_adapter_and_merged_weights(tmp_path: Path) -> None:
    peft = pytest.importorskip("peft")
    from safetensors.torch import load_file

    config = _build_lora_config(LORA, training_config={}, model_type="qwen3_5")
    config.init_lora_weights = False  # non-zero B, so merging changes the weights
    model = peft.get_peft_model(_tiny_qwen35(), config)
    target = model.base_model.model.model.language_model.layers[0].mlp.down_proj
    expected = (target.base_layer.weight + target.get_delta_weight("default")).detach().clone()
    lm_head = model.base_model.model.lm_head.weight.detach().clone()

    final_dir = _export_lora(
        model, tmp_path, tokenizer=_Tokenizer(), generation_config=GenerationConfig()
    )

    assert final_dir == tmp_path / "final"
    assert (tmp_path / "final-adapter" / "adapter_config.json").is_file()
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        assert (final_dir / name).is_file()
    weights = {}
    for shard in final_dir.glob("*.safetensors"):
        weights.update(load_file(shard))
    assert not any("lora_" in key for key in weights)
    torch.testing.assert_close(weights["model.language_model.layers.0.mlp.down_proj.weight"], expected)
    torch.testing.assert_close(weights["lm_head.weight"], lm_head)
