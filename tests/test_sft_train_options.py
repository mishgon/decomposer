from __future__ import annotations

import json
import os

import pytest
import torch

# Training runs FLA on its Triton kernels; its TileLang backend is not the validated path.
os.environ.setdefault("FLA_TILELANG", "0")

from sft.train import (
    _bf16_config,
    _enable_qwen35_native_gva,
    _freeze_modules,
    _rewrite_weights_in_bf16,
    _step_timing_summary,
)


def _tiny_text_config(**overrides):
    from transformers import Qwen3_5TextConfig

    values = {
        "vocab_size": 64,
        "hidden_size": 128,
        "intermediate_size": 192,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
    }
    return Qwen3_5TextConfig(**{**values, **overrides})


def _tiny_multimodal_config():
    from transformers import Qwen3_5Config

    return Qwen3_5Config(
        text_config=_tiny_text_config().to_dict(),
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 128},
    )


def test_step_timing_summary_skips_warmup_and_measures_imbalance() -> None:
    summary = _step_timing_summary(
        [30.0, 5.0, 5.0, 2.0, 2.0],
        [[1, 1], [1, 1], [1, 1], [100, 300], [200, 200]],
        skip=3,
    )
    assert summary["steady_tokens_per_second"] == 800 / 4.0
    # Largest-rank tokens (300 + 200) over mean-rank tokens ((400 + 400) / 2).
    assert summary["rank_imbalance_ratio"] == 500 / 400
    assert summary["step_seconds_p50"] == 2.0
    assert _step_timing_summary([1.0], [[5, 5]], skip=3) == {"steps": 1, "skipped": 3}


def test_freeze_modules_freezes_only_the_named_subtree() -> None:
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration(_tiny_multimodal_config())
    visual = sum(p.numel() for n, p in model.named_parameters() if n.startswith("model.visual."))

    assert _freeze_modules(model, ["model.visual"]) == visual
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is not name.startswith("model.visual.")
    with pytest.raises(ValueError, match="matched no parameters"):
        _freeze_modules(model, ["model.vis"])


def test_rewrite_weights_in_bf16_casts_every_shard(tmp_path) -> None:
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(_tiny_multimodal_config()).save_pretrained(
        tmp_path, max_shard_size="200KB"
    )
    shards = sorted(tmp_path.glob("*.safetensors"))
    assert len(shards) > 1

    def load():
        tensors = {}
        for shard in shards:
            tensors.update(load_file(shard))
        return tensors

    original = load()
    assert {tensor.dtype for tensor in original.values()} == {torch.float32}
    assert _rewrite_weights_in_bf16(tmp_path) == len(original)
    rewritten = load()
    assert rewritten.keys() == original.keys()
    for name, tensor in original.items():
        torch.testing.assert_close(rewritten[name], tensor.to(torch.bfloat16), rtol=0, atol=0)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == sum(tensor.numel() * 2 for tensor in rewritten.values())


def test_bf16_config_records_bf16_in_every_sub_config(tmp_path) -> None:
    config = _tiny_multimodal_config()
    config.dtype = torch.float32  # as an fp32-weight run loads it
    for key in config.sub_configs:
        getattr(config, key).dtype = torch.float32

    _bf16_config(config).save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["dtype"] == "bfloat16"
    assert saved["text_config"]["dtype"] == saved["vision_config"]["dtype"] == "bfloat16"
    assert config.dtype == torch.float32  # the training model's config is left alone


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA kernels need CUDA")
def test_native_gva_matches_repeated_key_heads() -> None:
    pytest.importorskip("fla")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    torch.manual_seed(0)
    layer = Qwen3_5GatedDeltaNet(_tiny_text_config(), layer_idx=0).cuda().to(torch.bfloat16)
    hidden = torch.randn(2, 96, 128, device="cuda", dtype=torch.bfloat16)

    def run():
        inputs = hidden.clone().requires_grad_(True)
        output = layer(inputs)
        output.float().square().mean().backward()
        return output.detach(), inputs.grad.detach()

    repeated_output, repeated_grad = run()
    assert _enable_qwen35_native_gva(layer) == 1
    assert layer.num_k_heads == layer.num_v_heads
    native_output, native_grad = run()
    torch.testing.assert_close(native_output, repeated_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(native_grad, repeated_grad, atol=2e-2, rtol=2e-2)
