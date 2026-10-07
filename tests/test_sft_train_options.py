from __future__ import annotations

import os

import pytest
import torch

# Training runs FLA on its Triton kernels; its TileLang backend is not the validated path.
os.environ.setdefault("FLA_TILELANG", "0")

from sft.train import _enable_qwen35_native_gva, _freeze_modules, _step_timing_summary


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
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(
            text_config=_tiny_text_config().to_dict(),
            vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 128},
        )
    )
    visual = sum(p.numel() for n, p in model.named_parameters() if n.startswith("model.visual."))

    assert _freeze_modules(model, ["model.visual"]) == visual
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is not name.startswith("model.visual.")
    with pytest.raises(ValueError, match="matched no parameters"):
        _freeze_modules(model, ["model.vis"])


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
