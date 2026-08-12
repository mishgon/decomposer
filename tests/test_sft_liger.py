from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from training.sft.liger import (
    configure_liger_for_model,
    liger_compatible_loss_type,
    liger_patch_strategy,
)


def test_gemma4_uses_multimodal_preload_patch() -> None:
    assert (
        liger_patch_strategy("gemma4", "gemma4_text")
        == "gemma4_multimodal_preload"
    )


def test_gemma4_liger_fails_closed_for_unknown_text_config() -> None:
    with pytest.raises(ValueError, match="expected 'gemma4_text'"):
        liger_patch_strategy("gemma4", "future_gemma4_text")


def test_other_models_keep_transformers_instance_hook() -> None:
    assert liger_patch_strategy("llama", None) == "transformers_instance"


def test_gemma4_preload_passes_fused_ce_only_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, bool]] = []
    fake_transformers = ModuleType("liger_kernel.transformers")
    fake_transformers.apply_liger_kernel_to_gemma4 = lambda **kwargs: calls.append(
        kwargs
    )
    monkeypatch.setitem(
        sys.modules,
        "liger_kernel.transformers",
        fake_transformers,
    )

    from transformers import AutoConfig

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="gemma4",
            text_config=SimpleNamespace(model_type="gemma4_text"),
        ),
    )
    kernel_config = {
        "fused_linear_cross_entropy": True,
        "cross_entropy": False,
        "rms_norm": False,
        "geglu": False,
        "rope": False,
        "layer_norm": False,
    }

    runtime = configure_liger_for_model(
        "google/gemma-4-E4B-it",
        revision="main",
        trust_remote_code=False,
        enabled=True,
        kernel_config=kernel_config,
    )

    assert calls == [kernel_config]
    assert runtime == {
        "enabled": True,
        "strategy": "gemma4_multimodal_preload",
        "model_type": "gemma4",
        "text_model_type": "gemma4_text",
        "kernel_config": kernel_config,
    }


def test_liger_uses_fused_ce_instead_of_chunked_nll() -> None:
    assert liger_compatible_loss_type(True, "chunked_nll") is None
    assert liger_compatible_loss_type(False, "chunked_nll") == "chunked_nll"
    assert liger_compatible_loss_type(True, "custom") == "custom"
