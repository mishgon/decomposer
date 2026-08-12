"""Liger setup for Gemma-4 conditional-generation checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def liger_patch_strategy(model_type: str, text_model_type: str | None) -> str:
    """Return the patch path needed by a Transformers model configuration."""
    if model_type != "gemma4":
        return "transformers_instance"
    if text_model_type != "gemma4_text":
        raise ValueError(
            "Liger was requested for Gemma-4, but its nested text config has "
            f"model_type={text_model_type!r}; expected 'gemma4_text'."
        )
    return "gemma4_multimodal_preload"


def liger_compatible_loss_type(enabled: bool, loss_type: str | None) -> str | None:
    """Use Liger's fused CE instead of TRL's incompatible chunked NLL."""
    if enabled and loss_type == "chunked_nll":
        return None
    return loss_type


def configure_liger_for_model(
    model_name_or_path: str,
    *,
    revision: str,
    trust_remote_code: bool,
    enabled: bool,
    kernel_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply Gemma-4's outer-model patch before it is instantiated.

    Liger 0.8.1 supports the outer ``gemma4`` conditional-generation class and
    its fused linear cross-entropy forward. Preloading the class patch ensures
    TRL never constructs the original logits-materializing forward; the later
    Transformers instance hook reapplies the same configuration and preserves
    normal Trainer bookkeeping.
    """
    if not enabled:
        return {"enabled": False, "strategy": "disabled"}

    from transformers import AutoConfig

    checkpoint_config = AutoConfig.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    model_type = str(checkpoint_config.model_type)
    text_config = getattr(checkpoint_config, "text_config", None)
    text_model_type = getattr(text_config, "model_type", None)
    strategy = liger_patch_strategy(model_type, text_model_type)
    resolved_kernel_config = dict(kernel_config or {})
    if strategy == "gemma4_multimodal_preload":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma4

        apply_liger_kernel_to_gemma4(**resolved_kernel_config)

    return {
        "enabled": True,
        "strategy": strategy,
        "model_type": model_type,
        "text_model_type": text_model_type,
        "kernel_config": resolved_kernel_config,
    }
