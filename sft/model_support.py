"""Model-family-specific chat-template behavior shared by preparation and SFT."""

from __future__ import annotations

from typing import Any

from .gemma4_template import build_gemma4_training_template
from .preprocessing import (
    GEMMA4_PREPARED_TOKENIZATION_PROFILE,
    QWEN35_PREPARED_TOKENIZATION_PROFILE,
)
from .qwen35_template import build_qwen35_training_template


def tokenization_profile_for_model_config(config: Any) -> str:
    model_type = getattr(config, "model_type", None)
    text_config = getattr(config, "text_config", None)
    text_model_type = getattr(text_config, "model_type", None)
    if model_type == "gemma4":
        if text_model_type != "gemma4_text":
            raise ValueError(
                "Gemma-4 text config must use model_type='gemma4_text'; got "
                f"{text_model_type!r}."
            )
        return GEMMA4_PREPARED_TOKENIZATION_PROFILE
    if model_type == "qwen3_5":
        if text_model_type != "qwen3_5_text":
            raise ValueError(
                "Qwen3.5 text config must use model_type='qwen3_5_text'; got "
                f"{text_model_type!r}."
            )
        return QWEN35_PREPARED_TOKENIZATION_PROFILE
    if model_type == "qwen3_5_text":
        return QWEN35_PREPARED_TOKENIZATION_PROFILE
    raise ValueError(f"Unsupported Decomposer SFT model_type: {model_type!r}.")


def build_training_template(profile: str, canonical_template: str) -> str:
    if profile == GEMMA4_PREPARED_TOKENIZATION_PROFILE:
        return build_gemma4_training_template(canonical_template)
    if profile == QWEN35_PREPARED_TOKENIZATION_PROFILE:
        return build_qwen35_training_template(canonical_template)
    raise ValueError(f"Unsupported prepared tokenization profile: {profile}")


def validate_reasoning_policy(profile: str, *, include_reasoning: bool) -> None:
    if profile == QWEN35_PREPARED_TOKENIZATION_PROFILE and include_reasoning:
        raise ValueError(
            "Qwen3.5 Decomposer SFT supports only non-thinking targets; set "
            "data.include_reasoning=false. Teacher reasoning remains available as "
            "unsupervised trace metadata."
        )
