"""Shared, model-facing preprocessing for Gemma-4 SFT examples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

JsonObject = dict[str, Any]
PREPARED_TOKENIZATION_ATTRIBUTE = "prepared_tokenization"
PREPARED_TOKENIZATION_PROFILE = "gemma4_sft_non_thinking"


def clean_message(
    message: Mapping[str, Any], *, include_reasoning: bool
) -> JsonObject:
    clean = {key: value for key, value in message.items() if value is not None}
    teacher_reasoning = clean.pop("teacher_reasoning", None)
    clean.pop("reasoning", None)
    clean.pop("reasoning_content", None)
    if include_reasoning and isinstance(teacher_reasoning, str) and teacher_reasoning:
        clean["reasoning"] = teacher_reasoning
    return clean


def configure_example(example: Mapping[str, Any], include_reasoning: bool) -> JsonObject:
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Prepared example has no messages list.")
    return {
        "messages": [
            clean_message(message, include_reasoning=include_reasoning)
            for message in messages
        ],
        "chat_template_kwargs": {
            "enable_thinking": include_reasoning,
            "preserve_thinking": include_reasoning,
        },
    }


def tokenization_stats(
    example: Mapping[str, Any],
    *,
    tokenizer: Any,
    training_template: str,
) -> JsonObject:
    kwargs = dict(example.get("chat_template_kwargs") or {})
    encoded = tokenizer.apply_chat_template(
        example["messages"],
        tools=example.get("tools"),
        chat_template=training_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        raise RuntimeError(
            "Gemma-4 training template did not produce an assistant mask."
        )
    supervised_tokens = sum(assistant_mask)
    if supervised_tokens <= 0:
        raise RuntimeError(
            f"Prepared example {example.get('id')} has no supervised tokens."
        )
    return {
        "_token_length": len(encoded["input_ids"]),
        "_supervised_tokens": supervised_tokens,
    }
