"""Shared, model-facing preprocessing for Decomposer SFT examples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

JsonObject = dict[str, Any]
# The student is trained non-thinking: the teacher's reasoning stays in the
# records as metadata and is never rendered.
_REASONING_KEYS = frozenset({"teacher_reasoning", "reasoning", "reasoning_content"})


def clean_message(message: Mapping[str, Any]) -> JsonObject:
    return {
        key: value
        for key, value in message.items()
        if value is not None and key not in _REASONING_KEYS
    }


def configure_example(example: Mapping[str, Any]) -> JsonObject:
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Prepared example has no messages list.")
    return {"messages": [clean_message(message) for message in messages]}


def tokenize_example(
    example: Mapping[str, Any],
    *,
    tokenizer: Any,
    training_template: str,
) -> JsonObject:
    """Tokenize once for training: TRL trains on these IDs and assistant mask."""
    encoded = tokenizer.apply_chat_template(
        example["messages"],
        tools=example.get("tools"),
        chat_template=training_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        raise RuntimeError("The training template did not produce an assistant mask.")
    supervised_tokens = sum(assistant_mask)
    if supervised_tokens <= 0:
        raise RuntimeError(
            f"Prepared example {example.get('id')} has no supervised tokens."
        )
    return {
        "input_ids": encoded["input_ids"],
        "assistant_masks": assistant_mask,
        "_token_length": len(encoded["input_ids"]),
        "_supervised_tokens": supervised_tokens,
    }
