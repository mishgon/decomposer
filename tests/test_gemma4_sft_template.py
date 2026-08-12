from __future__ import annotations

import pytest

from training.sft.gemma4_template import (
    UnsupportedGemma4TemplateError,
    build_gemma4_training_template,
)


@pytest.mark.integration
@pytest.mark.parametrize("include_reasoning", [False, True])
def test_gemma4_training_template_preserves_render_and_masks_assistant(
    include_reasoning: bool,
) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    canonical_template = tokenizer.chat_template
    training_template = build_gemma4_training_template(canonical_template)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "spawn_subagent",
                "description": "Spawn a subagent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Subtask."}
                    },
                    "required": ["prompt"],
                },
            },
        }
    ]
    assistant_call = {
        "role": "assistant",
        "content": "ASSISTANT_NARRATION",
        "tool_calls": [
            {
                "type": "function",
                "id": "call-1",
                "function": {
                    "name": "spawn_subagent",
                    "arguments": {"prompt": "ASSISTANT_TOOL_ARGUMENT"},
                },
            }
        ],
    }
    if include_reasoning:
        assistant_call["reasoning"] = "ASSISTANT_PRIVATE_REASONING"
    messages = [
        {"role": "system", "content": "SYSTEM_SECRET"},
        {"role": "user", "content": "USER_SECRET"},
        assistant_call,
        {
            "role": "tool",
            "name": "spawn_subagent",
            "tool_call_id": "call-1",
            "content": "TOOL_REPORT_SECRET",
        },
        {"role": "assistant", "content": "ASSISTANT_FINAL", "tool_calls": []},
    ]
    template_kwargs = {
        "enable_thinking": include_reasoning,
        "preserve_thinking": include_reasoning,
    }

    canonical = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=canonical_template,
        tokenize=False,
        **template_kwargs,
    )
    training = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=training_template,
        tokenize=False,
        **template_kwargs,
    )
    assert training == canonical

    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=training_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **template_kwargs,
    )
    mask = encoded["assistant_masks"]
    supervised = tokenizer.decode(
        [token for token, is_assistant in zip(encoded["input_ids"], mask) if is_assistant]
    )
    assert "ASSISTANT_TOOL_ARGUMENT" in supervised
    assert "ASSISTANT_NARRATION" in supervised
    assert "ASSISTANT_FINAL" in supervised
    assert "SYSTEM_SECRET" not in supervised
    assert "USER_SECRET" not in supervised
    assert "TOOL_REPORT_SECRET" not in supervised
    assert ("ASSISTANT_PRIVATE_REASONING" in supervised) is include_reasoning


def test_gemma4_training_template_fails_closed_on_unknown_template() -> None:
    with pytest.raises(UnsupportedGemma4TemplateError, match="reasoning output"):
        build_gemma4_training_template("{{ bos_token }}")
