from __future__ import annotations

import pytest

from gyms.gaia2.experiments import GEMMA4_26B_A4B_BASE, GEMMA4_E4B_BASE


@pytest.mark.integration
def test_pinned_gemma_templates_replay_reasoning_before_a_later_user_turn() -> None:
    transformers = pytest.importorskip("transformers")
    model_paths = (GEMMA4_E4B_BASE, GEMMA4_26B_A4B_BASE)
    missing = [path for path in model_paths if not path.is_dir()]
    if missing:
        pytest.skip(f"Pinned Gemma tokenizers are not available: {missing}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "Test tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    for model_path in model_paths:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        for reasoning_field in ("reasoning", "reasoning_content"):
            marker = f"PRIVATE_REPLAY_MARKER_{reasoning_field}"
            assistant = {
                "role": "assistant",
                "content": "",
                reasoning_field: marker,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {
                            "name": "test_tool",
                            "arguments": {"value": 1},
                        },
                    }
                ],
            }
            messages = [
                {"role": "user", "content": "First request"},
                assistant,
                {
                    "role": "tool",
                    "name": "test_tool",
                    "tool_call_id": "call_1",
                    "content": "done",
                },
                {"role": "user", "content": "Second request"},
            ]

            dropped = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            preserved = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                preserve_thinking=True,
            )

            assert marker not in dropped
            assert marker in preserved
            assert f"<|channel>thought\n{marker}\n<channel|>" in preserved
