from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import GenerationConfig

from sft.preprocessing import tokenize_example
from sft.qwen35_template import (
    UnsupportedQwen35TemplateError,
    build_qwen35_training_template,
)
from sft.train import (
    _configure_qwen35_generation,
    _require_qwen35,
    _save_final_configuration,
)


@pytest.mark.integration
def test_qwen35_training_template_preserves_render_and_masks_only_assistant() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    canonical_template = tokenizer.chat_template
    training_template = build_qwen35_training_template(canonical_template)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "wait",
                "description": "Wait for reports.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "SYSTEM_SECRET"},
        {"role": "user", "content": "USER_SECRET"},
        {
            "role": "assistant",
            "content": "ASSISTANT_NARRATION",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call-1",
                    "function": {"name": "wait", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "name": "wait",
            "tool_call_id": "call-1",
            "content": "TOOL_REPORT_SECRET",
        },
        # A parallel batch stays one assistant message with several calls.
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": f"call-{index}",
                    "function": {"name": "wait", "arguments": {}},
                }
                for index in (2, 3)
            ],
        },
        {
            "role": "tool",
            "name": "wait",
            "tool_call_id": "call-2",
            "content": "SECOND_REPORT_SECRET",
        },
        {
            "role": "tool",
            "name": "wait",
            "tool_call_id": "call-3",
            "content": "THIRD_REPORT_SECRET",
        },
        {"role": "assistant", "content": "ASSISTANT_FINAL", "tool_calls": []},
    ]
    kwargs = {"enable_thinking": False, "preserve_thinking": False}
    canonical = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=canonical_template,
        tokenize=False,
        **kwargs,
    )
    training = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=training_template,
        tokenize=False,
        **kwargs,
    )
    assert training == canonical

    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=training_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    supervised = tokenizer.decode(
        [
            token
            for token, is_assistant in zip(
                encoded["input_ids"], encoded["assistant_masks"]
            )
            if is_assistant
        ]
    )
    assert "ASSISTANT_NARRATION" in supervised
    assert "ASSISTANT_FINAL" in supervised
    assert supervised.count("<tool_call>") == 3
    assert "SYSTEM_SECRET" not in supervised
    assert "USER_SECRET" not in supervised
    assert "REPORT_SECRET" not in supervised

    tokenized = tokenize_example(
        {"messages": messages, "tools": tools},
        tokenizer=tokenizer,
        training_template=training_template,
    )
    assert tokenized["input_ids"] == encoded["input_ids"]
    assert tokenized["assistant_masks"] == encoded["assistant_masks"]
    assert tokenized["_token_length"] == len(encoded["input_ids"])
    assert tokenized["_supervised_tokens"] == sum(encoded["assistant_masks"])


def test_qwen35_template_fails_closed_on_unknown_template() -> None:
    with pytest.raises(UnsupportedQwen35TemplateError, match="assistant branch"):
        build_qwen35_training_template("{{ bos_token }}")


def test_training_accepts_only_qwen35() -> None:
    _require_qwen35(
        SimpleNamespace(
            model_type="qwen3_5",
            text_config=SimpleNamespace(model_type="qwen3_5_text"),
        )
    )


@pytest.mark.parametrize(
    "config",
    [
        # A text-only checkpoint has no `language_model` wrapper for the LoRA targets.
        SimpleNamespace(model_type="qwen3_5_text"),
        SimpleNamespace(
            model_type="gemma4", text_config=SimpleNamespace(model_type="gemma4_text")
        ),
    ],
)
def test_training_refuses_other_models(config: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="Unsupported Decomposer SFT model_type"):
        _require_qwen35(config)


def test_qwen35_generation_preserves_tokenizer_eos_and_model_eod() -> None:
    tokenizer = SimpleNamespace(
        bos_token_id=None,
        eos_token_id=248046,
        pad_token_id=248044,
    )
    generation = GenerationConfig(eos_token_id=248044, pad_token_id=None)
    runtime = _configure_qwen35_generation(generation, tokenizer=tokenizer)
    assert runtime == {
        "bos_token_id": None,
        "eos_token_id": [248046, 248044],
        "pad_token_id": 248044,
    }


def test_final_configuration_saves_the_qwen35_generation_config(tmp_path: Path) -> None:
    tokenizer = SimpleNamespace(
        bos_token_id=None,
        eos_token_id=248046,
        pad_token_id=248044,
        save_pretrained=lambda path: (Path(path) / "tokenizer_config.json").write_text("{}"),
    )
    model_config = SimpleNamespace(
        save_pretrained=lambda path: (Path(path) / "config.json").write_text("{}")
    )
    generation = GenerationConfig(eos_token_id=248044, pad_token_id=None)
    _configure_qwen35_generation(generation, tokenizer=tokenizer)
    _save_final_configuration(
        tmp_path,
        model_config=model_config,
        tokenizer=tokenizer,
        generation_config=generation,
    )
    saved = GenerationConfig.from_pretrained(tmp_path)
    assert saved.eos_token_id == [248046, 248044]
    assert saved.pad_token_id == 248044
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "tokenizer_config.json").is_file()
