from __future__ import annotations

import json
from pathlib import Path

import pytest

from decomposer.models import create_model
from gyms.workplace_assistant import model_presets

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_non_thinking_manager_body_is_the_preset_with_thinking_switched_off() -> None:
    body = model_presets.manager_responses_body(model_presets.QWEN38_FLASH_NON_THINKING_PRESET)
    preset = create_model("lmrouter/qwen_3_8_flash_next_non_thinking")
    assert body == {
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "presence_penalty": preset.presence_penalty,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
        # The proxy ignores chat_template_kwargs on the Responses API.
        "reasoning": {"effort": "none"},
    }
    assert (body["temperature"], body["top_p"], body["presence_penalty"]) == (0.7, 0.8, 1.5)
    assert model_presets.upstream_model_id(model_presets.QWEN38_FLASH_NON_THINKING_PRESET) == (
        "Qwen/Qwen3.8-Flash-Next-NVFP4"
    )
    assert not model_presets.is_thinking(model_presets.QWEN38_FLASH_NON_THINKING_PRESET)


def test_thinking_manager_body_keeps_the_preset_effort() -> None:
    body = model_presets.manager_responses_body("lmrouter/qwen_3_8_flash_next_low_thinking")
    assert body["include_reasoning"] is True
    assert body["reasoning"] == {"effort": "low"}
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    assert (body["temperature"], body["top_p"], body["presence_penalty"]) == (1.0, 0.95, 0.0)


def test_subagent_preset_record_has_parameters_but_no_endpoint_or_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROXY_MASTER_KEY", "secret-value")
    model_presets._preset.cache_clear()
    record = model_presets.preset_record(model_presets.QWEN35_UNLOOPED_THINKING_PRESET)
    assert record["model"] == "Qwen/Qwen3.5-4B-unlooped"
    assert (record["temperature"], record["top_p"]) == (0.6, 0.95)
    assert record["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
    assert record["preserve_reasoning"] is True
    assert record["max_completion_tokens"] is None
    text = json.dumps(record)
    assert "secret-value" not in text and "http" not in text
    assert model_presets.is_thinking(model_presets.QWEN35_UNLOOPED_THINKING_PRESET)
    model_presets._preset.cache_clear()


def test_only_lmrouter_presets_are_bridged() -> None:
    with pytest.raises(ValueError, match="lmrouter"):
        model_presets.manager_responses_body("openrouter/qwen_3_8_flash_next_low_thinking")


def test_every_subagent_preset_graph_is_registered() -> None:
    graphs = json.loads((REPO_ROOT / "gyms" / "workplace_assistant" / "subagents" / "langgraph.json").read_text())["graphs"]
    for graph in model_presets.SUBAGENT_PRESET_GRAPHS.values():
        assert graphs[graph] == f"./graph.py:{graph}"
