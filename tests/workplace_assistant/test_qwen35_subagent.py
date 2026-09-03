from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain.agents.middleware import ModelCallLimitMiddleware

sys.path.insert(0, str(Path(__file__).parents[2] / "external" / "Gym"))

from gyms.workplace_assistant.subagents import graph


def test_subagent_model_urls_can_be_routed_by_runtime_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        graph.MODEL_BASE_URLS_ENV,
        json.dumps(
            {
                "Qwen/Qwen3.5-4B": "http://127.0.0.1:20025/v1",
                "google/gemma-4-E4B-it": "http://127.0.0.1:32021/v1",
            }
        ),
    )
    assert graph._model_base_url("Qwen/Qwen3.5-4B", 8025) == (
        "http://127.0.0.1:20025/v1"
    )
    assert graph._model_base_url("google/gemma-4-E4B-it", 8021) == (
        "http://127.0.0.1:32021/v1"
    )
    assert graph._model_base_url("google/gemma-4-E2B-it", 8020) == (
        "http://127.0.0.1:8020/v1"
    )


def test_repository_langgraph_registers_every_workplace_subagent() -> None:
    path = Path(graph.__file__).with_name("langgraph.json")
    registered = json.loads(path.read_text())["graphs"]
    expected = {
        "qwen35_4b_non_thinking",
        "gemma_4_2b_thinking",
        "gemma_4_2b_non_thinking",
        "gemma_4_4b_thinking",
        "gemma_4_4b_non_thinking",
        "gemma_4_12b_thinking",
        "gemma_4_12b_non_thinking",
        "gemma_4_26b_a4b_thinking",
        "gemma_4_26b_a4b_non_thinking",
    }
    assert set(registered) == expected
    assert all(callable(getattr(graph, name)) for name in expected)


def test_qwen35_worker_uses_official_non_thinking_general_sampling(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    agent_kwargs: dict[str, Any] = {}

    def fake_chat_vllm(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(graph, "ChatVLLM", fake_chat_vllm)
    monkeypatch.setattr(graph, "SUBAGENT_MAX_COMPLETION_TOKENS", 32768)
    monkeypatch.setenv(
        graph.MODEL_BASE_URLS_ENV,
        '{"Qwen/Qwen3.5-4B":"http://127.0.0.1:20025/v1"}',
    )

    def fake_create_agent(**kwargs: Any) -> object:
        agent_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(graph, "create_agent", fake_create_agent)

    graph.qwen35_4b_non_thinking()

    assert captured["temperature"] == 0.7
    assert captured["base_url"] == "http://127.0.0.1:20025/v1"
    assert captured["top_p"] == 0.8
    assert captured["presence_penalty"] == 1.5
    assert captured["max_completion_tokens"] == 32768
    assert captured["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    limiter = next(
        item
        for item in agent_kwargs["middleware"]
        if isinstance(item, ModelCallLimitMiddleware)
    )
    assert limiter.run_limit == 100
    assert limiter.exit_behavior == "error"


def test_gemma_worker_uses_official_thinking_sampling(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    agent_kwargs: dict[str, Any] = {}

    monkeypatch.setattr(
        graph,
        "ChatVLLM",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(graph, "SUBAGENT_MAX_COMPLETION_TOKENS", 32768)
    monkeypatch.setenv(
        graph.MODEL_BASE_URLS_ENV,
        '{"google/gemma-4-E4B-it":"http://127.0.0.1:32021/v1"}',
    )
    monkeypatch.setattr(
        graph,
        "create_agent",
        lambda **kwargs: agent_kwargs.update(kwargs) or object(),
    )

    graph.gemma_4_4b_thinking()

    assert captured["temperature"] == 1.0
    assert captured["base_url"] == "http://127.0.0.1:32021/v1"
    assert captured["top_p"] == 0.95
    assert captured["max_completion_tokens"] == 32768
    assert captured["preserve_reasoning"] is True
    assert captured["extra_body"] == {
        "top_k": 64,
        "include_reasoning": True,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    }
    limiter = next(
        item
        for item in agent_kwargs["middleware"]
        if isinstance(item, ModelCallLimitMiddleware)
    )
    assert limiter.run_limit == 100
    assert limiter.exit_behavior == "error"
