from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[2] / "external" / "Gym"))

from gyms.workplace_assistant.subagents import graph


def test_qwen35_worker_uses_official_non_thinking_general_sampling(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_chat_vllm(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(graph, "ChatVLLM", fake_chat_vllm)
    monkeypatch.setattr(graph, "create_agent", lambda **_: object())

    graph.qwen35_4b_non_thinking()

    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.8
    assert captured["presence_penalty"] == 1.5
    assert captured["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
