from __future__ import annotations

import pytest

pytest.importorskip("langchain_openai")

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from gyms.gaia2 import service
from gyms.gaia2.prompts import GAIA2_AMBIGUITY_MANAGER_ADDENDUM


class FakeGraph:
    def __init__(self, content=None):
        self.calls = []
        self.content = content

    async def ainvoke(self, value, config, context):
        self.calls.append((value, config, context.copy()))
        content = self.content if self.content is not None else f"turn-{len(self.calls)}"
        return {"messages": [AIMessage(content=content)]}


def test_episode_persists_thread_and_forwards_runtime_context(monkeypatch):
    graph = FakeGraph()
    monkeypatch.setattr(service, "_model_from_config", lambda value: object())
    monkeypatch.setattr(service, "create_decomposer_agent", lambda **kwargs: graph)
    app = service.create_app(
        {
            "manager": {"model": "fake"},
            "subagent_types": [
                {
                    "subagent_type_id": "worker",
                    "description": "worker",
                    "assistant_id": "worker",
                    "url": "http://127.0.0.1:1",
                }
            ],
        }
    )
    context = {
        "tool_schemas": [],
        "broker_url": "http://broker/session",
        "session_token": "secret",
        "policy": "shared_serialized",
        "scenario_id": "scenario",
        "run_number": 1,
        "notification_cursor": 0,
    }
    with TestClient(app) as client:
        episode_id = client.post("/v1/episodes", json={"context": context}).json()[
            "episode_id"
        ]
        for turn in (1, 2):
            response = client.post(
                f"/v1/episodes/{episode_id}/turn",
                json={
                    "notifications": [
                        {
                            "type": "USER_MESSAGE",
                            "message": f"message-{turn}",
                            "simulated_timestamp": "2026-01-01T00:00:00+00:00",
                        }
                    ],
                    "notification_cursor": turn,
                    "turn_number": turn,
                },
            )
            assert response.status_code == 200
            assert response.json()["final_text"] == f"turn-{turn}"
            assert (
                response.json()["trace"]["runtime_context"]["session_token"]
                == "<redacted>"
            )

    assert len(graph.calls) == 2
    assert (
        graph.calls[0][1]["configurable"]["thread_id"]
        == graph.calls[1][1]["configurable"]["thread_id"]
    )
    assert graph.calls[0][2]["scenario_id"] == "scenario"
    assert graph.calls[1][2]["notification_cursor"] == 2


def test_uncollected_subagent_is_reported_as_outstanding():
    _, outstanding = service._subagent_summary(
        {
            "subagent_runs": {
                "running": {"status": "running", "report": None},
                "uncollected": {"status": "success", "report": None},
                "collected": {"status": "success", "report": {"content": "ok"}},
            }
        }
    )
    assert outstanding == ["running", "uncollected"]


def test_openrouter_content_blocks_return_only_visible_text(monkeypatch):
    graph = FakeGraph(
        [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "private"}],
            },
            {"type": "text", "text": "visible answer"},
        ]
    )
    monkeypatch.setattr(service, "_model_from_config", lambda value: object())
    monkeypatch.setattr(service, "create_decomposer_agent", lambda **kwargs: graph)
    app = service.create_app(
        {
            "manager": {"model": "fake"},
            "decomposer_system_prompt_profile": "teacher",
            "subagent_types": [{"subagent_type_id": "worker"}],
        }
    )
    with TestClient(app) as client:
        episode_id = client.post(
            "/v1/episodes", json={"context": _context()}
        ).json()["episode_id"]
        response = client.post(
            f"/v1/episodes/{episode_id}/turn", json=_turn()
        )

    assert response.status_code == 200
    assert response.json()["final_text"] == "visible answer"
    raw_content = response.json()["trace"]["manager_messages"][-1]["data"]["content"]
    assert raw_content[0]["type"] == "reasoning"


def test_reasoning_only_response_is_not_a_final_answer(monkeypatch):
    graph = FakeGraph([{"type": "reasoning", "content": []}])
    monkeypatch.setattr(service, "_model_from_config", lambda value: object())
    monkeypatch.setattr(service, "create_decomposer_agent", lambda **kwargs: graph)
    app = service.create_app(
        {
            "manager": {"model": "fake"},
            "subagent_types": [{"subagent_type_id": "worker"}],
        }
    )
    with TestClient(app) as client:
        episode_id = client.post(
            "/v1/episodes", json={"context": _context()}
        ).json()["episode_id"]
        response = client.post(
            f"/v1/episodes/{episode_id}/turn", json=_turn()
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "Decomposer produced no visible final text"


def test_prompt_profile_is_forwarded_to_decomposer(monkeypatch):
    captured = {}
    monkeypatch.setattr(service, "_model_from_config", lambda value: object())

    def fake_create_decomposer_agent(**kwargs):
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(service, "create_decomposer_agent", fake_create_decomposer_agent)
    service.create_app(
        {
            "manager": {"model": "fake"},
            "decomposer_system_prompt_profile": "teacher",
            "subagent_types": [{"subagent_type_id": "worker"}],
        }
    )

    assert captured["decomposer_system_prompt"] == service.DECOMPOSER_TEACHER_SYSTEM_PROMPT


def test_prompt_addendum_is_appended_only_when_configured(monkeypatch):
    captured = {}
    monkeypatch.setattr(service, "_model_from_config", lambda value: object())

    def fake_create_decomposer_agent(**kwargs):
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(service, "create_decomposer_agent", fake_create_decomposer_agent)
    service.create_app(
        {
            "manager": {"model": "fake"},
            "decomposer_system_prompt_profile": "teacher",
            "decomposer_system_prompt_addendum_profile": "gaia2-ambiguity",
            "subagent_types": [{"subagent_type_id": "worker"}],
        }
    )

    assert captured["decomposer_system_prompt"] == (
        f"{service.DECOMPOSER_TEACHER_SYSTEM_PROMPT}\n\n"
        f"{GAIA2_AMBIGUITY_MANAGER_ADDENDUM}"
    )


def _context():
    return {
        "tool_schemas": [],
        "broker_url": "http://broker/session",
        "session_token": "secret",
        "policy": "shared_serialized",
        "scenario_id": "scenario",
        "run_number": 1,
        "notification_cursor": 0,
    }


def _turn():
    return {
        "notifications": [
            {
                "type": "USER_MESSAGE",
                "message": "message",
                "simulated_timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
        "notification_cursor": 1,
        "turn_number": 1,
    }


def test_manager_model_forwards_non_thinking_sampling(monkeypatch):
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(service, "ChatOpenAI", fake_model)
    service._model_from_config(
        {
            "model": "manager",
            "base_url": "http://127.0.0.1:8020/v1",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_completion_tokens": 4096,
            "extra_body": {
                "top_k": 64,
                "include_reasoning": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
    )

    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    assert captured["max_completion_tokens"] == 4096
    assert captured["model_kwargs"] == {"parallel_tool_calls": False}
    assert captured["extra_body"] == {
        "top_k": 64,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_manager_model_allows_parallel_tool_call_override(monkeypatch):
    captured = {}

    def fake_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(service, "ChatOpenAI", fake_model)
    service._model_from_config(
        {
            "model": "manager",
            "parallel_tool_calls": True,
        }
    )

    assert captured["model_kwargs"] == {"parallel_tool_calls": True}


def test_manager_model_rejects_non_boolean_parallel_tool_calls():
    with pytest.raises(
        ValueError, match="manager.parallel_tool_calls must be a boolean"
    ):
        service._model_from_config(
            {
                "model": "manager",
                "parallel_tool_calls": "false",
            }
        )
