from __future__ import annotations

import pytest

pytest.importorskip("langchain_openai")

from fastapi.testclient import TestClient
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import AIMessage

from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT
from gyms.gaia2 import service
from gyms.gaia2.model_overflow import (
    ExactModelCallLimitMiddleware,
    Gaia2ModelOverflowError,
    Gaia2ModelOverflowMiddleware,
)
from gyms.gaia2.prompts import GAIA2_AMBIGUITY_MANAGER_ADDENDUM


class FakeGraph:
    def __init__(self, content=None):
        self.calls = []
        self.content = content

    async def ainvoke(self, value, config, context):
        self.calls.append((value, config, context.copy()))
        content = self.content if self.content is not None else f"turn-{len(self.calls)}"
        return {"messages": [AIMessage(content=content)]}


def test_manager_model_call_limit_is_installed_independently(monkeypatch):
    captured = {}
    graph = FakeGraph()
    monkeypatch.setattr(service, "_model_from_config", lambda value: object())

    def fake_create_decomposer_agent(**kwargs):
        captured.update(kwargs)
        return graph

    monkeypatch.setattr(service, "create_decomposer_agent", fake_create_decomposer_agent)
    service.create_app(
        {
            "manager": {"model": "fake"},
            "manager_max_model_calls": 200,
            "manager_recursion_limit": 1000,
            "subagent_recursion_limit": 1000,
            "subagent_types": [{"subagent_type_id": "worker"}],
        }
    )

    limiter = next(
        item
        for item in captured["middleware"]
        if isinstance(item, ModelCallLimitMiddleware)
    )
    assert isinstance(limiter, ExactModelCallLimitMiddleware)
    assert limiter.run_limit == 200
    assert limiter.exit_behavior == "end"
    assert any(
        isinstance(item, Gaia2ModelOverflowMiddleware)
        and item.actor == "manager"
        for item in captured["middleware"]
    )
    assert captured["subagent_recursion_limit"] == 1000


def test_manager_overflow_is_returned_as_structured_terminal_failure(monkeypatch):
    class OverflowGraph(FakeGraph):
        async def ainvoke(self, value, config, context):
            self.calls.append((value, config, context.copy()))
            raise Gaia2ModelOverflowError(
                actor="manager",
                kind="input_context_overflow",
                detail="maximum context length is 131072 tokens",
                max_completion_tokens=8192,
                max_model_len=131072,
            )

    graph = OverflowGraph()
    monkeypatch.setattr(service, "_model_from_config", lambda value: object())
    monkeypatch.setattr(service, "create_decomposer_agent", lambda **kwargs: graph)
    app = service.create_app(
        {
            "manager": {"model": "fake"},
            "max_model_len": 131072,
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
    assert response.json()["failure"] == {
        "actor": "manager",
        "kind": "input_context_overflow",
        "detail": "maximum context length is 131072 tokens",
        "policy": "fail_actor_v1",
        "max_completion_tokens": 8192,
        "max_model_len": 131072,
    }
    assert "final_text" not in response.json()


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
    summaries, outstanding = service._subagent_summary(
        {
            "subagents": {
                "s1": {"subagent_id": "s1", "subagent_type_id": "worker"},
            },
            "subagent_runs": {
                "running": {"subagent_id": "s1", "status": "running"},
                "uncollected": {
                    "subagent_id": "s1",
                    "status": "responded",
                    "response": "ok",
                },
                "collected": {
                    "subagent_id": "s1",
                    "status": "responded",
                    "response": "ok",
                    "response_sequence_number": 1,
                    "messages": [
                        AIMessage(
                            content="",
                            additional_kwargs={"reasoning_content": "private"},
                        )
                    ],
                },
            }
        }
    )
    assert outstanding == ["running", "uncollected"]
    assert all("messages" not in summary for summary in summaries)
    # The type lives on the persistent subagent, not on the run.
    assert {summary["subagent_type_id"] for summary in summaries} == {"worker"}
    assert summaries[2]["response"] == "ok"


def test_manager_sidecar_message_keeps_structured_reasoning_separate():
    serialized = service._safe_message(
        AIMessage(
            content="visible",
            additional_kwargs={"reasoning_content": "private rationale"},
        )
    )

    assert serialized["data"]["content"] == "visible"
    assert serialized["data"]["additional_kwargs"]["reasoning_content"] == (
        "private rationale"
    )


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
    monkeypatch.setattr(service, "system_prompt_middleware", lambda prompt: ("prompt", prompt))

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

    # The core takes no prompt argument; the profile reaches it as the first middleware.
    assert "decomposer_system_prompt" not in captured
    assert captured["middleware"][0] == ("prompt", DECOMPOSER_SYSTEM_PROMPT)


def test_prompt_addendum_is_appended_only_when_configured(monkeypatch):
    captured = {}
    monkeypatch.setattr(service, "_model_from_config", lambda value: object())
    monkeypatch.setattr(service, "system_prompt_middleware", lambda prompt: ("prompt", prompt))

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

    assert captured["middleware"][0] == (
        "prompt",
        f"{DECOMPOSER_SYSTEM_PROMPT}\n\n{GAIA2_AMBIGUITY_MANAGER_ADDENDUM}",
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

    monkeypatch.setattr(service, "ChatVLLM", fake_model)
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
    assert captured["preserve_reasoning"] is True
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

    monkeypatch.setattr(service, "ChatVLLM", fake_model)
    service._model_from_config(
        {
            "model": "manager",
            "parallel_tool_calls": True,
        }
    )

    assert captured["model_kwargs"] == {"parallel_tool_calls": True}
    assert captured["preserve_reasoning"] is True


def test_responses_api_manager_keeps_native_responses_adapter(monkeypatch):
    captured = {}

    def fake_responses_model(**kwargs):
        captured.update(kwargs)
        return object()

    def unexpected_vllm_model(**kwargs):
        raise AssertionError("Responses API must not use the Chat Completions adapter")

    monkeypatch.setattr(service, "ChatOpenAI", fake_responses_model)
    monkeypatch.setattr(service, "ChatVLLM", unexpected_vllm_model)

    service._model_from_config(
        {
            "model": "remote-manager",
            "use_responses_api": True,
            "reasoning": {"effort": "high"},
        }
    )

    assert captured["use_responses_api"] is True
    assert captured["reasoning"] == {"effort": "high"}
    assert "preserve_reasoning" not in captured


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
