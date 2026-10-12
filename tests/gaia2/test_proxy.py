from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("are")

from are.simulation.apps.agent_user_interface import AgentUserInterface

from gyms.gaia2 import are as proxy


class Broker:
    base_url = "http://broker/v1"


class Env:
    class Stop:
        def is_set(self):
            return False

    stop_event = Stop()


def test_sidecar_redacts_credentials_but_keeps_token_counts() -> None:
    assert proxy._redact(
        {
            "session_token": "secret-value",
            "api_key": "secret-key",
            "max_completion_tokens": 4096,
            "usage": {"output_tokens": 17},
        }
    ) == {
        "session_token": "<redacted>",
        "api_key": "<redacted>",
        "max_completion_tokens": 4096,
        "usage": {"output_tokens": 17},
    }


def test_stop_cancels_episode_once(monkeypatch):
    calls = []

    def fake_request(method, url, value=None, **kwargs):
        calls.append((method, url))
        if url.endswith("/state"):
            return {"values": {}}
        return {}

    monkeypatch.setattr(proxy, "_request", fake_request)
    agent = proxy.AgentServerAgent(
        Broker(),
        Env(),
        {
            "assistant_id": "decomposer",
            "agent_server_url": "http://manager",
            "gaia2_revision": "gaia",
            "decomposer_revision": "core",
        },
    )
    agent._thread_id = "episode"
    agent.stop()
    agent.stop()
    assert calls == [
        ("GET", "http://manager/threads/episode/state"),
        ("DELETE", "http://manager/threads/episode"),
    ]


class FrozenTurnWorld:
    """A proxy wired to fakes that record the clock and turn in call order."""

    def __init__(
        self, monkeypatch, tmp_path, *, turn_response=None, assistant_id="decomposer"
    ):
        self.calls: list[str] = []
        calls = self.calls
        self.session = SimpleNamespace(
            session_id="session",
            token="secret",
            schemas=[],
            trace=[],
            waits=[],
            lock=threading.RLock(),
            notifications_after=None,
        )
        session = self.session

        class Broker:
            base_url = "http://broker/v1"

            def register(self, scenario, notification_system):
                calls.append("register")
                return session

            def unregister(self, session_id):
                pass

        class Clock:
            def __init__(self, env):
                self.summaries = [{"frozen": True, "tool_calls": 0}]

            def freeze(self):
                calls.append("freeze")

            def before_tool(self):
                calls.append("before_tool")
                return 0.0

            def unfreeze(self):
                calls.append("unfreeze")
                return self.summaries.pop() if self.summaries else None

            def release(self):
                calls.append("release")

        class SendTool:
            name = "AgentUserInterface__send_message_to_user"

            def __call__(self, content):
                calls.append("send")

        response = turn_response or {"messages": [{"type": "ai", "content": "Done."}]}

        def fake_request(method, url, value=None, **kwargs):
            if method == "POST" and url.endswith("/threads"):
                return {"thread_id": "episode"}
            if method == "POST" and url.endswith("/runs"):
                assert value["assistant_id"] == assistant_id
                assert "agent_run_budget_seconds" not in value["context"]
                calls.append("turn")
                assert value["context"]["worker_system_prompt"]
                if isinstance(response, Exception):
                    raise response
                return {"run_id": "manager-run", "status": "success"}
            if method == "GET" and url.endswith("/state"):
                return {"values": {} if isinstance(response, Exception) else response}
            if method == "DELETE" or "/cancel?" in url:
                return {}
            raise AssertionError((method, url))

        monkeypatch.setattr(proxy, "_request", fake_request)
        monkeypatch.setattr(proxy, "FrozenTurnClock", Clock)
        monkeypatch.setattr(proxy, "frozen_turn_refusals", lambda scenario: [])
        self.scenario = SimpleNamespace(
            scenario_id="scenario",
            run_number=1,
            nb_turns=1,
            apps=[AgentUserInterface()],
            get_tools=lambda: [SendTool()],
        )
        self.agent = proxy.AgentServerAgent(
            Broker(),
            SimpleNamespace(stop_event=threading.Event()),
            {
                "assistant_id": assistant_id,
                "agent_server_url": "http://manager",
                "sidecar_root": str(tmp_path),
                "notification_poll_seconds": 0.001,
                "gaia2_revision": "gaia",
                "decomposer_revision": "core",
            },
        )
        monkeypatch.setattr(
            self.session,
            "notifications_after",
            lambda cursor: {
                "notifications": [{"type": "USER_MESSAGE", "message": "task"}],
                "next_cursor": 1,
                "environment_stopped": False,
            },
        )
        self.sidecar_path = tmp_path / "scenario__run1.json"


@pytest.mark.parametrize("assistant_id", ["decomposer", "qwen_3_5_4b_thinking"])
def test_frozen_turn_freezes_before_the_turn_and_unfreezes_after_the_send(
    monkeypatch, tmp_path, assistant_id
) -> None:
    world = FrozenTurnWorld(monkeypatch, tmp_path, assistant_id=assistant_id)

    world.agent.run_scenario(world.scenario, object())

    first_unfreeze = world.calls.index("unfreeze")
    assert world.calls[: first_unfreeze + 1] == [
        "register",
        "freeze",
        "turn",
        "before_tool",
        "send",
        "unfreeze",
    ]
    sidecar = json.loads(world.sidecar_path.read_text(encoding="utf-8"))
    assert world.calls.count("turn") == 1
    assert world.calls.count("send") == 1
    assert len(sidecar["turns"]) == 1
    assert sidecar["turns"][0]["turn_number"] == 1
    assert sidecar["turns"][0]["manager"]["trace"]["notification_cursor"] == 1
    assert sidecar["simulated_time"] == "frozen_turn"
    assert sidecar["turns"][0]["simulated_clock"] == {"frozen": True, "tool_calls": 0}
    assert sidecar["waits"] == []


def test_frozen_turn_unfreezes_when_the_turn_fails(monkeypatch, tmp_path) -> None:
    world = FrozenTurnWorld(
        monkeypatch, tmp_path, turn_response=proxy.RemoteError("manager down")
    )

    with pytest.raises(proxy.RemoteError, match="manager down"):
        world.agent.run_scenario(world.scenario, object())

    assert world.calls[:3] == ["register", "freeze", "turn"]
    assert "unfreeze" in world.calls[3:]
    assert "send" not in world.calls


def test_frozen_turn_refuses_scenarios_before_creating_an_episode(
    monkeypatch, tmp_path
) -> None:
    world = FrozenTurnWorld(monkeypatch, tmp_path)
    monkeypatch.setattr(
        proxy, "frozen_turn_refusals", lambda scenario: ["ENV event e1"]
    )

    with pytest.raises(proxy.UnsupportedScenarioError, match="ENV event e1"):
        world.agent.run_scenario(world.scenario, object())

    assert world.calls == []


def test_manager_failure_is_saved_with_traceback(monkeypatch, tmp_path):
    world = FrozenTurnWorld(
        monkeypatch, tmp_path, turn_response=proxy.RemoteError("failed")
    )
    with pytest.raises(proxy.RemoteError, match="failed"):
        world.agent.run_scenario(world.scenario, object())
    sidecar = json.loads(world.sidecar_path.read_text())
    failure = sidecar["turns"][0]["manager"]["failure"]
    assert failure["error"] == "failed"
    assert "RemoteError: failed" in failure["traceback"]
    assert "unfreeze" in world.calls
    assert "send" not in world.calls


def test_uncollected_final_is_always_rejected(monkeypatch, tmp_path):
    world = FrozenTurnWorld(
        monkeypatch,
        tmp_path,
        turn_response={
            "messages": [{"type": "ai", "content": "Done"}],
            "agent_runs": {
                "run": {"status": "responded", "response_sequence_number": None}
            },
        },
    )
    with pytest.raises(RuntimeError, match="uncollected"):
        world.agent.run_scenario(world.scenario, object())
    assert "send" not in world.calls
    assert "unfreeze" in world.calls


def test_cleanup_cancels_workers_after_completed_manager(monkeypatch):
    calls = []
    agent = proxy.AgentServerAgent(
        Broker(),
        Env(),
        {
            "assistant_id": "decomposer",
            "agent_server_url": "http://server",
            "gaia2_revision": "gaia",
            "decomposer_revision": "core",
        },
    )
    agent._thread_id, agent._run_id = "manager", "manager-run"

    def request(method, url, value=None, **kwargs):
        calls.append((method, url))
        if "manager-run/cancel" in url:
            raise proxy.RemoteError("already completed")
        if url.endswith("/state"):
            return {
                "values": {
                    "agents": {"a": {"thread_id": "worker"}},
                    "agent_runs": {
                        "r": {
                            "agent_id": "a",
                            "run_id": "worker-run",
                            "status": "running",
                        }
                    },
                }
            }
        return {}

    monkeypatch.setattr(proxy, "_request", request)
    agent.stop()
    assert calls == [
        ("POST", "http://server/threads/manager/runs/manager-run/cancel?wait=1"),
        ("GET", "http://server/threads/manager/state"),
        ("POST", "http://server/threads/worker/runs/worker-run/cancel?wait=1"),
        ("DELETE", "http://server/threads/worker"),
        ("DELETE", "http://server/threads/manager"),
    ]


def test_stop_before_creation_does_not_start_an_episode(monkeypatch, tmp_path):
    world = FrozenTurnWorld(monkeypatch, tmp_path)
    world.agent.stop()
    with pytest.raises(proxy.RemoteError, match="cancelled"):
        world.agent.run_scenario(world.scenario, object())
    assert "turn" not in world.calls


def test_run_deadline_cancels_manager_and_saves_error(monkeypatch, tmp_path):
    world = FrozenTurnWorld(monkeypatch, tmp_path)
    original = proxy._request

    def request(method, url, value=None, **kwargs):
        if method == "POST" and url.endswith("/runs"):
            return {"run_id": "manager-run", "status": "pending"}
        return original(method, url, value, **kwargs)

    monkeypatch.setattr(proxy, "_request", request)
    world.agent.request_timeout = 0
    with pytest.raises(TimeoutError, match="deadline"):
        world.agent.run_scenario(world.scenario, object())
    failure = json.loads(world.sidecar_path.read_text())["turns"][0]["manager"][
        "failure"
    ]
    assert "TimeoutError" in failure["traceback"]
    assert world.agent._cancelled
    assert "unfreeze" in world.calls


def test_multiturn_scenario_is_rejected_before_startup(monkeypatch, tmp_path):
    from gyms.gaia2.are import frozen_turn_refusals

    world = FrozenTurnWorld(monkeypatch, tmp_path)
    world.scenario.nb_turns = 2
    monkeypatch.setattr(proxy, "frozen_turn_refusals", frozen_turn_refusals)
    with pytest.raises(proxy.UnsupportedScenarioError, match="2 turns"):
        world.agent.run_scenario(world.scenario, object())
    assert world.calls == []


@pytest.mark.parametrize("failure", ["error", "timeout", "cancelled", "empty_final"])
@pytest.mark.parametrize("cleanup_state_unavailable", [False, True])
def test_failed_turn_preserves_messages_before_cleanup(
    monkeypatch, tmp_path, failure, cleanup_state_unavailable
):
    world = FrozenTurnWorld(monkeypatch, tmp_path)
    original = proxy._request
    messages = [
        {"type": "human", "content": "task"},
        {"type": "ai", "content": "", "usage_metadata": {"output_tokens": 17}},
    ]
    state = {
        "messages": messages,
        "agents": {"a": {"thread_id": "worker", "agent_type_id": "worker"}},
        "agent_runs": {
            "r": {"agent_id": "a", "run_id": "worker-run", "status": "running"}
        },
    }
    cancelled = False
    deleted = []

    def request(method, url, value=None, **kwargs):
        nonlocal cancelled
        if method == "POST" and url.endswith("/runs"):
            status = {"timeout": "pending", "empty_final": "success"}.get(failure, failure)
            return {"run_id": "manager-run", "status": status}
        if "/manager-run/cancel?" in url:
            cancelled = True
            return {}
        if method == "GET" and url.endswith("/state"):
            assert not deleted
            if cancelled and cleanup_state_unavailable:
                raise proxy.RemoteError("state unavailable")
            return {"values": state, "tasks": [{"error": "model failed"}]}
        if method == "DELETE":
            deleted.append(url)
        return original(method, url, value, **kwargs)

    monkeypatch.setattr(proxy, "_request", request)
    if failure == "timeout":
        world.agent.request_timeout = 0
    with pytest.raises((proxy.RemoteError, TimeoutError, RuntimeError)):
        world.agent.run_scenario(world.scenario, object())

    manager = json.loads(world.sidecar_path.read_text())["turns"][0]["manager"]
    if failure == "timeout" and cleanup_state_unavailable:
        assert manager["trace"]["manager_messages"] == []
        assert "deadline" in manager["failure"]["error"]
    else:
        assert manager["trace"]["manager_messages"] == [
            {"type": message["type"], "data": message} for message in messages
        ]
        assert manager["usage"] == {"output_tokens": 17}
        assert manager["outstanding_subagents"] == ["r"]
        assert manager["subagent_states"][0]["agent_type_id"] == "worker"
    assert deleted[-1].endswith("/threads/episode")
    assert "send" not in world.calls
