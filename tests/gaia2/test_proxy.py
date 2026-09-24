from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("are")

from are.simulation.apps.agent_user_interface import AgentUserInterface
from gyms.gaia2 import proxy


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
        return {"deleted": True}

    monkeypatch.setattr(proxy, "_request", fake_request)
    agent = proxy.DecomposerProxyAgent(
        Broker(), Env(), {"service_url": "http://manager"}
    )
    agent._episode_id = "episode"
    agent.stop()
    agent.stop()
    assert calls == [("DELETE", "http://manager/v1/episodes/episode")]


def test_uncollected_final_is_strict_by_default_and_opt_in() -> None:
    strict = proxy.DecomposerProxyAgent(Broker(), Env(), {})
    probe = proxy.DecomposerProxyAgent(
        Broker(), Env(), {"allow_uncollected_final": True}
    )

    assert strict.allow_uncollected_final is False
    assert probe.allow_uncollected_final is True
    response = {"outstanding_subagents": [{"status": "running"}]}
    assert strict._uncollected_final_is_blocking(response) is True
    assert probe._uncollected_final_is_blocking(response) is False


def test_manager_overflow_is_persisted_then_fails_rollout(monkeypatch, tmp_path) -> None:
    session = SimpleNamespace(
        session_id="session",
        token="secret",
        schemas=[],
        trace=[],
        waits=[],
    )

    class FakeBroker:
        base_url = "http://broker/v1"

        def register(self, scenario, notification_system):
            return session

        def unregister(self, session_id):
            assert session_id == "session"

    failure = {
        "actor": "manager",
        "kind": "output_token_overflow",
        "policy": "fail_actor_v1",
        "max_completion_tokens": 8192,
        "max_model_len": 131072,
        "detail": "finish_reason=length",
    }

    def fake_request(method, url, value=None, **kwargs):
        if method == "POST" and url.endswith("/v1/episodes"):
            return {"episode_id": "episode"}
        if method == "POST" and url.endswith("/turn"):
            return {"failure": failure, "trace": {}, "timing": {}}
        if method == "DELETE":
            return {"deleted": True}
        raise AssertionError((method, url))

    monkeypatch.setattr(proxy, "_request", fake_request)
    aui = AgentUserInterface()
    scenario = SimpleNamespace(
        scenario_id="scenario",
        run_number=2,
        nb_turns=1,
        apps=[aui],
        get_tools=aui.get_tools,
    )
    agent = proxy.DecomposerProxyAgent(
        FakeBroker(),
        SimpleNamespace(stop_event=threading.Event()),
        {
            "service_url": "http://manager",
            "sidecar_root": str(tmp_path),
            "notification_poll_seconds": 0.001,
        },
    )
    monkeypatch.setattr(
        agent,
        "_broker_get",
        lambda suffix: {
            "notifications": [{"type": "USER_MESSAGE", "message": "task"}],
            "next_cursor": 1,
            "environment_stopped": False,
        },
    )

    with pytest.raises(proxy.RemoteModelOverflowError, match="output_token_overflow"):
        agent.run_scenario(scenario, object())

    sidecar = json.loads(
        (tmp_path / "scenario__run2.json").read_text(encoding="utf-8")
    )
    assert len(sidecar["turns"]) == 1
    assert sidecar["turns"][0]["manager"]["failure"] == failure
    assert aui.messages == []
    # Configs without the setting keep the wall clock.
    assert session.clock is None
    assert sidecar["simulated_time"] == "wall_clock"


class FrozenTurnWorld:
    """A proxy wired to fakes that record the clock and turn in call order."""

    def __init__(self, monkeypatch, tmp_path, *, turn_response=None):
        self.calls: list[str] = []
        calls = self.calls
        self.session = SimpleNamespace(
            session_id="session",
            token="secret",
            schemas=[],
            trace=[],
            waits=[],
            lock=threading.RLock(),
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

        response = turn_response or {"final_text": "Done.", "outstanding_subagents": []}

        def fake_request(method, url, value=None, **kwargs):
            if method == "POST" and url.endswith("/v1/episodes"):
                return {"episode_id": "episode"}
            if method == "POST" and url.endswith("/turn"):
                calls.append("turn")
                if isinstance(response, Exception):
                    raise response
                return response
            if method == "DELETE":
                return {"deleted": True}
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
        self.agent = proxy.DecomposerProxyAgent(
            Broker(),
            SimpleNamespace(stop_event=threading.Event()),
            {
                "service_url": "http://manager",
                "sidecar_root": str(tmp_path),
                "notification_poll_seconds": 0.001,
                "simulated_time": "frozen_turn",
            },
        )
        monkeypatch.setattr(
            self.agent,
            "_broker_get",
            lambda suffix: {
                "notifications": [{"type": "USER_MESSAGE", "message": "task"}],
                "next_cursor": 1,
                "environment_stopped": False,
            },
        )
        self.sidecar_path = tmp_path / "scenario__run1.json"


def test_frozen_turn_freezes_before_the_turn_and_unfreezes_after_the_send(
    monkeypatch, tmp_path
) -> None:
    world = FrozenTurnWorld(monkeypatch, tmp_path)

    world.agent.run_scenario(world.scenario, object())

    first_unfreeze = world.calls.index("unfreeze")
    assert world.calls[:first_unfreeze + 1] == [
        "register",
        "freeze",
        "turn",
        "before_tool",
        "send",
        "unfreeze",
    ]
    sidecar = json.loads(world.sidecar_path.read_text(encoding="utf-8"))
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


def test_unknown_simulated_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="simulated_time"):
        proxy.DecomposerProxyAgent(Broker(), Env(), {"simulated_time": "fast"})
