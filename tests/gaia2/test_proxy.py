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
