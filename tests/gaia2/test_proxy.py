from __future__ import annotations

import pytest

pytest.importorskip("are")

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
