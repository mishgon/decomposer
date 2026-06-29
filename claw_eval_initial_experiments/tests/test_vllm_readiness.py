"""Tests for vLLM readiness polling."""

from __future__ import annotations

import pytest

from claw_eval.runner.vllm_readiness import (
    VllmReadinessTimeoutError,
    _model_listed,
    normalize_vllm_base_url,
    wait_for_vllm_model,
    wait_for_vllm_targets,
)


def test_normalize_vllm_base_url():
    assert normalize_vllm_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
    assert normalize_vllm_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"


def test_model_listed_exact_and_tail():
    payload = {"data": [{"id": "Qwen/Qwen3.5-122B-A10B"}]}
    assert _model_listed("Qwen/Qwen3.5-122B-A10B", payload)
    assert _model_listed("Qwen3.5-122B-A10B", payload)  # tail match allowed
    payload_alias = {"data": [{"id": "my-prefix/Qwen3.5-4B"}]}
    assert _model_listed("Qwen/Qwen3.5-4B", payload_alias)


def test_wait_for_vllm_model_success(monkeypatch):
    calls = {"n": 0}

    class FakeResponse:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self.request = None
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"status {self.status_code}")

    def fake_get(url, **kwargs):
        calls["n"] += 1
        if url.endswith("/health"):
            return FakeResponse(200)
        if calls["n"] < 3:
            return FakeResponse(200, {"data": []})
        return FakeResponse(200, {"data": [{"id": "test-model"}]})

    monkeypatch.setattr("claw_eval.runner.vllm_readiness.httpx.get", fake_get)
    monkeypatch.setattr("claw_eval.runner.vllm_readiness.time.sleep", lambda _: None)

    wait_for_vllm_model(
        base_url="http://127.0.0.1:8000/v1",
        model_id="test-model",
        timeout_s=30,
        poll_s=0.01,
    )
    assert calls["n"] >= 3


def test_wait_for_vllm_model_timeout(monkeypatch):
    class FakeResponse:
        status_code = 200
        request = None

        def json(self):
            return {"data": []}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "claw_eval.runner.vllm_readiness.httpx.get",
        lambda *a, **k: FakeResponse(),
    )
    monkeypatch.setattr("claw_eval.runner.vllm_readiness.time.sleep", lambda _: None)
    monkeypatch.setattr("claw_eval.runner.vllm_readiness.time.monotonic", iter([0, 0, 1000]).__next__)

    with pytest.raises(VllmReadinessTimeoutError):
        wait_for_vllm_model(
            base_url="http://127.0.0.1:8000/v1",
            model_id="missing",
            timeout_s=1,
            poll_s=0.01,
        )


def test_wait_for_vllm_targets_deduped(monkeypatch):
    seen: list[str] = []

    def fake_wait(**kwargs):
        seen.append(kwargs["model_id"])

    monkeypatch.setattr("claw_eval.runner.vllm_readiness.wait_for_vllm_model", fake_wait)
    wait_for_vllm_targets([
        ("http://127.0.0.1:8000/v1", "unused", "m1"),
        ("http://127.0.0.1:8001/v1", "unused", "m2"),
    ])
    assert seen == ["m1", "m2"]
