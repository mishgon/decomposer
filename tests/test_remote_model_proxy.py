from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gyms import remote_model_proxy


def _payload() -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "name": "run",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent_id": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["subagent_id", "prompt"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "wait",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        ]
    }


def test_qwen_xml_response_becomes_function_calls_and_keeps_reasoning() -> None:
    response = {
        "output": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Delegate."}]},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "\n<tool_call><function=run>"
                            "<parameter=subagent_id>s1</parameter>"
                            "<parameter=prompt>Find the report.</parameter>"
                            "</function></tool_call>"
                            "<tool_call><function=wait></function></tool_call>\n"
                        ),
                    }
                ],
            },
        ]
    }

    normalized = remote_model_proxy.normalize_qwen3_xml_response(
        response, _payload()
    )

    assert normalized["output"][0] == response["output"][0]
    calls = [item for item in normalized["output"] if item["type"] == "function_call"]
    assert [item["name"] for item in calls] == ["run", "wait"]
    assert json.loads(calls[0]["arguments"]) == {
        "subagent_id": "s1",
        "prompt": "Find the report.",
    }
    assert json.loads(calls[1]["arguments"]) == {}
    assert all(item["id"].startswith("fc_") for item in calls)
    assert all(item["call_id"].startswith("call_") for item in calls)


@pytest.mark.parametrize(
    "text, message",
    [
        ("<tool_call><function=missing></function></tool_call>", "unknown tool"),
        ("<tool_call><function=wait>", "Malformed Qwen"),
        (
            "<tool_call><function=wait><parameter=x>1</parameter></function></tool_call>",
            "unknown argument",
        ),
    ],
)
def test_qwen_xml_parser_rejects_invalid_calls(text: str, message: str) -> None:
    with pytest.raises(remote_model_proxy.ToolCallParseError, match=message):
        remote_model_proxy.parse_qwen3_xml_tool_calls(
            text, remote_model_proxy._tool_schemas(_payload())
        )


def test_proxy_normalizes_live_response_shape_and_isolates_credential(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["verify"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def request(self, method, url, **kwargs):
            calls.append({"method": method, "url": url, **kwargs})
            request = httpx.Request(method, url)
            return httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "<tool_call><function=wait></function></tool_call>",
                                }
                            ],
                        }
                    ]
                },
                request=request,
            )

    monkeypatch.setattr(remote_model_proxy.httpx, "AsyncClient", FakeClient)
    app = remote_model_proxy.create_app(
        upstream_url="https://internal.test/v1",
        api_key="internal-secret",
        extra_body={
            "temperature": 0.7,
            "max_output_tokens": 32768,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        response_tool_parser="qwen3_xml",
        verify_tls=False,
    )
    payload = _payload()
    payload.update({"model": "Qwen/test", "input": [], "stream": False})
    with TestClient(app) as client:
        result = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer untrusted-caller"},
            json=payload,
        )

    assert result.status_code == 200
    assert result.json()["output"][0]["type"] == "function_call"
    assert calls[0]["headers"]["Authorization"] == "Bearer internal-secret"
    forwarded = calls[0]["json"]
    assert forwarded["temperature"] == 0.7
    assert forwarded["max_output_tokens"] == 32768
    assert forwarded["chat_template_kwargs"] == {"enable_thinking": False}


class _FakeModelsClient:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload
        self.requested: list[str] = []

    def __call__(self, **kwargs: object) -> "_FakeModelsClient":
        self.kwargs = kwargs
        return self

    def __enter__(self) -> "_FakeModelsClient":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, url: str, headers: dict[str, str]) -> "_FakeModelsClient":
        self.requested.append(url)
        return self

    def json(self) -> object:
        return self.payload


def test_upstream_model_preflight_names_missing_models_without_the_url(monkeypatch) -> None:
    monkeypatch.setenv("TEST_UPSTREAM_URL", "https://secret-host.example/v1")
    monkeypatch.setenv("TEST_UPSTREAM_KEY", "sk-test")
    client = _FakeModelsClient(200, {"data": [{"id": "Qwen/A"}, {"id": "Qwen/B"}]})
    monkeypatch.setattr(remote_model_proxy.httpx, "Client", client)

    remote_model_proxy.require_upstream_models(
        "TEST_UPSTREAM_URL", "TEST_UPSTREAM_KEY", ["Qwen/A", "Qwen/B"], verify_tls=False
    )
    assert client.requested == ["https://secret-host.example/v1/models"]
    assert client.kwargs["verify"] is False and client.kwargs["trust_env"] is False

    with pytest.raises(RuntimeError) as missing:
        remote_model_proxy.require_upstream_models(
            "TEST_UPSTREAM_URL", "TEST_UPSTREAM_KEY", ["Qwen/A", "Qwen/C"]
        )
    assert "Qwen/C" in str(missing.value)
    assert "secret-host" not in str(missing.value)

    monkeypatch.setattr(
        remote_model_proxy.httpx, "Client", _FakeModelsClient(503, {})
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        remote_model_proxy.require_upstream_models(
            "TEST_UPSTREAM_URL", "TEST_UPSTREAM_KEY", ["Qwen/A"]
        )
    monkeypatch.delenv("TEST_UPSTREAM_KEY")
    with pytest.raises(RuntimeError, match="must be set"):
        remote_model_proxy.require_upstream_models(
            "TEST_UPSTREAM_URL", "TEST_UPSTREAM_KEY", ["Qwen/A"]
        )
