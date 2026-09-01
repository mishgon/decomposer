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
                "name": "spawn_subagent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent_type_id": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["subagent_type_id", "prompt"],
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
                            "\n<tool_call><function=spawn_subagent>"
                            "<parameter=subagent_type_id>qwen</parameter>"
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
    assert [item["name"] for item in calls] == ["spawn_subagent", "wait"]
    assert json.loads(calls[0]["arguments"]) == {
        "subagent_type_id": "qwen",
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
