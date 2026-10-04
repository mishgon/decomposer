import pytest
from langchain_core.messages import AIMessage
from langchain_openrouter import ChatOpenRouter

from decomposer.models import ChatVLLM, create_model


def _response(*, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning": "reasoning",
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def test_reasoning_round_trip() -> None:
    model = ChatVLLM(
        model="test",
        api_key="test",
        preserve_reasoning=True,
        use_responses_api=False,
    )

    result = model._create_chat_result(_response())
    message = result.generations[0].message
    assert message.additional_kwargs["reasoning_content"] == "reasoning"
    assert message.content_blocks[0] == {
        "type": "reasoning",
        "reasoning": "reasoning",
    }

    payload = model._get_request_payload([message])
    assert payload["messages"][0]["reasoning"] == "reasoning"


def test_legacy_reasoning_round_trip() -> None:
    model = ChatVLLM(
        model="test",
        api_key="test",
        preserve_reasoning=True,
        use_responses_api=False,
    )
    message = AIMessage(
        content="answer",
        additional_kwargs={"reasoning": "legacy reasoning"},
    )

    payload = model._get_request_payload([message])

    assert payload["messages"][0]["reasoning"] == "legacy reasoning"


def test_length_limit_is_an_error() -> None:
    model = ChatVLLM(model="test", api_key="test")

    with pytest.raises(RuntimeError, match="max_completion_tokens"):
        model._create_chat_result(_response(finish_reason="length"))


def test_flash_next_uses_nonthinking_sampling():
    model = create_model("lmrouter/qwen_3_8_flash_next_non_thinking")
    payload = model._get_request_payload("test")
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == .7
    assert payload["top_p"] == .8
    assert payload["presence_penalty"] == 1.5
    assert payload["extra_body"] == {
        "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert model.preserve_reasoning is False
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


@pytest.mark.parametrize("effort", ["low", "medium"])
def test_flash_next_uses_thinking_sampling(effort):
    model = create_model(f"lmrouter/qwen_3_8_flash_next_{effort}_thinking")
    payload = model._get_request_payload("test")
    assert model.openai_api_base == "https://lmrouter.2a2i.org/v1"
    assert payload["model"] == "Qwen/Qwen3.8-Flash-Next-NVFP4"
    assert payload["reasoning_effort"] == effort
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == .95
    assert payload["presence_penalty"] == 0.0
    assert payload["extra_body"] == {
        "allowed_openai_params": ["reasoning_effort"],
        "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
        "chat_template_kwargs": {
            "enable_thinking": True, "preserve_thinking": True,
        },
    }
    assert model.preserve_reasoning is True
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


@pytest.mark.parametrize("profile, thinking, temperature, top_p", [
    ("lmrouter/qwen_3_5_4b_unlooped_thinking", True, .6, .95),
    ("lmrouter/qwen_3_5_4b_unlooped_non_thinking", False, .7, .8),
])
def test_unlooped_uses_checkpoint_sampling(profile, thinking, temperature, top_p):
    model = create_model(model_id=profile)
    payload = model._get_request_payload("test")
    assert payload["model"] == "Qwen/Qwen3.5-4B-unlooped"
    assert payload["temperature"] == temperature
    assert payload["top_p"] == top_p
    assert "presence_penalty" not in payload
    assert payload["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking}}
    assert model.preserve_reasoning is thinking


@pytest.mark.parametrize("profile, thinking, temperature, top_p", [
    ("vllm/qwen_3_5_4b_thinking", True, 1.0, .95),
    ("vllm/qwen_3_5_4b_non_thinking", False, .7, .8),
])
def test_local_qwen_sampling_and_transport(monkeypatch, profile, thinking, temperature, top_p):
    from unittest.mock import MagicMock

    import httpx
    import decomposer.models as models

    monkeypatch.setenv("LLM_PROXY_UNIX_SOCKET", "/unused/router.sock")
    transport = MagicMock(wraps=httpx.HTTPTransport)
    async_transport = MagicMock(wraps=httpx.AsyncHTTPTransport)
    monkeypatch.setattr(models.httpx, "HTTPTransport", transport)
    monkeypatch.setattr(models.httpx, "AsyncHTTPTransport", async_transport)
    launch = MagicMock()
    monkeypatch.setattr(models.asyncio, "create_subprocess_exec", launch)
    model = create_model(profile)
    payload = model._get_request_payload("test")
    assert model.openai_api_base == "http://127.0.0.1:8024/v1"
    assert payload["model"] == "Qwen/Qwen3.5-4B"
    assert payload["temperature"] == temperature
    assert payload["top_p"] == top_p
    assert payload["presence_penalty"] == 1.5
    assert payload["extra_body"] == {
        "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    assert model.preserve_reasoning is thinking
    assert all(call.kwargs.get("uds") is None for call in transport.call_args_list)
    assert all(call.kwargs.get("uds") is None for call in async_transport.call_args_list)
    launch.assert_not_called()


@pytest.fixture
def mock_vllm(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import httpx
    import decomposer.models as models

    process = SimpleNamespace(pid=123, returncode=None, wait=AsyncMock())
    launch = AsyncMock(return_value=process)
    monkeypatch.setattr(models.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(models, "socket", SimpleNamespace(socket=MagicMock()))
    import decomposer.agent_server as server_module
    monkeypatch.setattr(server_module, "socket", SimpleNamespace(socket=MagicMock()))
    kill = MagicMock()
    monkeypatch.setattr(models.os, "killpg", kill)
    client = AsyncMock()
    client.get.return_value = httpx.Response(
        200,
    )
    client.__aenter__.return_value = client
    monkeypatch.setattr(models.httpx, "AsyncClient", MagicMock(return_value=client))
    return SimpleNamespace(process=process, launch=launch, kill=kill, client=client)


@pytest.mark.parametrize("body_fails", [False, True])
def test_vllm_server_lifecycle(mock_vllm, body_fails):
    import asyncio
    import signal
    from decomposer.models import vllm_server

    async def run():
        async with vllm_server("vllm/qwen_3_5_4b_thinking", gpu="2"):
            mock_vllm.client.get.assert_awaited_once_with(
                "http://127.0.0.1:8024/health", timeout=2,
            )
            if body_fails:
                raise RuntimeError("body failed")

    if body_fails:
        with pytest.raises(RuntimeError, match="body failed"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert mock_vllm.launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    mock_vllm.kill.assert_called_once_with(123, signal.SIGTERM)
    mock_vllm.process.wait.assert_awaited_once()


def test_vllm_server_reports_startup_failure(mock_vllm):
    import asyncio
    from decomposer.models import vllm_server

    mock_vllm.process.returncode = 1

    async def run():
        async with vllm_server("vllm/qwen_3_5_4b_non_thinking", gpu="2"):
            pytest.fail("Failed server must not enter the context")

    with pytest.raises(RuntimeError, match="Server exited during startup: 1"):
        asyncio.run(run())


def test_vllm_server_rejects_router_model(mock_vllm):
    import asyncio
    from decomposer.models import vllm_server

    async def run():
        async with vllm_server("lmrouter/qwen_3_5_4b_unlooped_thinking", gpu="2"):
            pytest.fail("Router model must not start a local server")

    with pytest.raises(ValueError, match="Unsupported local vLLM model"):
        asyncio.run(run())
    mock_vllm.launch.assert_not_called()


def test_agent_server_lifecycle(mock_vllm, tmp_path):
    import asyncio
    from pathlib import Path
    from decomposer.agent_server import agent_server

    config = tmp_path / "langgraph.json"
    config.write_text('{"graphs": {}}')

    async def run():
        async with agent_server(config, port=2025) as url:
            assert url == "http://127.0.0.1:2025"
            mock_vllm.client.get.assert_awaited_once_with(f"{url}/ok", timeout=2)
            workdir = Path(mock_vllm.launch.call_args.kwargs["cwd"])
            assert (workdir / "langgraph.json").resolve() == config
        assert not workdir.exists()

    asyncio.run(run())
    mock_vllm.process.wait.assert_awaited_once()


def test_agent_server_startup_timeout_cleans_up(mock_vllm, tmp_path):
    import asyncio
    import httpx
    from decomposer.agent_server import agent_server

    mock_vllm.client.get.return_value = httpx.Response(503)

    async def run():
        async with agent_server(tmp_path / "langgraph.json", startup_timeout=0):
            pytest.fail("Unready server must not enter the context")

    with pytest.raises(TimeoutError):
        asyncio.run(run())
    mock_vllm.kill.assert_called_once()
    mock_vllm.process.wait.assert_awaited_once()


def test_vllm_server_forces_shutdown_if_term_fails(mock_vllm):
    import asyncio
    import signal
    from unittest.mock import call
    from decomposer.models import vllm_server

    mock_vllm.process.wait.side_effect = [TimeoutError, None]

    async def run():
        async with vllm_server("vllm/qwen_3_5_4b_thinking", gpu="2"):
            pass

    asyncio.run(run())
    assert mock_vllm.kill.call_args_list == [call(123, signal.SIGTERM), call(123, signal.SIGKILL)]


def test_vllm_server_rejects_occupied_port(mock_vllm):
    import asyncio
    import decomposer.models as models

    models.socket.socket.return_value.__enter__.return_value.bind.side_effect = OSError("Address in use")

    async def run():
        async with models.vllm_server("vllm/qwen_3_5_4b_thinking", gpu="2"):
            pytest.fail("Occupied port must not start a server")

    with pytest.raises(OSError, match="Address in use"):
        asyncio.run(run())
    mock_vllm.launch.assert_not_called()


@pytest.mark.parametrize("effort", ["low", "medium"])
def test_openrouter_flash_next_request(monkeypatch, effort):
    from unittest.mock import Mock

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("LLM_PROXY_UNIX_SOCKET", "/unused/router.sock")
    model = create_model(f"openrouter/qwen_3_8_flash_next_{effort}_thinking")
    assert isinstance(model, ChatOpenRouter)
    assert model.openrouter_api_base == "https://openrouter.ai/api/v1"
    assert model.openrouter_api_key.get_secret_value() == "test-openrouter-key"
    model.client = Mock()
    model.client.chat.send.return_value = _response()
    assert model.invoke("test").content == "answer"
    payload = model.client.chat.send.call_args.kwargs
    assert payload["model"] == "qwen/qwen3.8-flash"
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == .95
    assert payload["presence_penalty"] == 0.0
    assert payload["top_k"] == 20
    assert payload["reasoning"] == {"effort": effort}
    assert model.request_timeout == 600_000


def test_openrouter_reasoning_details_round_trip():
    model = ChatOpenRouter(model="test", api_key="test")
    response = _response()
    details = [{"type": "reasoning.text", "text": "reasoning", "index": 0}]
    response["choices"][0]["message"]["reasoning_details"] = details
    message = model._create_chat_result(response).generations[0].message
    assert message.additional_kwargs["reasoning_details"] == details
    messages, _ = model._create_message_dicts([message], None)
    assert messages[0]["reasoning_details"] == details


@pytest.mark.parametrize("thinking", [True, False])
def test_local_qwen_uses_container_host(monkeypatch, thinking):
    monkeypatch.setenv("VLLM_HOST", "host.docker.internal")
    profile = "thinking" if thinking else "non_thinking"
    model = create_model(f"vllm/qwen_3_5_4b_{profile}")
    assert model.openai_api_base == "http://host.docker.internal:8024/v1"
