import asyncio
import importlib
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from decomposer import models
from scripts.lmrouter.socket_relay import relay


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_registry_models_use_socket_for_sync_and_async_calls(monkeypatch, tmp_path):
    import httpx
    seen = {}
    requests = []

    def respond(request):
        assert request.url.host == "lmrouter.2a2i.org"
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        requests.append(body["model"])
        return httpx.Response(200, json={
            "id": "chatcmpl-proxy-test", "object": "chat.completion", "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "OK", "reasoning_content": "thinking",
            }}],
        })

    def sync_transport(**kwargs):
        seen["sync"] = kwargs
        return httpx.MockTransport(respond)

    def async_transport(**kwargs):
        seen["async"] = kwargs
        return httpx.MockTransport(respond)

    with monkeypatch.context() as patch:
        patch.setenv("LLM_PROXY_UNIX_SOCKET", str(tmp_path / "router.sock"))
        patch.setenv("LLM_PROXY_MASTER_KEY", "test-key")
        patch.setattr(httpx, "HTTPTransport", sync_transport)
        patch.setattr(httpx, "AsyncHTTPTransport", async_transport)
        importlib.reload(models)
        for model in models.MODELS.values():
            sync_response = model.invoke("Reply OK")
            async_response = await model.ainvoke("Reply OK")
            assert sync_response.content == async_response.content == "OK"
            if model.preserve_reasoning:
                assert sync_response.additional_kwargs["reasoning_content"] == "thinking"
                assert async_response.additional_kwargs["reasoning_content"] == "thinking"
        model.http_client.close()
        await model.http_async_client.aclose()
    importlib.reload(models)
    assert seen["sync"]["uds"] == seen["async"]["uds"] == str(tmp_path / "router.sock")
    assert len(requests) == 4


@pytest.mark.anyio
async def test_relay_preserves_bytes(socket_path):
    async def echo(reader, writer):
        writer.write(await reader.readexactly(5))
        await writer.drain()
        writer.close()

    upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = upstream.sockets[0].getsockname()[1]
    path = socket_path
    server = await asyncio.start_unix_server(lambda r, w: relay(r, w, port), path=path)
    async with upstream, server:
        reader, writer = await asyncio.open_unix_connection(path)
        writer.write(b"hello")
        await writer.drain()
        assert await asyncio.wait_for(reader.readexactly(5), 2) == b"hello"
        writer.close()
        await writer.wait_closed()


@pytest.fixture
def socket_path():
    # macOS Unix sockets have a short path limit; pytest's temp root can exceed it.
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        yield Path(directory) / "relay.sock"


@pytest.mark.parametrize("port", ["0", "65536", "invalid", "18443:other"])
def test_tunnel_rejects_invalid_listener_port(port):
    script = Path(__file__).resolve().parents[1] / "scripts/lmrouter/router_tunnel.sh"
    result = subprocess.run(["bash", str(script), "application", "key", "router", port],
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 2
    assert "LISTEN_PORT must be between 1 and 65535" in result.stderr


@pytest.mark.parametrize("extra,forward", [
    ([], "127.0.0.1:18443:lmrouter.2a2i.org:443"),
    (["192.0.2.10", "19443"], "127.0.0.1:19443:192.0.2.10:443"),
])
def test_tunnel_accepts_any_host_and_stops_its_ssh_child(tmp_path, extra, forward):
    script = Path(__file__).resolve().parents[1] / "scripts/lmrouter/router_tunnel.sh"
    arguments = tmp_path / "ssh-arguments"
    child_pid = tmp_path / "ssh-pid"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text("#!/usr/bin/env bash\n"
                        f"printf '%s\\n' \"$@\" > {shlex.quote(str(arguments))}\n"
                        f"printf '%s\\n' \"$$\" > {shlex.quote(str(child_pid))}\n"
                        "exec sleep 60\n")
    fake_ssh.chmod(0o700)
    process = subprocess.Popen(["bash", str(script), "application-alias", "private-key", *extra],
                               env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]})
    try:
        deadline = time.monotonic() + 3
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid.exists()
        values = arguments.read_text().splitlines()
        assert values[values.index("-R") + 1] == forward
        assert values[values.index("-i") + 1] == "private-key"
        assert values[-1] == "application-alias"
        assert "StrictHostKeyChecking=yes" in values
        assert "IdentityAgent=none" in values
    finally:
        process.terminate()
        assert process.wait(timeout=3) == 0
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid.read_text()), 0)
