import asyncio
import tempfile
from pathlib import Path

import pytest

from gyms.toolathlon_gym.subagents.model_config import model_http_client
from sft.toolathlon_gym.inference.socket_relay import relay


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_model_client_uses_socket_without_changing_https_hostname(monkeypatch, tmp_path):
    import httpx
    seen = {}

    def transport(**kwargs):
        seen.update(kwargs)
        async def respond(request):
            assert request.url.host == "lmrouter.example"
            return httpx.Response(200, json={"ok": True})
        return httpx.MockTransport(respond)

    monkeypatch.setenv("LLM_PROXY_UNIX_SOCKET", str(tmp_path / "router.sock"))
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    async with model_http_client() as client:
        response = await client.get("https://lmrouter.example/v1/models")
        assert response.json() == {"ok": True}
    assert seen["uds"] == str(tmp_path / "router.sock")


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
