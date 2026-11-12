import asyncio
import importlib
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

server = importlib.import_module("decomposer.agent_server")


def test_socket_stays_reserved_through_startup_and_shutdown(monkeypatch, tmp_path):
    config = tmp_path / "langgraph.json"
    config.write_text("{}")
    launches = []

    async def spawn(*command, **kwargs):
        fd = kwargs["pass_fds"][0]
        with socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM) as inherited:
            host, port = inherited.getsockname()
        with socket.socket() as competing:
            with pytest.raises(OSError):
                competing.bind((host, port))
        assert kwargs["env"]["WORKER_URL"] == f"http://127.0.0.1:{port}"
        assert command[command.index("--fd") + 1] == str(fd)
        launches.append(port)
        return SimpleNamespace(pid=12345, returncode=None, wait=AsyncMock(return_value=0))

    client = MagicMock()
    client.__aenter__.return_value = client
    client.get = AsyncMock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(server.os, "killpg", lambda *args: None)

    async def run():
        async with server.agent_server(config, port=0, url_env="WORKER_URL") as first:
            async with server.agent_server(config, port=0, url_env="WORKER_URL") as second:
                assert first != second
                for port in launches:
                    with socket.socket() as competing:
                        with pytest.raises(OSError):
                            competing.bind(("127.0.0.1", port))
        for port in launches:
            with socket.socket() as released:
                released.bind(("127.0.0.1", port))

    asyncio.run(run())
