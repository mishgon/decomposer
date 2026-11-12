import asyncio
import os
import signal
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx


@asynccontextmanager
async def agent_server(
    config_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 2024,
    startup_timeout: float = 60,
    n_jobs_per_worker: int = 16,
    url_env: str | None = None,
) -> AsyncIterator[str]:
    """Run a local Agent Server and stop it when the context exits."""
    config_path = Path(config_path).resolve()
    client_host = "127.0.0.1" if host == "0.0.0.0" else host
    with socket.socket() as listener, TemporaryDirectory(prefix="agent-server-") as workdir:
        # Keep the socket reserved across startup. A free-port probe followed
        # by a later bind can connect readiness checks to another episode.
        listener.bind((host, port))
        port = listener.getsockname()[1]
        url = f"http://{client_host}:{port}"
        for path in config_path.parent.iterdir():
            if path.name != ".langgraph_api":
                (Path(workdir) / path.name).symlink_to(path)
        command = [
            sys.executable, "-m", "decomposer.agent_server",
            "--config", str(config_path),
            "--host", host, "--port", str(port),
            "--fd", str(listener.fileno()),
            "--n-jobs-per-worker", str(n_jobs_per_worker),
        ]
        process = await asyncio.create_subprocess_exec(
            *command, cwd=workdir, start_new_session=True,
            pass_fds=(listener.fileno(),),
            env={**os.environ, **({url_env: url} if url_env else {})},
        )
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                async with asyncio.timeout(startup_timeout):
                    while True:
                        if process.returncode is not None:
                            raise RuntimeError(f"Server exited during startup: {process.returncode}")
                        try:
                            response = await client.get(f"{url}/ok", timeout=2)
                            if response.status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        await asyncio.sleep(0.2)
            yield url
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()


def main():
    """Use the same dev-server entrypoint as LangGraph CLI with an inherited FD."""
    import argparse
    from langgraph_api.cli import run_server
    from langgraph_cli.config import validate_config_file

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--host", required=True)
    cli.add_argument("--port", type=int, required=True)
    cli.add_argument("--fd", type=int, required=True)
    cli.add_argument("--n-jobs-per-worker", type=int, required=True)
    args = cli.parse_args()
    config = validate_config_file(args.config)
    sys.path.append(os.getcwd())
    for dependency in config["dependencies"]:
        path = Path.cwd() / dependency
        if path.is_dir():
            sys.path.append(str(path))
    options = {key: config[key] for key in (
        "graphs", "env", "store", "auth", "http", "ui", "ui_config",
        "webhooks", "checkpointer", "disable_persistence",
    ) if key in config}
    run_server(host=args.host, port=args.port, fd=args.fd,
               n_jobs_per_worker=args.n_jobs_per_worker, **options)


if __name__ == "__main__":
    main()
