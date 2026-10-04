"""Serve the agents for one preprocessed Toolathlon task inside its container."""

import asyncio
import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import httpx


PID_FILE = Path("/run/decomposer-agent-server.pid")


async def serve() -> None:
    from agents import DECOMPOSER_MODEL_ID, QWEN_3_5_4B_THINKING_MODEL_ID
    from decomposer.agent_server import agent_server

    bundle_path = Path(os.environ["TOOLATHLON_BUNDLE"])
    data_dir = Path(os.environ.get("TOOLATHLON_DATA_DIR", "/workspace/dumps"))
    gateway_port = int(os.environ["GATEWAY_PORT"])
    agent_port = int(os.environ["AGENT_SERVER_PORT"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopped.set)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    with (data_dir / "gateway.log").open("ab") as log:
        gateway = await asyncio.create_subprocess_exec(
            "uv", "run", "python", str(Path(__file__).with_name("tool_gateway.py")),
            "--bundle-file", str(bundle_path), "--port", str(gateway_port),
            cwd="/workspace", stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        gateway_url = f"http://127.0.0.1:{gateway_port}"
        await wait_ready(f"{gateway_url}/health", gateway, timeout=120)
        # The trusted bundle names grader paths, so agents must never read it.
        bundle_path.unlink()
        runtime = {
            "agent_system_prompt": bundle["system_prompts"]["agent"],
            "agent_workspace": bundle["container_paths"]["agent_workspace"],
            "gateway_url": f"{gateway_url}/sse",
            "agent_server_url": f"http://127.0.0.1:{agent_port}",
            "agent_model": model_metadata(QWEN_3_5_4B_THINKING_MODEL_ID),
            "decomposer_model": model_metadata(DECOMPOSER_MODEL_ID),
        }
        (data_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
        async with agent_server(
            Path(__file__).with_name("langgraph.json"),
            port=agent_port,
            startup_timeout=180,
            n_jobs_per_worker=int(os.environ.get("N_JOBS_PER_WORKER", "16")),
        ):
            await stopped.wait()
    finally:
        with suppress(ProcessLookupError):
            os.killpg(gateway.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(gateway.wait(), timeout=30)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(gateway.pid, signal.SIGKILL)
            await gateway.wait()
        PID_FILE.unlink(missing_ok=True)


async def wait_ready(url: str, process, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"Tool gateway exited during startup: {process.returncode}")
            try:
                if (await client.get(url, timeout=2)).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.2)
    raise TimeoutError(f"{url} did not become ready within {timeout:g}s")


def model_metadata(model_id):
    """Describe the image's model configuration without credentials or clients."""
    from decomposer.models import create_model

    model = create_model(model_id)
    values = model.model_dump(include={
        "model_name", "openai_api_base", "openrouter_api_base", "temperature", "top_p",
        "presence_penalty", "max_tokens", "extra_body", "model_kwargs", "reasoning",
        "reasoning_effort", "request_timeout", "max_retries",
    }, exclude_none=True)
    if hasattr(model, "preserve_reasoning"):
        values["preserve_reasoning"] = model.preserve_reasoning
    return {
        "model_id": model_id,
        "api_model": values["model_name"],
        "base_url": values.get("openai_api_base", values.get("openrouter_api_base")),
        "generation_config": values,
    }


if __name__ == "__main__":
    asyncio.run(serve())
