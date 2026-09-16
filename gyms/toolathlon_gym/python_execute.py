"""Bounded Python tool execution; cancellation kills the entire process group."""

import asyncio
import os
from pathlib import Path
import signal
import time
from uuid import uuid4


async def run_command(command, workspace, timeout):
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *command, cwd=workspace, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, start_new_session=True)

    async def drain(stream):
        data = bytearray()
        while chunk := await stream.read(65536):
            data.extend(chunk[:max(0, 8001 - len(data))])
        return bytes(data)

    readers = [asyncio.create_task(drain(stream)) for stream in (process.stdout, process.stderr)]
    async def wait_parent():
        # process.wait() can wait for pipes inherited by background descendants.
        while process.returncode is None:
            await asyncio.sleep(.02)

    timed_out = False
    try:
        try:
            await asyncio.wait_for(wait_parent(), timeout)
        except TimeoutError:
            timed_out = True
    finally:
        # Kill descendants even when their parent exits successfully.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        outputs = await asyncio.gather(*readers)
    parts = []
    for label, data in zip(("STDOUT", "STDERR"), outputs):
        if data:
            parts.extend([f"=== {label} ===", data[:8000].decode(errors="replace")
                          + ("\n...[truncated]" if len(data) > 8000 else "")])
    if timed_out:
        parts.extend(["=== TIMEOUT ===", f"Exceeded {timeout}s limit."])
    parts.extend(["=== INFO ===", f"Return code: {process.returncode}",
                  f"Time: {time.monotonic() - started:.2f}s / {timeout}s limit"])
    return "\n".join(parts)


def make_python_execute(workspace):
    workspace = Path(workspace)

    async def python_execute(code: str, filename: str = "", timeout: int = 30) -> str:
        """Execute Python code in the task workspace (timeout capped at 120 seconds)."""
        filename = Path(filename).name if filename else f"{uuid4().hex}.py"
        if not filename.endswith(".py"):
            filename += ".py"
        directory = workspace / ".python_tmp"
        await asyncio.to_thread(directory.mkdir, exist_ok=True)
        script = directory / filename
        await asyncio.to_thread(script.write_text, code, encoding="utf-8")
        return await run_command(
            ["uv", "run", "--offline", "--no-sync", "--directory", str(workspace), str(script)],
            workspace, max(1, min(int(timeout), 120)))

    return python_execute
