"""Bounded Python tool execution; cancellation kills the entire process group."""

import asyncio
import os
from pathlib import Path
import signal
import tempfile
import time
from uuid import uuid4


async def run_command(command, workspace, timeout):
    started = time.monotonic()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=workspace, stdout=stdout, stderr=stderr,
            start_new_session=True)
        timed_out = False
        try:
            try:
                await asyncio.wait_for(process.wait(), timeout)
            except TimeoutError:
                timed_out = True
        finally:
            # Kill descendants even when their parent exits successfully.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        parts = []
        for label, stream in (("STDOUT", stdout), ("STDERR", stderr)):
            stream.seek(0)
            data = stream.read(8001)
            if data:
                parts.extend([f"=== {label} ===", data[:8000].decode(errors="replace")
                              + ("\n...[truncated]" if len(data) > 8000 else "")])
        if timed_out:
            parts.extend(["=== TIMEOUT ===", f"Exceeded {timeout}s limit."])
        parts.extend(["=== INFO ===", f"Return code: {process.returncode}",
                      f"Time: {time.monotonic() - started:.2f}s / {timeout}s limit"])
        return "\n".join(parts)


def make_python_execute(workspace):
    workspace = Path(workspace).resolve()

    async def python_execute(code: str, filename: str = "", timeout: int = 30) -> str:
        """Execute Python code in the task workspace (timeout capped at 120 seconds)."""
        filename = Path(filename).name if filename else f"{uuid4().hex}.py"
        if not filename.endswith(".py"):
            filename += ".py"
        directory = workspace / ".python_tmp"
        directory.mkdir(exist_ok=True)
        script = directory / filename
        script.write_text(code, encoding="utf-8")
        return await run_command(
            ["uv", "run", "--offline", "--no-sync", "--directory", str(workspace), str(script)],
            workspace, max(1, min(int(timeout), 120)))

    return python_execute
