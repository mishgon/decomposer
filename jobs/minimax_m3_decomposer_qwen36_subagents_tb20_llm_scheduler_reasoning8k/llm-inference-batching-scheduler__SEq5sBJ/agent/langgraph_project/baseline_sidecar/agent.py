from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_WORKDIR = "/solution"
MAX_TOOL_OUTPUT_CHARS = 20_000


def _task_workdir() -> Path:
    configured = os.environ.get("T_BENCH_WORKDIR") or DEFAULT_WORKDIR
    path = Path(configured)
    if path.exists():
        return path
    return Path.cwd()


def _truncate(value: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    keep = limit // 2
    return (
        value[:keep]
        + f"\n\n[... truncated {len(value) - limit} characters ...]\n\n"
        + value[-keep:]
    )


@tool
def bash(command: str, timeout_seconds: int = 300) -> str:
    """Run a bash command in the Terminal-Bench task workspace.

    Each call starts in /solution by default. Use absolute paths or include `cd`
    in the command when needed.

    Args:
        command: Bash command to execute.
        timeout_seconds: Maximum runtime in seconds, capped at 900.
    """
    timeout = max(1, min(int(timeout_seconds), 900))
    cwd = _task_workdir()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return _truncate(
            f"cwd: {cwd}\n"
            f"timeout_seconds: {timeout}\n"
            f"status: timed out\n\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )

    return _truncate(
        f"cwd: {cwd}\n"
        f"exit_code: {completed.returncode}\n\n"
        f"stdout:\n{completed.stdout}\n\n"
        f"stderr:\n{completed.stderr}"
    )


def _configured_model(config: dict[str, Any] | None) -> str:
    configurable = (config or {}).get("configurable") or {}
    model = (
        os.environ.get("SUBAGENT_OPENAI_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or configurable.get("model")
        or os.environ.get("HARBOR_MODEL")
        or DEFAULT_MODEL
    )
    if isinstance(model, str) and model.startswith("openai:"):
        return model.split(":", 1)[1]
    return str(model)


def _env_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _extra_body() -> dict[str, Any]:
    body: dict[str, Any] = {
        "top_k": _env_int("OPENAI_TOP_K", "20"),
        "min_p": _env_float("OPENAI_MIN_P", "0"),
        "repetition_penalty": _env_float("OPENAI_REPETITION_PENALTY", "1.0"),
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    }
    thinking_budget = _env_optional_int("OPENAI_THINKING_TOKEN_BUDGET")
    if thinking_budget is not None:
        body["thinking_token_budget"] = thinking_budget
    return body


def make_baseline(config: dict[str, Any] | None = None):
    model = ChatOpenAI(
        model=_configured_model(config),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=_env_float("OPENAI_TEMPERATURE", "0.6"),
        top_p=_env_float("OPENAI_TOP_P", "0.95"),
        presence_penalty=_env_float("OPENAI_PRESENCE_PENALTY", "0.0"),
        timeout=_env_float("OPENAI_TIMEOUT", "120"),
        max_retries=2,
        max_completion_tokens=_env_int("OPENAI_MAX_TOKENS", "8192"),
        extra_body=_extra_body(),
    )
    return create_agent(
        model=model,
        tools=[bash],
        system_prompt=(
            "You are an autonomous tool-using agent running in a sandboxed workspace at /solution. "
            "Follow the caller's request exactly. Use the bash tool to inspect files, run commands, "
            "edit files, and test work when needed. Each bash call starts in /solution; shell state "
            "such as `cd` does not persist between calls. "
            "Your final response is the only information returned to the caller. Make it self-contained: "
            "include the requested results, facts, file contents, command outputs, paths changed, errors, "
            "or test results needed for the caller to continue."
        ),
    )
