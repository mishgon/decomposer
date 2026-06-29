from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph_sdk import get_sync_client


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from decomposer.core import create_decomposer_agent  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_BASELINE_PROJECT = PROJECT_DIR / "baseline_sidecar"


def _configured_model(config: dict[str, Any] | None) -> str:
    configurable = (config or {}).get("configurable") or {}
    model = (
        configurable.get("model")
        or os.environ.get("OPENAI_MODEL")
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


def _build_model(config: dict[str, Any] | None) -> ChatOpenAI:
    return ChatOpenAI(
        model=_configured_model(config),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=_env_float("OPENAI_TEMPERATURE", "0.6"),
        top_p=_env_float("OPENAI_TOP_P", "0.95"),
        presence_penalty=_env_float("OPENAI_PRESENCE_PENALTY", "0.0"),
        timeout=_env_float("OPENAI_TIMEOUT", "120"),
        max_retries=2,
        max_completion_tokens=_env_int("OPENAI_MAX_TOKENS", "8192"),
        extra_body={
            "top_k": _env_int("OPENAI_TOP_K", "20"),
            "min_p": _env_float("OPENAI_MIN_P", "0"),
            "repetition_penalty": _env_float("OPENAI_REPETITION_PENALTY", "1.0"),
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class BaselineSidecar:
    def __init__(self, project_dir: Path, port: int | None = None) -> None:
        self.project_dir = project_dir
        self.port = port or _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.assistant_id: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._log = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="decomposer-baseline-sidecar-",
            suffix=".log",
            delete=False,
        )

    @property
    def log_path(self) -> str:
        return self._log.name

    def start(self) -> "BaselineSidecar":
        langgraph = Path(sys.executable).with_name("langgraph")
        if not langgraph.exists():
            msg = f"LangGraph CLI not found next to Python executable: {langgraph}"
            raise FileNotFoundError(msg)

        env = os.environ.copy()
        env["NO_PROXY"] = _append_no_proxy(env.get("NO_PROXY"), "127.0.0.1,localhost")
        env["no_proxy"] = _append_no_proxy(env.get("no_proxy"), "127.0.0.1,localhost")

        self._process = subprocess.Popen(
            [
                str(langgraph),
                "dev",
                "--config",
                "langgraph.json",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--no-browser",
                "--no-reload",
                "--allow-blocking",
            ],
            cwd=self.project_dir,
            env=env,
            text=True,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        self._wait_until_ready()
        return self

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        self._log.close()

    def _wait_until_ready(self, timeout: float = 90.0) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Baseline sidecar exited with code {self._process.returncode}. "
                    f"Log: {self.log_path}\n{_tail(self.log_path)}"
                )
            try:
                with urllib.request.urlopen(f"{self.url}/ok", timeout=2) as response:
                    if response.status == 200:
                        self.assistant_id = self._find_assistant_id()
                        return
            except Exception as exc:
                last_error = exc
            time.sleep(0.5)

        raise TimeoutError(
            f"Baseline sidecar did not become ready at {self.url}. "
            f"Last error: {last_error!r}. Log: {self.log_path}\n{_tail(self.log_path)}"
        )

    def _find_assistant_id(self) -> str:
        client = get_sync_client(
            url=self.url,
            headers={"x-auth-scheme": "langsmith"},
        )
        assistants = client.assistants.search()
        for assistant in assistants:
            if assistant.get("graph_id") == "baseline" or assistant.get("name") == "baseline":
                return str(assistant["assistant_id"])
        raise RuntimeError(f"No baseline assistant found at {self.url}: {assistants!r}")


def _append_no_proxy(existing: str | None, values: str) -> str:
    if not existing:
        return values
    parts = [part.strip() for part in existing.split(",") if part.strip()]
    for value in values.split(","):
        if value not in parts:
            parts.append(value)
    return ",".join(parts)


def _tail(path: str, limit: int = 4000) -> str:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as exc:
        return f"<failed to read log: {exc}>"
    return text[-limit:]


def _baseline_project_dir() -> Path:
    configured = os.environ.get("BASELINE_LANGGRAPH_PROJECT")
    project_dir = Path(configured) if configured else DEFAULT_BASELINE_PROJECT
    if not project_dir.exists():
        raise FileNotFoundError(f"Baseline LangGraph project not found: {project_dir}")
    return project_dir


@contextmanager
def make_decomposer(config: dict[str, Any] | None = None):
    sidecar = BaselineSidecar(
        _baseline_project_dir(),
        port=int(os.environ["BASELINE_SIDECAR_PORT"]) if os.environ.get("BASELINE_SIDECAR_PORT") else None,
    ).start()
    try:
        yield create_decomposer_agent(
            decomposer_model=_build_model(config),
            subagent_types=[
                {
                    "subagent_type_id": "baseline",
                    "assistant_id": sidecar.assistant_id or "",
                    "url": sidecar.url,
                    "description": (
                        "A Terminal-Bench baseline agent with a bash tool. It can inspect "
                        "and modify files in the task workspace."
                    ),
                }
            ],
        )
    finally:
        sidecar.stop()
