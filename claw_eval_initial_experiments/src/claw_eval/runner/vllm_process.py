"""Launch and manage local vLLM OpenAI-compatible servers."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .vllm_readiness import normalize_vllm_base_url, wait_for_vllm_targets


DEFAULT_VLLM_TOOL_ARGS = [
    ("--enable-auto-tool-choice", []),
    ("--tool-call-parser", ["hermes"]),
    ("--gdn-prefill-backend", ["triton"]),
]

DEFAULT_VLLM_REASONING_ARGS = [
    ("--reasoning-parser", ["deepseek_r1"]),
]


@dataclass(frozen=True)
class VllmServerSpec:
    """Configuration for one role-specific vLLM server."""

    role: str
    model_id: str
    port: int
    gpu: str | None = None
    host: str = "127.0.0.1"
    api_key: str = "unused"
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def log_name(self) -> str:
        safe_role = self.role.replace("/", "_")
        safe_model = self.model_id.replace("/", "_")
        return f"{safe_role}_{safe_model}_{self.port}.log"

    def command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model",
            self.model_id,
            "--served-model-name",
            self.model_id,
        ]
        cmd.extend(self.extra_args)
        return cmd


@dataclass
class VllmServerHandle:
    spec: VllmServerSpec
    process: subprocess.Popen | None
    log_path: Path | None
    reused: bool = False


class VllmProcessGroup:
    """Owns the vLLM processes launched by this command."""

    def __init__(self, handles: list[VllmServerHandle], *, stop_on_exit: bool) -> None:
        self.handles = handles
        self.stop_on_exit = stop_on_exit

    def close(self) -> None:
        if not self.stop_on_exit:
            return
        for handle in self.handles:
            if handle.reused or handle.process is None:
                continue
            proc = handle.process
            if proc.poll() is not None:
                continue
            print(f"[vllm] stopping {handle.spec.role} server (pid={proc.pid})")
            stop_process_group(proc.pid)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait(timeout=10)


def _role_env_prefix(role: str) -> str:
    if role == "model":
        return "VLLM"
    return f"VLLM_{role.upper()}"


def apply_vllm_env(specs: list[VllmServerSpec]) -> None:
    """Set env vars so config loading resolves to the launched vLLM servers."""

    os.environ["VLLM_ENABLED"] = "1"
    for spec in specs:
        prefixes = [_role_env_prefix(spec.role)]
        if spec.role == "model":
            prefixes.append("VLLM_MODEL")
        for prefix in prefixes:
            os.environ[f"{prefix}_BASE_URL"] = spec.base_url
            os.environ[f"{prefix}_MODEL_ID"] = spec.model_id
            os.environ[f"{prefix}_API_KEY"] = spec.api_key
        if spec.role == "model":
            # Generic model vars are what single-server flat runs already use.
            os.environ["VLLM_BASE_URL"] = spec.base_url
            os.environ["VLLM_MODEL_ID"] = spec.model_id
            os.environ["VLLM_API_KEY"] = spec.api_key


def make_vllm_subprocess_env(spec: VllmServerSpec) -> dict[str, str]:
    """Build subprocess env for a launched local vLLM server."""

    env = os.environ.copy()
    if spec.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)
    # Small local models can emit malformed partial tool-call JSON with huge
    # numeric literals. vLLM parses that server-side during streaming.
    env.setdefault("PYTHONINTMAXSTRDIGITS", "0")
    return env


def is_vllm_model_ready(base_url: str, model_id: str, api_key: str = "unused") -> bool:
    """Return True when a live vLLM endpoint lists the requested model."""

    base_url = normalize_vllm_base_url(base_url)
    try:
        resp = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        )
    except Exception:
        return False
    if resp.status_code != 200:
        return False
    try:
        payload = resp.json()
    except Exception:
        return False
    data = payload.get("data", [])
    if not isinstance(data, list):
        return False
    model_ids = {item.get("id") for item in data if isinstance(item, dict)}
    if model_id in model_ids:
        return True
    tail = model_id.rsplit("/", 1)[-1]
    return any(tail in (mid or "") for mid in model_ids)


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _server_extra_args(
    *,
    role: str,
    max_model_len: str | None,
    gpu_memory_utilization: str | None,
    extra_args: list[str] | None,
) -> list[str]:
    user_args = list(extra_args or [])
    args: list[str] = []
    if max_model_len:
        args.extend(["--max-model-len", str(max_model_len)])
    if gpu_memory_utilization:
        args.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    default_args = list(DEFAULT_VLLM_TOOL_ARGS)
    default_args.extend(DEFAULT_VLLM_REASONING_ARGS)
    for option, values in default_args:
        if not _has_option(user_args, option):
            args.append(option)
            args.extend(values)
    args.extend(user_args)
    return args


def make_vllm_server_spec(
    *,
    role: str,
    model_id: str,
    port: int,
    gpu: str | None,
    host: str,
    api_key: str = "unused",
    max_model_len: str | None = None,
    gpu_memory_utilization: str | None = None,
    extra_args: list[str] | None = None,
) -> VllmServerSpec:
    """Build a role-specific vLLM server spec with shared serving options."""

    return VllmServerSpec(
        role=role,
        model_id=model_id,
        port=port,
        gpu=gpu,
        host=host,
        api_key=api_key,
        extra_args=_server_extra_args(
            role=role,
            max_model_len=max_model_len or os.environ.get("VLLM_MAX_MODEL_LEN"),
            gpu_memory_utilization=gpu_memory_utilization or os.environ.get("VLLM_GPU_MEMORY_UTILIZATION"),
            extra_args=extra_args,
        ),
    )


def ensure_vllm_servers(
    specs: list[VllmServerSpec],
    *,
    log_dir: str | Path = "logs/vllm",
    timeout_s: int = 600,
    poll_s: float = 5.0,
    check_health: bool = True,
    stop_on_exit: bool = False,
) -> VllmProcessGroup:
    """Launch missing vLLM servers, then wait until all requested models are ready."""

    apply_vllm_env(specs)
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    handles: list[VllmServerHandle] = []

    for spec in specs:
        if is_vllm_model_ready(spec.base_url, spec.model_id, spec.api_key):
            print(f"[vllm] reusing ready {spec.role}: {spec.model_id} at {spec.base_url}")
            handles.append(VllmServerHandle(spec=spec, process=None, log_path=None, reused=True))
            continue

        log_path = log_root / spec.log_name
        env = make_vllm_subprocess_env(spec)
        fh = open(log_path, "ab", buffering=0)
        cmd = spec.command()
        print(
            f"[vllm] launching {spec.role}: {spec.model_id} "
            f"on gpu={spec.gpu or 'all'} port={spec.port} log={log_path}"
        )
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        fh.close()
        handles.append(VllmServerHandle(spec=spec, process=proc, log_path=log_path))
        pid_path = log_path.with_suffix(".pid")
        pid_path.write_text(str(proc.pid), encoding="utf-8")
        # Give immediate import/configuration failures a chance to surface.
        time.sleep(1.0)
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM {spec.role} server exited early with code {proc.returncode}. "
                f"See log: {log_path}"
            )

    targets = [(spec.base_url, spec.api_key, spec.model_id) for spec in specs]
    wait_for_vllm_targets(
        targets,
        timeout_s=timeout_s,
        poll_s=poll_s,
        check_health=check_health,
    )
    return VllmProcessGroup(handles, stop_on_exit=stop_on_exit)


def stop_process_group(pid: int) -> None:
    """Best-effort stop for a launched process group."""

    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        return
