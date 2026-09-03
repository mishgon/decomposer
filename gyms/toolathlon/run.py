from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import posixpath
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import message_to_dict
from decomposer.chat_vllm import ChatVLLM
from decomposer.core import create_decomposer_agent
from decomposer.prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    DECOMPOSER_TEACHER_SYSTEM_PROMPT,
)

try:
    from .subagents.openrouter_compat import create_openrouter_model
    from .teacher_models import create_lmrouter_teacher
    from .usage import build_usage_summary
    from .settings import (
        DECOMPOSER_RECURSION_LIMIT,
        DEEPSEEK_REASONING_EFFORT,
        SUBAGENT_CONTEXT_TOKENS,
        SUBAGENT_RECURSION_LIMIT,
    )
except ImportError:  # Executed directly as ``python gyms/toolathlon/run.py``.
    from subagents.openrouter_compat import create_openrouter_model
    from teacher_models import create_lmrouter_teacher
    from usage import build_usage_summary
    from settings import (  # type: ignore[no-redef]
        DECOMPOSER_RECURSION_LIMIT,
        DEEPSEEK_REASONING_EFFORT,
        SUBAGENT_CONTEXT_TOKENS,
        SUBAGENT_RECURSION_LIMIT,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLATHLON_ROOT = REPO_ROOT / "external" / "toolathlon"
DEFAULT_BENCH_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "gyms" / "toolathlon"
DEFAULT_ARTIFACTS_DIR = DEFAULT_BENCH_ARTIFACTS_DIR / "traces"
DEFAULT_EVALS_DIR = DEFAULT_BENCH_ARTIFACTS_DIR / "evals"
DEFAULT_STASHES_DIR = DEFAULT_BENCH_ARTIFACTS_DIR / "stashes"
DEFAULT_IMAGE = "decomposer-toolathlon-bench:latest"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_SUBAGENT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SUBAGENT_PORT = 8030
DEFAULT_BASE_IMAGE = "docker.io/lockon0927/toolathlon-task-image:1016beta"
DEFAULT_EVAL_CONFIG = "scripts/formal_run_v0.json"
DEFAULT_MAX_STEPS = 200
DEFAULT_CONTAINER_SLEEP = 7200
PODMAN_SHORTNAMES_FILE = Path(__file__).with_name("podman-shortnames.conf")
PORT_LOCK_DIR = Path("/tmp/decomposer-toolathlon-port-locks")
CONTAINER_PODMAN_SHORTNAMES_FILE = "/workspace/configs/podman-shortnames.conf"
CONTAINER_DOCKER_API_VERSION = "1.44"
CONTAINER_TASK_ROOT_TEMPLATE = "/workspace/tasks/{task}"
TASK_PATH_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
SUBAGENT_TYPES = (
    # subagent_type_id, assistant_id, model_description
    ("qwen_3_5_4b_non_thinking", "qwen_3_5_4b_non_thinking", "Qwen-3.5-4B non-thinking"),
    ("gemma_4_e4b_thinking", "gemma_4_e4b_thinking", "Gemma-4-E4B thinking"),
    (
        "gemma_4_26b_a4b_non_thinking",
        "gemma_4_26b_a4b_non_thinking",
        "Gemma-4-26B-A4B non-thinking",
    ),
)
AGENT_MODES = ("simple", "decomposer")
SUBAGENT_PROVIDERS = ("vllm", "openrouter")
DECOMPOSER_PROVIDERS = ("openrouter", "vllm", "lmrouter")
DECOMPOSER_PROMPTS = {
    "student": DECOMPOSER_SYSTEM_PROMPT,
    "teacher": DECOMPOSER_TEACHER_SYSTEM_PROMPT,
}
# Untracked, user-provided files that the benchmark copies from its host
# checkout into the task container at run time.
USER_CONFIG_FILES = (
    "configs/global_configs.py",
    "configs/gcp-oauth.keys.json",
    "configs/google_credentials.json",
    "configs/token_key_session.py",
    "configs/notion_state.json",
    "configs/credentials.json",
    "configs/snowflake_rsa_key.p8",
    "configs/snowflake_rsa_key.pub",
)
K8S_TASK_CLEANUP_COMMANDS = {
    "finalpool/k8s-deployment-cleanup": (
        "bash",
        "/workspace/tasks/finalpool/k8s-deployment-cleanup/scripts/k8s_deployment_cleanup.sh",
        "stop",
    ),
    "finalpool/k8s-mysql": (
        "bash",
        "/workspace/tasks/finalpool/k8s-mysql/scripts/k8s_mysql.sh",
        "stop",
    ),
    "finalpool/k8s-pr-preview-testing": (
        "bash",
        "/workspace/tasks/finalpool/k8s-pr-preview-testing/scripts/k8s_pr_preview_testing.sh",
        "_",
        "stop",
    ),
    "finalpool/k8s-redis-helm-upgrade": (
        "bash",
        "/workspace/tasks/finalpool/k8s-redis-helm-upgrade/scripts/init_redis_helm.sh",
        "stop",
    ),
    "finalpool/k8s-safety-audit": (
        "bash",
        "/workspace/tasks/finalpool/k8s-safety-audit/scripts/k8s_safety_audit.sh",
        "stop",
    ),
}


DOCKER_SOCKET_CANDIDATES = (
    "/var/run/docker.sock",
    "/run/docker.sock",
    "/run/podman/podman.sock",
)
XDG_DOCKER_SOCKET_CANDIDATES = ("docker.sock", "podman/podman.sock")
DOCKER_COMMAND_TIMEOUT = 60.0
DOCKER_EXEC_TIMEOUT = 1800.0


class _TeeTextIO:
    def __init__(self, console, log) -> None:
        self.console = console
        self.log = log

    def write(self, value: str) -> int:
        self.console.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name: str):
        return getattr(self.console, name)


def git_provenance(root: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _docker(
    *args: str,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    effective_timeout = (
        timeout
        if timeout is not None
        else DOCKER_EXEC_TIMEOUT if args and args[0] == "exec" else DOCKER_COMMAND_TIMEOUT
    )
    try:
        process = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"docker {' '.join(args)} timed out after {effective_timeout:g}s"
        ) from error
    if check and process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise RuntimeError(
            f"docker {' '.join(args)} failed with exit code {process.returncode}"
            + (f": {detail}" if detail else "")
        )
    return process


def resolve_docker_socket(explicit: str | None = None) -> str:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise RuntimeError(f"Docker socket path must be absolute: {path}")
        # Docker resolves bind sources on the daemon host.  With Colima that
        # is its Linux VM, where /var/run/docker.sock is valid even though the
        # path does not exist on the macOS client.
        return str(path)
    docker_host = os.environ.get("DOCKER_HOST", "")
    if docker_host.startswith("unix://"):
        candidate = docker_host.removeprefix("unix://")
        if Path(candidate).exists():
            return candidate
    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    xdg_candidates = (
        [
            str(Path(xdg_runtime_dir) / relative)
            for relative in XDG_DOCKER_SOCKET_CANDIDATES
        ]
        if xdg_runtime_dir
        else []
    )
    # Prefer the current user's rootless engine.  On hosts with both rootful
    # and rootless Podman sockets, the Docker-compatible CLI normally uses the
    # latter; mounting the former would give tools in the task container a
    # different image/container store from the outer runner.
    for candidate in (*xdg_candidates, *DOCKER_SOCKET_CANDIDATES):
        if Path(candidate).exists():
            return candidate
    tried = [*xdg_candidates, *DOCKER_SOCKET_CANDIDATES]
    raise RuntimeError(
        "No Docker socket found (tried: "
        + ", ".join(tried)
        + "); pass --docker-socket or set DOCKER_HOST=unix:///path/to/docker.sock"
        + "; on rootless podman start the API service first: "
        + "systemctl --user enable --now podman.socket"
    )


def container_socket_mounts(docker_socket: str) -> list[str]:
    """Expose one engine socket at both Docker- and Podman-native paths."""
    return [
        "-v",
        f"{docker_socket}:/var/run/docker.sock",
        "-v",
        f"{docker_socket}:/run/podman/podman.sock",
    ]


def ensure_benchmark_checkout(toolathlon_root: Path) -> None:
    if not (toolathlon_root / "tasks").is_dir():
        raise RuntimeError(
            f"Toolathlon checkout at {toolathlon_root} is incomplete (missing "
            "tasks/); run: git submodule update --init external/toolathlon"
        )
    configs = toolathlon_root / "configs"
    global_configs = configs / "global_configs.py"
    example = configs / "global_configs_example.py"
    if not global_configs.is_file():
        if example.is_file():
            configs.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(example, global_configs)
            print(
                f"First-run setup: created {global_configs} from "
                "global_configs_example.py (fill in real credentials later; "
                "see the benchmark's global_preparation/how2register_accounts.md)",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"Missing {global_configs} and no global_configs_example.py to "
                "seed it from; the checkout is incomplete — run: git submodule "
                "update --init external/toolathlon"
            )
    token_config = configs / "token_key_session.py"
    token_example = configs / "token_key_session_example.py"
    if not token_config.is_file() and token_example.is_file():
        shutil.copyfile(token_example, token_config)
        print(
            f"First-run setup: created {token_config} from "
            "token_key_session_example.py (fill in credentials needed by the "
            "selected tasks)",
            flush=True,
        )
    (configs / ".mcp-auth").mkdir(parents=True, exist_ok=True)
    _warn_submodule_drift(toolathlon_root)


def _warn_submodule_drift(toolathlon_root: Path) -> None:
    try:
        listing = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "ls-tree",
                "HEAD",
                "external/toolathlon",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        pinned = listing[2] if len(listing) >= 3 else None
        current = subprocess.run(
            ["git", "-C", str(toolathlon_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return
    if pinned and current and not current.startswith(pinned):
        print(
            f"Warning: external/toolathlon is at {current[:12]} but the "
            f"repository pins {pinned[:12]}; the task image may not match the "
            "checkout.",
            flush=True,
        )


def _handle_termination(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def _open_container_lock(path: Path | None):
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a+")


def _acquire_container_lock(lock_file) -> None:
    if lock_file is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _release_container_lock(lock_file) -> None:
    if lock_file is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _exec_in_container(
    container: str,
    *command: str,
    env: dict[str, str] | None = None,
    detach: bool = False,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    args = ["exec"]
    if detach:
        args.append("--detach")
    for key, value in (env or {}).items():
        args.extend(["--env", f"{key}={value}"])
    if env is None or "DOCKER_API_VERSION" not in (env or {}):
        args.extend(["--env", f"DOCKER_API_VERSION={CONTAINER_DOCKER_API_VERSION}"])
    args.append(container)
    args.extend(command)
    return _docker(*args, check=check, timeout=timeout)


def served_subagent_model_name(model: str) -> str:
    model_lower = model.lower()
    if "gemma-4-26b-a4b" in model_lower:
        return "google/gemma-4-26B-A4B-it"
    if "gemma-4-e4b" in model_lower:
        return "google/gemma-4-E4B-it"
    return DEFAULT_SUBAGENT_MODEL


def vllm_command(
    model: str,
    port: int,
    *,
    max_model_len: int,
    gpu_memory_utilization: float,
    data_parallel_size: int = 1,
) -> list[str]:
    model_lower = model.lower()
    is_gemma = "gemma-4" in model_lower
    served_model = served_subagent_model_name(model)
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        model,
        "--served-model-name",
        served_model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "gemma4" if is_gemma else "qwen3_xml",
    ]
    if is_gemma:
        command.extend(["--reasoning-parser", "gemma4"])
    if not is_gemma or "gemma-4-26b-a4b" in model_lower:
        command.extend(
            ["--default-chat-template-kwargs", '{"enable_thinking":false}']
        )
    if data_parallel_size > 1:
        command.extend(
            [
                "--data-parallel-size",
                str(data_parallel_size),
                # A single frontend dispatches every turn across all DP
                # engines. The default one-frontend-per-engine topology can
                # pin a persistent episode connection to one GPU.
                "--api-server-count",
                "1",
            ]
        )
    return command


def wait_for_vllm(
    process: subprocess.Popen[bytes] | None,
    *,
    port: int,
    expected_model: str,
    timeout: float,
    log_path: Path,
) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {process.returncode}; inspect {log_path}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                models = json.load(response)["data"]
            if expected_model in {model["id"] for model in models}:
                return
            last_error = RuntimeError(f"{url} does not serve {expected_model}")
        except (OSError, KeyError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(
        f"vLLM did not become ready at {url} within {timeout:g}s; "
        f"inspect {log_path}"
    ) from last_error


def start_vllm(
    *,
    model: str,
    port: int,
    gpu: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    timeout: float,
    log_path: Path,
    reuse: bool,
    data_parallel_size: int = 1,
) -> subprocess.Popen[bytes] | None:
    no_proxy = {
        item
        for item in (
            os.environ.get("NO_PROXY", "") + "," + os.environ.get("no_proxy", "")
        ).split(",")
        if item
    }
    no_proxy.update({"127.0.0.1", "localhost", "host.docker.internal"})
    no_proxy_value = ",".join(sorted(no_proxy))
    os.environ["NO_PROXY"] = no_proxy_value
    os.environ["no_proxy"] = no_proxy_value

    if reuse:
        wait_for_vllm(
            None,
            port=port,
            expected_model=served_subagent_model_name(model),
            timeout=2,
            log_path=log_path,
        )
        return None

    reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        reservation.bind(("127.0.0.1", port))
    except OSError as error:
        reservation.close()
        raise RuntimeError(
            f"Refusing to start vLLM: port {port} is already in use; choose a "
            "dedicated --subagent-port or pass --reuse-vllm intentionally"
        ) from error

    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = vllm_command(
        model,
        port,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        data_parallel_size=data_parallel_size,
    )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "VLLM_ENGINE_READY_TIMEOUT_S": str(max(1, int(timeout))),
    }
    # vLLM/FlashInfer invoke helper executables such as ``ninja`` by name.
    # ``uv run`` can use the virtualenv interpreter without adding its bin
    # directory to PATH (notably in non-interactive SSH sessions), so preserve
    # the interpreter's toolchain explicitly for the server process.
    # Do not resolve the virtualenv's python symlink: helper executables such
    # as ninja live beside that symlink in .venv/bin, not beside the underlying
    # uv-managed interpreter target.
    executable_dir = str(Path(sys.executable).parent)
    environment["PATH"] = os.pathsep.join(
        part for part in (executable_dir, environment.get("PATH", "")) if part
    )
    with log_path.open("ab") as log:
        reservation.close()
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_for_vllm(
            process,
            port=port,
            expected_model=served_subagent_model_name(model),
            timeout=timeout,
            log_path=log_path,
        )
    except BaseException:
        stop_vllm(process)
        raise
    return process


def stop_vllm(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def allocate_port() -> tuple[int, socket.socket, object]:
    """Reserve a host port and lease it across concurrent episode processes."""
    PORT_LOCK_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        lock_file = (PORT_LOCK_DIR / f"{port}.lock").open("a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            sock.close()
            continue
        return port, sock, lock_file


def start_host_subagent_server(
    *,
    port: int,
    gateway_port: int,
    model: str,
    n_jobs_per_worker: int,
    log_path: Path,
    model_call_log_path: Path,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(Path(sys.executable).with_name("langgraph")),
        "dev",
        "--config",
        "langgraph.json",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--n-jobs-per-worker",
        str(n_jobs_per_worker),
        "--no-browser",
        "--no-reload",
    ]
    environment = {
        **os.environ,
        "TOOLATHLON_GATEWAY_URL": f"http://127.0.0.1:{gateway_port}/sse",
        "TOOLATHLON_OPENROUTER_MODEL": model,
        "TOOLATHLON_SUBAGENT_CALL_LOG": str(model_call_log_path),
    }
    with log_path.open("ab") as log:
        return subprocess.Popen(
            command,
            cwd=Path(__file__).with_name("subagents"),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_for_url(
    url: str,
    *,
    timeout: float,
    hint: str,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{hint} exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError(f"{hint} did not become ready at {url} within {timeout:g}s")


def wait_for_container_url(
    url: str,
    *,
    timeout: float,
    hint: str,
    container: str,
    pid: str,
) -> None:
    """Wait for a container service while failing fast if its process exits."""
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError as error:
            last_error = error
        alive = _exec_in_container(
            container, "kill", "-0", pid, check=False
        )
        if alive.returncode != 0:
            raise RuntimeError(f"{hint} process {pid} exited before becoming ready")
        time.sleep(1)
    raise TimeoutError(
        f"{hint} did not become ready at {url} within {timeout:g}s"
    ) from last_error


def missing_preprocess_runtime_files(container: str, bundle: dict) -> list[str]:
    """Return required runtime files that a successful preprocess omitted."""
    resolved = bundle.get("resolved_task_config") or {}
    token_config = resolved.get("local_token_key_session") or {}
    kubeconfig = token_config.get("kubeconfig_path")
    if not isinstance(kubeconfig, str) or not kubeconfig:
        return []
    probe = _exec_in_container(container, "test", "-s", kubeconfig, check=False)
    return [] if probe.returncode == 0 else [kubeconfig]


def wait_for_container_ready(container: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        running = _docker("ps", "-q", "--filter", f"name={container}", check=False)
        if not running.stdout.strip():
            logs = _docker("logs", container, check=False)
            raise RuntimeError(
                "Task container exited before becoming ready:\n"
                + logs.stdout
                + logs.stderr
            )
        probe = _docker("exec", container, "echo", "container ready", check=False)
        if probe.returncode == 0:
            return
        time.sleep(1)
    raise TimeoutError(f"Task container {container} was not ready within {timeout:g}s")


def _stage_bundle(container: str, host_bundle: Path) -> str:
    probe = _exec_in_container(
        container, "mktemp", "/run/toolathlon-decoupled-bundle.XXXXXX.json"
    )
    bundle_path = probe.stdout.strip()
    if not bundle_path:
        raise RuntimeError("Failed to allocate a private container bundle path")
    _docker("cp", str(host_bundle), f"{container}:{bundle_path}")
    _exec_in_container(container, "chmod", "600", bundle_path)
    return bundle_path


def _discard_bundle(container: str, bundle_path: str | None) -> None:
    if bundle_path:
        _exec_in_container(
            container, "rm", "-f", "--", bundle_path, check=False
        )


def _artifact_guard(
    action: str,
    *,
    container: str | None = None,
    task_path: str | None = None,
    stash_root: str | None = None,
    stash_dir: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.containerized.task_artifact_guard",
        action,
    ]
    if action in {"stash", "restore"}:
        command.extend(["--runtime", "docker", "--container", container or ""])
        command.extend(["--task-path", task_path or ""])
    if action == "stash" and stash_root:
        command.extend(["--stash-root", stash_root])
    if action in {"restore", "cleanup"} and stash_dir:
        command.extend(["--stash-dir", stash_dir])
    environment = {**os.environ, "PYTHONPATH": str(TOOLATHLON_ROOT)}
    return subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        check=check,
    )


def validate_bundle(bundle: dict, task_dir: str, host_output_folder: str) -> None:
    if bundle.get("schema_version") != 2:
        raise ValueError("preprocess produced an unsupported task bundle")
    if bundle.get("task_dir") != task_dir:
        raise ValueError("trusted bundle task_dir mismatch")
    resolved = bundle.get("resolved_task_config")
    if not isinstance(resolved, dict):
        raise ValueError("trusted bundle is missing resolved_task_config")
    container_paths = bundle.get("container_paths")
    host_paths = bundle.get("host_paths")
    if not isinstance(container_paths, dict) or not isinstance(host_paths, dict):
        raise ValueError("trusted bundle is missing phase paths")

    def require_normal_absolute(path: object, label: str, flavor: type) -> str:
        if not isinstance(path, str) or not flavor.isabs(path):
            raise ValueError(f"{label} must be absolute")
        if flavor.normpath(path) != path:
            raise ValueError(f"{label} must be normalized")
        return path

    container_root = require_normal_absolute(
        container_paths.get("task_root"), "container task root", posixpath
    )
    try:
        inside_workspace = (
            posixpath.commonpath(("/workspace", container_root)) == "/workspace"
        )
    except ValueError:
        inside_workspace = False
    if not inside_workspace:
        raise ValueError("container task root must be below /workspace")

    for key in ("agent_workspace", "log_file"):
        value = require_normal_absolute(
            container_paths.get(key), f"container {key}", posixpath
        )
        if posixpath.commonpath((container_root, value)) != container_root:
            raise ValueError(f"container {key} must be below the container task root")
        if resolved.get(key) != value:
            raise ValueError(f"resolved config {key} does not match phase paths")
    if resolved.get("task_root") != container_root:
        raise ValueError("resolved config task_root does not match phase paths")

    expected_host_root = os.path.abspath(host_output_folder)
    host_root = require_normal_absolute(
        host_paths.get("task_root"), "host task root", os.path
    )
    if host_root != expected_host_root:
        raise ValueError("trusted bundle host output root mismatch")
    for key in ("agent_workspace", "log_file"):
        value = require_normal_absolute(host_paths.get(key), f"host {key}", os.path)
        if os.path.commonpath((host_root, value)) != host_root:
            raise ValueError(f"host {key} must be below the host task root")


def _trajectory_messages(
    state: dict, subagent_runs: dict[str, dict]
) -> list[dict]:
    role_map = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}
    messages: list[dict] = []
    for message in state.get("messages", []):
        role = role_map.get(message.type, message.type)
        content = message.content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        entry: dict = {"role": role, "content": content}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"id": call.get("id"), "name": call.get("name"), "args": call.get("args")}
                for call in tool_calls
            ]
        messages.append(entry)
    for run in subagent_runs.values():
        for call in run.get("tool_calls", []):
            messages.append(
                {
                    "role": "subagent_tool_call",
                    "subagent_run_id": run["subagent_run_id"],
                    "content": json.dumps(
                        {"name": call["name"], "args": call["args"]},
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )
    return messages


def _trajectory_tool_names(
    state: dict, subagent_runs: dict[str, dict]
) -> list[str]:
    names: set[str] = set()
    for message in state.get("messages", []):
        for call in getattr(message, "tool_calls", None) or []:
            names.add(call.get("name") or "")
    for run in subagent_runs.values():
        for call in run.get("tool_calls", []):
            names.add(call["name"])
    return sorted(name for name in names if name)


def write_trajectory(
    episode_dir: Path,
    *,
    task: str,
    episode_id: str,
    status: str,
    state: dict | None,
    subagent_runs: dict[str, dict],
    started_at: str,
    resolved_task_config: dict | None,
    error: str | None = None,
) -> None:
    state = state or {}
    messages = _trajectory_messages(state, subagent_runs)
    ai_messages = [
        message for message in state.get("messages", []) if message.type == "ai"
    ]
    tool_calls = sum(
        len(getattr(message, "tool_calls", None) or []) for message in ai_messages
    )
    trajectory = {
        "config": resolved_task_config or {},
        "request_id": str(uuid.uuid4()),
        "task_dir": task,
        "initial_run_time": started_at,
        "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool_calls": {
            "tools": _trajectory_tool_names(state, subagent_runs),
            "tool_choice": "auto",
        },
        "status": status,
        "messages": messages,
        "key_stats": {
            "interaction_turns": len(ai_messages),
            "tool_calls": tool_calls,
            "agent_llm_requests": len(ai_messages),
            "subagent_runs": len(subagent_runs),
        },
        "agent_cost": {},
        "user_cost": {},
        "resumed": False,
        "session_id": episode_id,
        "history_file": None,
    }
    if error is not None:
        trajectory["error"] = error
    (episode_dir / "traj_log.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def evaluation_scores(
    eval_res: dict[str, Any], *, agent_success: bool
) -> tuple[bool | None, bool | None]:
    """Return strict benchmark score and artifact-only diagnostic score."""
    artifact_pass = eval_res.get("pass")
    if not isinstance(artifact_pass, bool):
        artifact_pass = None
    return artifact_pass if agent_success else None, artifact_pass


def _cleanup_episode(
    *, episode_dir: Path, task_container: str, task: str = ""
) -> None:
    cleanup: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "captures": [],
        "removals": [],
    }
    task_cleanup = K8S_TASK_CLEANUP_COMMANDS.get(task)
    if task_cleanup is not None:
        try:
            completed = _exec_in_container(
                task_container,
                *task_cleanup,
                env={
                    "CONTAINERS_REGISTRIES_CONF": CONTAINER_PODMAN_SHORTNAMES_FILE,
                },
                check=False,
            )
            cleanup["task_cleanup"] = {
                "command": task_cleanup,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except BaseException as error:
            cleanup["task_cleanup"] = {"command": task_cleanup, "error": repr(error)}
    for kind, command in (
        ("log", ("logs", task_container)),
        ("inspect.json", ("inspect", task_container)),
    ):
        try:
            completed = _docker(*command, check=False)
            content = completed.stdout + completed.stderr
            if content:
                (episode_dir / f"task.{kind}").write_text(content, encoding="utf-8")
            cleanup["captures"].append(
                {"container": "task", "kind": kind, "returncode": completed.returncode}
            )
        except BaseException as error:
            cleanup["captures"].append(
                {"container": "task", "kind": kind, "error": repr(error)}
            )
    try:
        completed = _docker(
            "rm", "--force", "--time", "0", task_container, check=False
        )
        cleanup["removals"].append(
            {"command": ("rm", "--force", "--time", "0", task_container),
             "returncode": completed.returncode,
             "stdout": completed.stdout, "stderr": completed.stderr}
        )
    except BaseException as error:
        cleanup["removals"].append(
            {
                "command": ("rm", "--force", "--time", "0", task_container),
                "error": repr(error),
            }
        )
    cleanup["finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        (episode_dir / "cleanup.json").write_text(
            json.dumps(cleanup, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except BaseException:
        pass


def _copy_user_configs(container: str) -> None:
    _docker(
        "cp",
        str(PODMAN_SHORTNAMES_FILE),
        f"{container}:{CONTAINER_PODMAN_SHORTNAMES_FILE}",
    )
    for relative in USER_CONFIG_FILES:
        source = (TOOLATHLON_ROOT / relative).resolve()
        if source.is_file():
            _docker("cp", str(source), f"{container}:/workspace/{relative}")
    gcp_keys = TOOLATHLON_ROOT / "configs" / "gcp-oauth.keys.json"
    gcp_credentials = TOOLATHLON_ROOT / "configs" / "google_credentials.json"
    if gcp_keys.is_file() and gcp_credentials.is_file():
        _exec_in_container(
            container,
            "bash",
            "-c",
            "for dir in ~/.gmail-mcp ~/.calendar-mcp; do "
            "mkdir -p $dir; "
            "cp ./configs/gcp-oauth.keys.json $dir/; "
            "cp ./configs/google_credentials.json $dir/credentials.json; "
            "done",
            check=False,
        )


async def _run_simple_agent(
    task: str,
    *,
    provider: str,
    model: str,
    gateway_port: int,
    subagent_port: int,
    recursion_limit: int,
) -> dict:
    os.environ["TOOLATHLON_GATEWAY_URL"] = (
        f"http://127.0.0.1:{gateway_port}/sse"
    )
    os.environ["QWEN_3_5_4B_BASE_URL"] = (
        f"http://127.0.0.1:{subagent_port}/v1"
    )
    os.environ["GEMMA_4_E4B_BASE_URL"] = (
        f"http://127.0.0.1:{subagent_port}/v1"
    )
    os.environ["GEMMA_4_26B_A4B_BASE_URL"] = (
        f"http://127.0.0.1:{subagent_port}/v1"
    )
    if __package__:
        from .subagents import graph, webapp
    else:
        from subagents import graph, webapp

    async with webapp.lifespan(webapp.app):
        if provider == "vllm":
            model_lower = model.lower()
            if "gemma-4-26b-a4b" in model_lower:
                agent = graph.gemma_4_26b_a4b_non_thinking()
            elif "gemma-4-e4b" in model_lower:
                agent = graph.gemma_4_e4b_thinking()
            else:
                agent = graph.qwen_3_5_4b_non_thinking()
        else:
            os.environ["TOOLATHLON_OPENROUTER_MODEL"] = model
            agent = graph.deepseek_openrouter()
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": recursion_limit},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Benchmark task path: domain/task_name")
    parser.add_argument("--episode-id", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--purpose", choices=("trace-generation", "evaluation"), required=True
    )
    parser.add_argument("--agent-mode", choices=AGENT_MODES, default="decomposer")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--decomposer-provider", choices=DECOMPOSER_PROVIDERS, default="openrouter"
    )
    parser.add_argument(
        "--decomposer-base-url", default="http://127.0.0.1:8040/v1"
    )
    parser.add_argument(
        "--decomposer-prompt",
        choices=tuple(DECOMPOSER_PROMPTS),
        default="teacher",
        help="System prompt used by the decomposer model (default: teacher).",
    )
    parser.add_argument(
        "--subagent-provider", choices=SUBAGENT_PROVIDERS, default="vllm"
    )
    parser.add_argument("--subagent-model", default=DEFAULT_SUBAGENT_MODEL)
    parser.add_argument("--subagent-port", type=int, default=DEFAULT_SUBAGENT_PORT)
    parser.add_argument(
        "--subagent-base-url",
        help="OpenAI-compatible subagent URL as seen from task containers.",
    )
    parser.add_argument("--subagent-gpu", default="0")
    parser.add_argument(
        "--vllm-max-model-len", type=int, default=SUBAGENT_CONTEXT_TOKENS
    )
    parser.add_argument(
        "--subagent-recursion-limit", type=int, default=SUBAGENT_RECURSION_LIMIT
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-data-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-startup-timeout", type=float, default=1800)
    parser.add_argument("--reuse-vllm", action="store_true")
    parser.add_argument(
        "--publish-service-ports",
        action="store_true",
        help="Publish container services to host loopback (required by Colima).",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--docker-socket",
        help="Host Docker socket mounted into the task container as /var/run/docker.sock",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--evals-dir", type=Path, default=DEFAULT_EVALS_DIR)
    parser.add_argument("--stashes-dir", type=Path, default=DEFAULT_STASHES_DIR)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--n-jobs-per-worker", type=int, default=16)
    parser.add_argument("--agent-timeout", type=float, default=2700)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--container-sleep", type=int, default=DEFAULT_CONTAINER_SLEEP)
    parser.add_argument("--container-lock-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.n_jobs_per_worker < 1:
        parser.error("--n-jobs-per-worker must be at least 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.subagent_recursion_limit < 1:
        parser.error("--subagent-recursion-limit must be at least 1")
    if args.agent_timeout <= 0:
        parser.error("--agent-timeout must be positive")
    if args.vllm_data_parallel_size < 1:
        parser.error("--vllm-data-parallel-size must be at least 1")
    visible_gpus = [item for item in args.subagent_gpu.split(",") if item]
    if (
        args.subagent_provider == "vllm"
        and len(visible_gpus) != args.vllm_data_parallel_size
    ):
        parser.error(
            "--subagent-gpu must list exactly --vllm-data-parallel-size GPU IDs"
        )
    if args.subagent_provider == "vllm" and args.vllm_data_parallel_size != 1:
        parser.error("Toolathlon evaluation permits exactly one GPU per model")
    ensure_benchmark_checkout(TOOLATHLON_ROOT)
    if not TASK_PATH_RE.fullmatch(args.task):
        raise ValueError(
            f"Unknown Toolathlon task {args.task!r}: expected domain/task_name"
        )
    tasks_root = (TOOLATHLON_ROOT / "tasks").resolve()
    task_dir = (tasks_root / args.task).resolve()
    if task_dir.parent.parent != tasks_root or not task_dir.is_dir():
        raise ValueError(f"Unknown Toolathlon task: {args.task!r}")
    needs_openrouter = args.subagent_provider == "openrouter" or (
        args.agent_mode == "decomposer" and args.decomposer_provider == "openrouter"
    )
    if needs_openrouter and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY for the selected model")
    if args.agent_mode == "decomposer" and args.decomposer_provider == "lmrouter":
        missing = [
            name
            for name in ("LLM_PROXY_URL", "LLM_PROXY_MASTER_KEY")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError("Set " + " and ".join(missing) + " for lmrouter")
    docker_socket = resolve_docker_socket(args.docker_socket)

    try:
        _docker("image", "inspect", args.image)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} (build it with gyms/toolathlon/build.sh)") from exc

    episode_id = args.episode_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:8]
    )
    episode_dir = args.artifacts_dir.resolve() / args.task / episode_id
    evaluation_path = (
        args.evals_dir.resolve() / args.task / episode_id / "result.json"
    )
    episode_dir.mkdir(parents=True)
    container = f"decomposer-bench-{args.task.replace('/', '-')}-{uuid.uuid4().hex[:8]}"
    container_task_path = CONTAINER_TASK_ROOT_TEMPLATE.format(task=args.task)
    started_at = datetime.now(timezone.utc).isoformat()
    vllm_log = (
        args.artifacts_dir.resolve().parent
        / "logs"
        / args.task
        / episode_id
        / "vllm.log"
    )
    trusted_stash_dir = (args.stashes_dir.resolve() / args.task / episode_id).resolve()
    trusted_stash_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(trusted_stash_dir, 0o700)
    trusted_bundle_file = trusted_stash_dir / "task_bundle.json"

    vllm_process = None
    if args.subagent_provider == "vllm":
        print(f"Starting vLLM on GPU {args.subagent_gpu}...", flush=True)
        vllm_process = start_vllm(
            model=args.subagent_model,
            port=args.subagent_port,
            gpu=args.subagent_gpu,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            timeout=args.vllm_startup_timeout,
            log_path=vllm_log,
            reuse=args.reuse_vllm,
            data_parallel_size=args.vllm_data_parallel_size,
        )

    runner_log_dir = vllm_log.parent
    runner_log_dir.mkdir(parents=True, exist_ok=True)
    runner_stdout_log = (runner_log_dir / "runner.stdout.log").open(
        "a", encoding="utf-8", buffering=1
    )
    runner_stderr_log = (runner_log_dir / "runner.stderr.log").open(
        "a", encoding="utf-8", buffering=1
    )
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeTextIO(original_stdout, runner_stdout_log)
    sys.stderr = _TeeTextIO(original_stderr, runner_stderr_log)

    container_lock = _open_container_lock(args.container_lock_file)
    container_lock_held = False
    gateway_port, gateway_sock, gateway_port_lock = None, None, None
    subagent_webapp_port, subagent_sock, subagent_port_lock = None, None, None
    subagent_server_process: subprocess.Popen[bytes] | None = None
    container_bundle_path: str | None = None
    stash_dir: str | None = None
    agent_exit_code = 1
    agent_state: dict | None = None
    subagent_runs: dict[str, dict] = {}
    try:
        _acquire_container_lock(container_lock)
        container_lock_held = container_lock is not None

        service_network_args = ["--network", "host"]
        if args.publish_service_ports:
            gateway_port, gateway_sock, gateway_port_lock = allocate_port()
            if gateway_sock is not None:
                gateway_sock.close()
                gateway_sock = None
            service_network_args = [
                "--publish", f"127.0.0.1:{gateway_port}:{gateway_port}"
            ]
            if args.agent_mode == "decomposer" and args.subagent_provider == "vllm":
                subagent_webapp_port, subagent_sock, subagent_port_lock = allocate_port()
                subagent_sock.close()
                subagent_sock = None
                service_network_args += [
                    "--publish",
                    f"127.0.0.1:{subagent_webapp_port}:{subagent_webapp_port}",
                ]

        print("Starting task environment...", flush=True)
        mounts: list[str] = [
            *container_socket_mounts(docker_socket),
            "-v", f"{episode_dir.resolve()}:/workspace/dumps",
            "-v", f"{episode_dir.resolve()}:/workspace/logs",
        ]
        mcp_auth_dir = (TOOLATHLON_ROOT / "configs" / ".mcp-auth").resolve()
        if mcp_auth_dir.is_dir():
            mounts.extend(["-v", f"{mcp_auth_dir}:/workspace/configs/.mcp-auth"])
        notion_patch = (
            TOOLATHLON_ROOT / "configs" / "notion-mcp-patches" / "notion-openapi.json"
        ).resolve()
        if notion_patch.is_file():
            mounts.extend(
                [
                    "-v",
                    (
                        f"{notion_patch}:"
                        "/workspace/node_modules/@notionhq/notion-mcp-server/"
                        "scripts/notion-openapi.json:ro"
                    ),
                ]
            )
        _docker(
            "run",
            "--detach",
            "--name",
            container,
            *service_network_args,
            *mounts,
            "-w",
            "/workspace",
            args.image,
            "sleep",
            str(args.container_sleep),
        )
        wait_for_container_ready(container, args.startup_timeout)

        _exec_in_container(
            container,
            "mkdir",
            "-p",
            f"/workspace/tasks/{args.task.rsplit('/', 1)[0]}",
        )
        _exec_in_container(container, "rm", "-rf", "--", container_task_path)
        _docker("cp", str(task_dir), f"{container}:/workspace/tasks/{args.task.rsplit('/', 1)[0]}/")
        _copy_user_configs(container)

        print("Running container preprocess...", flush=True)
        preprocess_attempts = 3 if args.task in K8S_TASK_CLEANUP_COMMANDS else 1
        preprocess_failure = "preprocess did not run"
        bundle = None
        for preprocess_attempt in range(1, preprocess_attempts + 1):
            preprocess_bundle_path = _exec_in_container(
                container, "mktemp", "/run/toolathlon-preprocess-bundle.XXXXXX.json"
            ).stdout.strip()
            if not preprocess_bundle_path:
                raise RuntimeError("Failed to allocate the preprocess bundle path")
            preprocess = _exec_in_container(
                container,
                "uv",
                "run",
                "python",
                "-m",
                "scripts.decoupled.container_preprocess",
                "--eval_config",
                args.eval_config,
                "--task_dir",
                args.task,
                "--max_steps_under_single_turn_mode",
                str(args.max_steps),
                "--model_short_name",
                "decomposer",
                "--provider",
                "unified",
                "--bundle_file",
                preprocess_bundle_path,
                "--host_output_folder",
                str(episode_dir.resolve()),
                "--debug",
                env={
                    "CONTAINERS_REGISTRIES_CONF": CONTAINER_PODMAN_SHORTNAMES_FILE,
                },
                check=False,
                timeout=600,
            )
            with (episode_dir / "preprocess.log").open(
                "a" if preprocess_attempt > 1 else "w", encoding="utf-8"
            ) as log:
                log.write(f"\n=== preprocess attempt {preprocess_attempt} ===\n")
                log.write(preprocess.stdout)
                log.write(preprocess.stderr)

            if preprocess.returncode == 0:
                _docker(
                    "cp",
                    f"{container}:{preprocess_bundle_path}",
                    str(trusted_bundle_file),
                )
                os.chmod(trusted_bundle_file, 0o600)
                candidate = json.loads(
                    trusted_bundle_file.read_text(encoding="utf-8")
                )
                validate_bundle(candidate, args.task, str(episode_dir.resolve()))
                missing = missing_preprocess_runtime_files(container, candidate)
                if not missing:
                    bundle = candidate
                    _exec_in_container(
                        container, "rm", "-f", "--", preprocess_bundle_path
                    )
                    break
                preprocess_failure = (
                    "preprocess reported success but omitted required runtime "
                    f"files: {', '.join(missing)}"
                )
            else:
                preprocess_failure = (
                    f"preprocess exited with code {preprocess.returncode}"
                )

            _exec_in_container(
                container, "rm", "-f", "--", preprocess_bundle_path, check=False
            )
            if preprocess_attempt < preprocess_attempts:
                task_cleanup = K8S_TASK_CLEANUP_COMMANDS.get(args.task)
                if task_cleanup:
                    _exec_in_container(
                        container,
                        *task_cleanup,
                        env={
                            "CONTAINERS_REGISTRIES_CONF": (
                                CONTAINER_PODMAN_SHORTNAMES_FILE
                            ),
                        },
                        check=False,
                    )
                _exec_in_container(
                    container,
                    "rm",
                    "-rf",
                    "--",
                    "/workspace/dumps/workspace",
                    check=False,
                )
                time.sleep(5)

        if bundle is None:
            raise RuntimeError(
                f"Container preprocess failed after {preprocess_attempts} attempt(s): "
                f"{preprocess_failure}; inspect {episode_dir / 'preprocess.log'}"
            )
        container_paths = bundle["container_paths"]
        eval_result_path = posixpath.join(
            posixpath.dirname(container_paths["log_file"]), "eval_res.json"
        )

        print("Hiding evaluator and ground-truth artifacts...", flush=True)
        stash_root = trusted_stash_dir / "artifacts"
        stash_result = _artifact_guard(
            "stash",
            container=container,
            task_path=container_task_path,
            stash_root=str(stash_root),
        )
        stash_dir = stash_result.stdout.strip()
        if not stash_dir:
            raise RuntimeError("Artifact guard stash did not report a stash directory")

        print("Starting container MCP gateway...", flush=True)
        gateway_failure: BaseException | None = None
        for gateway_attempt in range(1, 6):
            if gateway_port is None:
                gateway_port, gateway_sock, gateway_port_lock = allocate_port()
            container_bundle_path = _stage_bundle(container, trusted_bundle_file)
            gateway_sock.close()
            gateway_sock = None
            launch = _exec_in_container(
                container,
                "bash",
                "-c",
                (
                    "nohup uv run python -m scripts.decoupled.container_tool_gateway "
                    f"--bundle_file {container_bundle_path} "
                    f"--host 0.0.0.0 --port {gateway_port} --debug "
                    f"> /workspace/dumps/gateway.log 2>&1 & echo $!"
                ),
            )
            gateway_pid = launch.stdout.strip()
            try:
                if not gateway_pid.isdigit():
                    raise RuntimeError(
                        f"gateway did not report a process id: {gateway_pid!r}"
                    )
                wait_for_container_url(
                    f"http://127.0.0.1:{gateway_port}/health",
                    timeout=120,
                    hint=f"Container MCP gateway for {container}",
                    container=container,
                    pid=gateway_pid,
                )
                gateway_failure = None
                break
            except BaseException as error:
                gateway_failure = error
                _exec_in_container(
                    container, "kill", gateway_pid, check=False
                )
                if gateway_port_lock is not None:
                    gateway_port_lock.close()
                gateway_port = None
                gateway_port_lock = None
                if gateway_attempt == 5:
                    raise RuntimeError(
                        "Container MCP gateway failed to start after 5 attempts"
                    ) from error
            finally:
                _discard_bundle(container, container_bundle_path)
                container_bundle_path = None
        if gateway_failure is not None:
            raise gateway_failure
        print(f"Gateway is ready: http://127.0.0.1:{gateway_port}/sse", flush=True)

        if args.agent_mode == "decomposer":
            print("Starting subagent server...", flush=True)
            if subagent_webapp_port is None:
                subagent_webapp_port, subagent_sock, subagent_port_lock = allocate_port()
                subagent_sock.close()
                subagent_sock = None
            if args.subagent_provider == "openrouter":
                subagent_server_process = start_host_subagent_server(
                    port=subagent_webapp_port,
                    gateway_port=gateway_port,
                    model=args.subagent_model,
                    n_jobs_per_worker=args.n_jobs_per_worker,
                    log_path=episode_dir / "subagent_server.log",
                    model_call_log_path=episode_dir / "subagent_model_calls.jsonl",
                )
            else:
                _exec_in_container(
                    container,
                    "bash",
                    "-c",
                    "exec bash /opt/decomposer/gyms/toolathlon/subagents/serve.sh "
                    ">> /workspace/dumps/subagent_server.log 2>&1",
                    detach=True,
                    env={
                        "QWEN_3_5_4B_BASE_URL": (
                            args.subagent_base_url
                            or f"http://127.0.0.1:{args.subagent_port}/v1"
                        ),
                        "GEMMA_4_E4B_BASE_URL": (
                            args.subagent_base_url
                            or f"http://127.0.0.1:{args.subagent_port}/v1"
                        ),
                        "TOOLATHLON_GATEWAY_URL": f"http://127.0.0.1:{gateway_port}/sse",
                        "HOST": "0.0.0.0",
                        "PORT": str(subagent_webapp_port),
                        "N_JOBS_PER_WORKER": str(args.n_jobs_per_worker),
                        "TOOLATHLON_SUBAGENT_CALL_LOG": (
                            "/workspace/dumps/subagent_model_calls.jsonl"
                        ),
                    },
                )
            wait_for_url(
                f"http://127.0.0.1:{subagent_webapp_port}/ok",
                timeout=args.startup_timeout,
                hint=f"Subagent server for {container}",
                process=subagent_server_process,
            )
            print(
                f"Subagent server is ready: http://127.0.0.1:{subagent_webapp_port}",
                flush=True,
            )

        _release_container_lock(container_lock)
        container_lock_held = False

        agent_error: str | None = None
        try:
            if args.agent_mode == "simple":
                print(f"Running simple {args.subagent_model} agent...", flush=True)
                os.environ["TOOLATHLON_SUBAGENT_CALL_LOG"] = str(
                    episode_dir / "subagent_model_calls.jsonl"
                )
                agent_state = asyncio.run(
                    asyncio.wait_for(
                        _run_simple_agent(
                            bundle["task_str"],
                            provider=args.subagent_provider,
                            model=args.subagent_model,
                            gateway_port=gateway_port,
                            subagent_port=args.subagent_port,
                            recursion_limit=args.subagent_recursion_limit,
                        ),
                        timeout=args.agent_timeout,
                    )
                )
            else:
                print("Running Decomposer...", flush=True)
                agent = create_decomposer_agent(
                    decomposer_model=(
                        ChatVLLM(
                            model=args.model,
                            api_key="EMPTY",
                            base_url=args.decomposer_base_url,
                            temperature=1,
                            top_p=1,
                            timeout=180,
                            max_retries=5,
                            preserve_reasoning="gemma-4" in args.model.lower(),
                        )
                        if args.decomposer_provider == "vllm"
                        else create_lmrouter_teacher(
                            model=args.model,
                            timeout=180,
                            max_retries=5,
                        )
                        if args.decomposer_provider == "lmrouter"
                        else create_openrouter_model(
                            model=args.model,
                            reasoning={"effort": DEEPSEEK_REASONING_EFFORT},
                            # A single stalled OpenRouter response must not consume
                            # ten minutes and fail the whole episode. DeepSeek's
                            # normal responses are comfortably below this bound;
                            # more retry opportunities handle transient backend
                            # stalls without changing the episode wall-clock cap.
                            timeout=180,
                            max_retries=5,
                        )
                    ),
                    subagent_types=[
                        {
                            "subagent_type_id": subagent_type_id,
                            "description": (
                                f"Tool-calling agent based on a {model_description} model. "
                                "Has access to all the available tools."
                            ),
                            "assistant_id": assistant_id,
                            "url": f"http://127.0.0.1:{subagent_webapp_port}",
                        }
                        for subagent_type_id, assistant_id, model_description in (
                            tuple(
                                spec
                                for spec in SUBAGENT_TYPES
                                if (
                                    ("gemma" in spec[0])
                                    == ("gemma-4" in args.subagent_model.lower())
                                )
                            )
                            if args.subagent_provider == "vllm"
                            else (("deepseek_openrouter", "deepseek_openrouter", args.subagent_model),)
                        )
                    ],
                    decomposer_system_prompt=DECOMPOSER_PROMPTS[
                        args.decomposer_prompt
                    ],
                    # LangGraph counts model and tool nodes separately. Match
                    # the allowance used by the direct/simple agent so a
                    # delegated Qwen run gets args.max_steps actual turns.
                    subagent_recursion_limit=args.subagent_recursion_limit,
                )
                agent_state = asyncio.run(
                    asyncio.wait_for(
                        agent.ainvoke(
                            {"messages": [{"role": "user", "content": bundle["task_str"]}]},
                            config={"recursion_limit": DECOMPOSER_RECURSION_LIMIT},
                        ),
                        timeout=args.agent_timeout,
                    )
                )
        except BaseException as error:
            agent_error = repr(error)
            if agent_state is None:
                agent_state = {}
        messages = agent_state.get("messages", [])
        subagent_runs = agent_state.get("subagent_runs", {}) or {}
        answer = ""
        for message in reversed(messages):
            if message.type == "ai":
                content = message.content
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "")
                        if isinstance(part, dict)
                        else str(part)
                        for part in content
                    )
                answer = str(content)
                break
        agent_success = agent_error is None and bool(answer.strip())
        agent_exit_code = 0 if agent_success else 1
        serialized_messages = [message_to_dict(message) for message in messages]
        usage = build_usage_summary(serialized_messages, subagent_runs)

        (episode_dir / "trace.json").write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "run_id": args.run_id,
                    "task": args.task,
                    "repetition": args.repetition,
                    "attempt": args.attempt,
                    "purpose": args.purpose,
                    "code": git_provenance(REPO_ROOT),
                    "agent_mode": args.agent_mode,
                    "decomposer_model": args.model,
                    "decomposer_provider": args.decomposer_provider,
                    "decomposer_reasoning_effort": (
                        DEEPSEEK_REASONING_EFFORT
                        if args.decomposer_provider == "openrouter"
                        else None
                    ),
                    "decomposer_thinking_enabled": (
                        True
                        if args.decomposer_provider == "lmrouter"
                        else "gemma-4" in args.model.lower()
                        if args.decomposer_provider == "vllm"
                        else None
                    ),
                    "decomposer_recursion_limit": DECOMPOSER_RECURSION_LIMIT,
                    "subagent_model": args.subagent_model,
                    "subagent_provider": args.subagent_provider,
                    "subagent_base_url": (
                        args.subagent_base_url
                        or f"http://127.0.0.1:{args.subagent_port}/v1"
                    ),
                    "subagent_context_tokens": args.vllm_max_model_len,
                    "subagent_recursion_limit": args.subagent_recursion_limit,
                    "deepseek_reasoning_effort": (
                        DEEPSEEK_REASONING_EFFORT
                        if "deepseek" in (args.model + " " + args.subagent_model).lower()
                        else None
                    ),
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "agent_error": agent_error,
                    "messages": serialized_messages,
                    "subagent_runs": subagent_runs,
                    "usage": usage,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        (episode_dir / "usage.json").write_text(
            json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (episode_dir / "answer.txt").write_text(answer, encoding="utf-8")
        write_trajectory(
            episode_dir,
            task=args.task,
            episode_id=episode_id,
            status="success" if agent_success else "failed",
            state=agent_state,
            subagent_runs=subagent_runs,
            started_at=started_at,
            resolved_task_config=bundle.get("resolved_task_config"),
            error=agent_error,
        )

        print("Restoring trusted evaluator artifacts...", flush=True)
        if stash_dir:
            restore = _artifact_guard(
                "restore",
                container=container,
                task_path=container_task_path,
                stash_dir=stash_dir,
                check=False,
            )
            if restore.returncode != 0:
                raise RuntimeError(
                    "Failed to restore trusted evaluator artifacts; refusing to grade:\n"
                    + restore.stdout
                    + restore.stderr
                )
            _artifact_guard("cleanup", stash_dir=stash_dir, check=False)
            stash_dir = None

        print("Running native evaluation...", flush=True)
        _exec_in_container(
            container, "rm", "-rf", "--", eval_result_path, check=False
        )
        container_bundle_path = _stage_bundle(container, trusted_bundle_file)
        try:
            eval_run = _exec_in_container(
                container,
                "uv",
                "run",
                "python",
                "-m",
                "scripts.decoupled.container_eval",
                "--bundle_file",
                container_bundle_path,
                "--require_resolved_task_config",
                "--consume_bundle",
                "--agent_exit_code",
                # Grade the artifacts even when the agent loop timed out or
                # otherwise failed.  The host keeps that failure authoritative
                # below and exposes this evaluator result only as diagnostic
                # ``artifact_pass``; it never becomes the official pass score.
                "0",
                check=False,
            )
        finally:
            _discard_bundle(container, container_bundle_path)
            container_bundle_path = None
        with (episode_dir / "eval.log").open("w", encoding="utf-8") as log:
            log.write(eval_run.stdout)
            log.write(eval_run.stderr)

        eval_result_file = episode_dir / "eval_res.json"
        if eval_result_file.is_file():
            eval_res = json.loads(eval_result_file.read_text(encoding="utf-8"))
        else:
            eval_res = {
                "pass": None,
                "details": (
                    "evaluator produced no eval_res.json; "
                    f"exit code {eval_run.returncode}"
                ),
            }

        # container_eval temporarily promotes the trajectory to SUCCESS so the
        # native evaluator does not short-circuit.  Restore the real agent
        # status in the persisted trace after grading.
        if not agent_success:
            trajectory_path = episode_dir / "traj_log.json"
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            trajectory["status"] = "failed"
            if agent_error is not None:
                trajectory["error"] = agent_error
            trajectory_path.write_text(
                json.dumps(trajectory, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        trusted_bundle_file.replace(episode_dir / "task_bundle.json")
        os.chmod(episode_dir / "task_bundle.json", 0o600)

        official_pass, artifact_pass = evaluation_scores(
            eval_res, agent_success=agent_success
        )
        evaluation = {
            "episode_id": episode_id,
            "task": args.task,
            # Official Toolathlon metrics require a successful agent finish.
            "pass": official_pass,
            # Diagnostic only: whether the artifacts present at termination
            # satisfy the native evaluator, including after timeouts.
            "artifact_pass": artifact_pass,
            "agent_success": agent_success,
            "agent_error": agent_error,
            "details": eval_res.get("details"),
            "failure": eval_res.get("failure"),
            "agent_exit_code": agent_exit_code,
            "evaluation_exit_code": eval_run.returncode,
            "stdout": eval_run.stdout,
            "stderr": eval_run.stderr,
        }
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(answer)
        print(f"\nArtifacts: {episode_dir}")
        print(
            f"Evaluation: {evaluation['pass']} "
            f"(artifact_pass={evaluation['artifact_pass']}, {evaluation_path})"
        )

        # The evaluation artifacts above are preserved, but a failed agent
        # loop must still fail the episode, matching the benchmark's exit
        # code semantics (host loop failure overrides a passing eval).
        if not agent_success:
            raise RuntimeError(
                f"{args.agent_mode.capitalize()} agent loop failed: "
                + (agent_error or "the agent produced no final answer")
            )
    finally:
        if container_lock_held:
            _release_container_lock(container_lock)
            container_lock_held = False
        _acquire_container_lock(container_lock)
        if gateway_sock is not None:
            gateway_sock.close()
        if subagent_sock is not None:
            subagent_sock.close()
        if gateway_port_lock is not None:
            gateway_port_lock.close()
        if subagent_port_lock is not None:
            subagent_port_lock.close()
        artifacts_stash = trusted_stash_dir / "artifacts"
        if artifacts_stash.exists():
            shutil.rmtree(artifacts_stash, ignore_errors=True)
        print("Cleaning up...", flush=True)
        try:
            _cleanup_episode(
                episode_dir=episode_dir,
                task_container=container,
                task=args.task,
            )
        finally:
            _release_container_lock(container_lock)
            if container_lock is not None:
                container_lock.close()
            stop_vllm(subagent_server_process)
            stop_vllm(vllm_process)
            sys.stdout, sys.stderr = original_stdout, original_stderr
            runner_stdout_log.close()
            runner_stderr_log.close()


if __name__ == "__main__":
    import batch

    signal.signal(signal.SIGTERM, _handle_termination)
    if batch.wants_batch(sys.argv[1:]):
        batch.main(
            sys.argv[1:],
            repo_root=REPO_ROOT,
            toolathlon_root=TOOLATHLON_ROOT,
            default_artifacts_dir=DEFAULT_BENCH_ARTIFACTS_DIR,
            default_image=DEFAULT_IMAGE,
            default_model=DEFAULT_MODEL,
            default_subagent_model=DEFAULT_SUBAGENT_MODEL,
            default_subagent_port=DEFAULT_SUBAGENT_PORT,
            start_vllm=start_vllm,
            stop_vllm=stop_vllm,
            docker=_docker,
        )
    else:
        main()
