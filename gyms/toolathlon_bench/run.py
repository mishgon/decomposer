from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import message_to_dict

from gyms.agent_server import invoke_and_capture
from decomposer.visualization import write_trace_html

try:
    from .task import PID_FILE
    from .usage import build_usage_summary
except ImportError:  # Executed directly as a script.
    from task import PID_FILE
    from usage import build_usage_summary


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLATHLON_ROOT = REPO_ROOT / "external" / "toolathlon"
DEFAULT_GYM_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "gyms" / "toolathlon_bench"
DEFAULT_ARTIFACTS_DIR = DEFAULT_GYM_ARTIFACTS_DIR / "traces"
DEFAULT_EVALS_DIR = DEFAULT_GYM_ARTIFACTS_DIR / "evals"
DEFAULT_IMAGE = "decomposer-toolathlon-bench:latest"
EVAL_CONFIG = "scripts/formal_run_v0.json"
MAX_STEPS = 200
CONTAINER_BUNDLE = "/run/decomposer-task-bundle.json"
PORT_LOCK_DIR = Path(tempfile.gettempdir()) / "toolathlon-bench-ports"
# Kind in Kubernetes tasks drives the host engine through the image's older
# docker CLI, at the Docker or Podman socket path Toolathlon is configured for.
DOCKER_SOCKET = Path("/var/run/docker.sock")
CONTAINER_ENGINE_SOCKETS = ("/var/run/docker.sock", "/run/podman/podman.sock")
DOCKER_API_VERSION = "1.44"
# Untracked, user-provided credentials that Toolathlon reads from configs/.
USER_CONFIG_FILES = (
    "configs/global_configs.py",
    "configs/gcp-oauth.keys.json",
    "configs/gcp-service_account.keys.json",
    "configs/google_credentials.json",
    "configs/token_key_session.py",
    "configs/notion_state.json",
    "configs/credentials.json",
    "configs/snowflake_rsa_key.p8",
    "configs/snowflake_rsa_key.pub",
)
# Gmail and Calendar MCP servers read Google OAuth files from their home directories.
GOOGLE_MCP_DIRS = ("/root/.gmail-mcp", "/root/.calendar-mcp")
GOOGLE_MCP_FILES = {
    "configs/gcp-oauth.keys.json": "gcp-oauth.keys.json",
    "configs/google_credentials.json": "credentials.json",
}
# Kubernetes tasks create Kind clusters on the host engine; their stop
# scripts delete them. Cluster creation is flaky, so preprocess is retried.
K8S_PREPROCESS_ATTEMPTS = 3
K8S_TASK_CLEANUP_COMMANDS = {
    "k8s-deployment-cleanup": (
        "bash", "tasks/finalpool/k8s-deployment-cleanup/scripts/k8s_deployment_cleanup.sh", "stop",
    ),
    "k8s-mysql": ("bash", "tasks/finalpool/k8s-mysql/scripts/k8s_mysql.sh", "stop"),
    "k8s-pr-preview-testing": (
        "bash", "tasks/finalpool/k8s-pr-preview-testing/scripts/k8s_pr_preview_testing.sh", "_", "stop",
    ),
    "k8s-redis-helm-upgrade": (
        "bash", "tasks/finalpool/k8s-redis-helm-upgrade/scripts/init_redis_helm.sh", "stop",
    ),
    "k8s-safety-audit": (
        "bash", "tasks/finalpool/k8s-safety-audit/scripts/k8s_safety_audit.sh", "stop",
    ),
}
# Some preprocess scripts exit zero after failing to reach a service.
PREPROCESS_FATAL_OUTPUT_PATTERNS = (
    re.compile(r"connection refused", re.IGNORECASE),
    re.compile(r"course setup encountered errors", re.IGNORECASE),
)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
    )
    if check and process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise RuntimeError(
            f"docker {' '.join(args)} failed with exit code {process.returncode}"
            + (f": {detail}" if detail else "")
        )
    return process


def _exec(container: str, *command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _docker("exec", "--workdir", "/workspace", container, *command, check=check)


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


def _reserve_port():
    """Reserve a free host port for the episode until the returned file is closed.

    Episodes share the host network, and a server may start on its port long
    after the port was chosen, so a port that looks free can be taken already.
    """
    PORT_LOCK_DIR.mkdir(exist_ok=True)
    while True:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        reservation = (PORT_LOCK_DIR / f"{port}.lock").open("w")
        try:
            fcntl.flock(reservation.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            reservation.close()
            continue
        return port, reservation


def _prepare_configs() -> None:
    """Create Toolathlon's untracked config files from its examples, as its setup does."""
    configs = TOOLATHLON_ROOT / "configs"
    for name in ("global_configs", "token_key_session"):
        config = configs / f"{name}.py"
        if not config.is_file():
            shutil.copyfile(configs / f"{name}_example.py", config)
    (configs / ".mcp-auth").mkdir(exist_ok=True)


def _preprocess_failure(completed: subprocess.CompletedProcess[str]) -> str | None:
    if completed.returncode != 0:
        return f"Toolathlon preprocess exited with code {completed.returncode}"
    output = completed.stdout + completed.stderr
    for pattern in PREPROCESS_FATAL_OUTPUT_PATTERNS:
        if pattern.search(output):
            return f"Toolathlon preprocess reported {pattern.pattern!r} despite exiting zero"
    return None


def _artifact_guard(action: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Hide or restore the task's evaluator and ground truth from the host."""
    return subprocess.run(
        [sys.executable, "-m", "scripts.containerized.task_artifact_guard", action, *args],
        env={**os.environ, "PYTHONPATH": str(TOOLATHLON_ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )


def _stop_agent_server(container: str, server: subprocess.Popen) -> None:
    """Stop all agents before the evaluator and ground truth come back."""
    if server.poll() is None:
        _exec(container, "sh", "-c", f'kill -TERM "$(cat {PID_FILE})"', check=False)
        server.wait(timeout=60)


def trajectory(state, *, bundle, episode_id, started_at, error):
    """Write the agent loop in Toolathlon's traj_log.json format for its evaluator."""
    roles = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}
    messages = []
    tool_names = set()
    for message in serialize_messages(state.get("messages", [])):
        data = message.get("data", message)
        content = data.get("content")
        entry = {
            "role": roles.get(data.get("type"), data.get("type")),
            "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
        }
        if data.get("tool_calls"):
            entry["tool_calls"] = [
                {"id": call.get("id"), "name": call.get("name"), "args": call.get("args")}
                for call in data["tool_calls"]
            ]
            tool_names.update(call.get("name") for call in data["tool_calls"])
        messages.append(entry)
    for agent_run_id, run in state.get("agent_runs", {}).items():
        for call in run.get("tool_calls", []):
            tool_names.add(call["name"])
            messages.append({
                "role": "agent_tool_call",
                "agent_run_id": agent_run_id,
                "content": json.dumps({"name": call["name"], "args": call["args"]}, ensure_ascii=False),
            })
    ai_messages = [message for message in messages if message["role"] == "assistant"]
    return {
        "config": bundle["resolved_task_config"],
        "request_id": str(uuid.uuid4()),
        "task_dir": bundle["task_dir"],
        "initial_run_time": started_at,
        "completion_time": datetime.now(timezone.utc).isoformat(),
        "tool_calls": {"tools": sorted(name for name in tool_names if name), "tool_choice": "auto"},
        "status": "failed" if error else "success",
        "messages": messages,
        "key_stats": {
            "interaction_turns": len(ai_messages),
            "tool_calls": sum(len(message.get("tool_calls", [])) for message in ai_messages),
            "agent_llm_requests": len(ai_messages),
            "agent_runs": len(state.get("agent_runs", {})),
        },
        "agent_cost": {},
        "user_cost": {},
        "resumed": False,
        "session_id": episode_id,
        "history_file": None,
        **({"error": error} if error else {}),
    }


def _cleanup_episode(*, episode_dir: Path, task_container: str, task: str, dumps_owner: str | None) -> None:
    """Capture raw container state and remove all per-episode resources."""
    cleanup: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "captures": [],
        "removals": [],
    }
    task_cleanup = K8S_TASK_CLEANUP_COMMANDS.get(task)
    if task_cleanup is not None:
        try:
            completed = _exec(task_container, *task_cleanup, check=False)
            cleanup["task_cleanup"] = {
                "command": task_cleanup, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr,
            }
        except BaseException as error:
            cleanup["task_cleanup"] = {"command": task_cleanup, "error": repr(error)}
    # Container root writes the episode directory; give it back to its owner.
    if dumps_owner is not None:
        try:
            completed = _exec(task_container, "chown", "-R", dumps_owner, "/workspace/dumps", check=False)
            cleanup["ownership"] = {"owner": dumps_owner, "returncode": completed.returncode}
        except BaseException as error:
            cleanup["ownership"] = {"owner": dumps_owner, "error": repr(error)}
    for kind, command in (
        ("log", ("logs", task_container)),
        ("inspect.json", ("inspect", task_container)),
    ):
        try:
            completed = _docker(*command, check=False)
            if kind == "inspect.json":
                # Container config includes API keys and command arguments.
                # Save only the lifecycle fields needed for failure analysis.
                inspected = json.loads(completed.stdout)
                content = json.dumps([
                    {
                        "Id": item.get("Id"),
                        "Name": item.get("Name"),
                        "Image": item.get("Image"),
                        "State": {
                            key: item.get("State", {}).get(key)
                            for key in ("Status", "Running", "ExitCode", "OOMKilled",
                                        "StartedAt", "FinishedAt")
                        },
                    }
                    for item in inspected
                ], indent=2)
            else:
                content = completed.stdout + completed.stderr
            if content:
                (episode_dir / f"task.{kind}").write_text(content, encoding="utf-8")
            cleanup["captures"].append(
                {"container": "task", "kind": kind, "returncode": completed.returncode}
            )
        except BaseException as error:
            cleanup["captures"].append({"container": "task", "kind": kind, "error": repr(error)})
    command = ("rm", "--force", "--volumes", task_container)
    try:
        completed = _docker(*command, check=False)
        cleanup["removals"].append(
            {"command": command, "returncode": completed.returncode,
             "stdout": completed.stdout, "stderr": completed.stderr}
        )
    except BaseException as error:
        cleanup["removals"].append({"command": command, "error": repr(error)})
    cleanup["finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        (episode_dir / "cleanup.json").write_text(
            json.dumps(cleanup, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except BaseException:
        pass


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run raw Toolathlon benchmark episodes.")
    parser.add_argument("task", nargs="?")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--all", action="store_true")
    assistants = json.loads(Path(__file__).with_name("langgraph.json").read_text())["graphs"]
    parser.add_argument("--agent", choices=tuple(assistants), default="decomposer")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("-n", "--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GYM_ARTIFACTS_DIR / "raw")
    # Toolathlon preprocess and native evaluation can each take tens of minutes.
    parser.add_argument("--episode-timeout", type=float, default=6000)
    parser.add_argument("--episode-id", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--purpose", default="raw", help=argparse.SUPPRESS)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--evals-dir", type=Path, default=DEFAULT_EVALS_DIR)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--n-jobs-per-worker", type=int, default=1000)
    parser.add_argument("--agent-timeout", type=float, default=2700)
    parser.add_argument("--container-lock-file", type=Path, help=argparse.SUPPRESS)
    return parser


def run_episode(args) -> None:
    parser = create_parser()
    args.harness = "decomposer" if args.agent == "decomposer" else "react"
    if args.n_jobs_per_worker < 1:
        parser.error("--n-jobs-per-worker must be at least 1")
    if args.agent_timeout <= 0:
        parser.error("--agent-timeout must be positive")
    tasks_dir = (TOOLATHLON_ROOT / "tasks" / "finalpool").resolve()
    task_dir = (tasks_dir / args.task).resolve()
    if task_dir.parent != tasks_dir or not task_dir.is_dir():
        raise ValueError(f"Unknown Toolathlon task: {args.task!r}")
    if not os.environ.get("LLM_PROXY_MASTER_KEY"):
        raise RuntimeError("Set LLM_PROXY_MASTER_KEY for the registered lmrouter models")
    proxy_mount = []
    proxy_socket = os.environ.get("LLM_PROXY_UNIX_SOCKET")
    if proxy_socket:
        socket_path = Path(proxy_socket).resolve()
        if not socket_path.is_socket():
            raise RuntimeError(f"Model proxy socket is missing: {socket_path}")
        proxy_mount = ["--volume", f"{socket_path.parent}:/run/model-proxy:ro",
                       "--env", f"LLM_PROXY_UNIX_SOCKET=/run/model-proxy/{socket_path.name}"]

    _docker("image", "inspect", args.image)
    _prepare_configs()

    episode_id = args.episode_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:8]
    )
    episode_dir = args.artifacts_dir.resolve() / args.task / episode_id
    evaluation_path = args.evals_dir.resolve() / args.task / episode_id / "result.json"
    episode_dir.mkdir(parents=True)
    task_container = f"decomposer-toolathlon-bench-{uuid.uuid4().hex[:16]}"
    task_path = f"/workspace/tasks/finalpool/{args.task}"
    # Holds the trusted bundle and the evaluator stash outside agent reach.
    private_dir = Path(tempfile.mkdtemp(prefix="toolathlon-bench-"))
    trusted_bundle = private_dir / "task_bundle.json"
    started_at = datetime.now(timezone.utc).isoformat()
    agent_server: subprocess.Popen | None = None
    agent_server_log = None
    port_reservations = []
    dumps_owner = None
    container_lock = _open_container_lock(args.container_lock_file)
    container_lock_held = False
    try:
        _acquire_container_lock(container_lock)
        container_lock_held = container_lock is not None
        print("Starting task environment...", flush=True)
        mounts = [
            "--volume", f"{episode_dir}:/workspace/dumps",
            "--volume", f"{episode_dir}:/workspace/logs",
            "--volume", f"{TOOLATHLON_ROOT / 'configs/.mcp-auth'}:/workspace/configs/.mcp-auth",
        ]
        notion_patch = TOOLATHLON_ROOT / "configs/notion-mcp-patches/notion-openapi.json"
        if notion_patch.is_file():
            mounts += ["--volume", f"{notion_patch}:/workspace/node_modules/"
                       "@notionhq/notion-mcp-server/scripts/notion-openapi.json:ro"]
        docker_host = os.environ.get("DOCKER_HOST", "")
        engine_socket = (Path(docker_host.removeprefix("unix://"))
                         if docker_host.startswith("unix://") else DOCKER_SOCKET)
        if engine_socket.exists():
            for target in CONTAINER_ENGINE_SOCKETS:
                mounts += ["--volume", f"{engine_socket}:{target}"]
        # Toolathlon's local services listen on host loopback.
        _docker(
            "run", "--detach", "--name", task_container, "--network", "host",
            *mounts, "--env", "LLM_PROXY_MASTER_KEY", *proxy_mount,
            "--env", f"DOCKER_API_VERSION={DOCKER_API_VERSION}", args.image,
        )
        # Read the owner inside the container, where cleanup runs chown: under
        # rootless Podman the host user is container root.
        dumps_owner = _exec(task_container, "stat", "-c", "%u:%g", "/workspace/dumps").stdout.strip()
        for relative in USER_CONFIG_FILES:
            if (TOOLATHLON_ROOT / relative).is_file():
                _docker("cp", str(TOOLATHLON_ROOT / relative), f"{task_container}:/workspace/{relative}")
        _exec(task_container, "mkdir", "-p", *GOOGLE_MCP_DIRS)
        for relative, name in GOOGLE_MCP_FILES.items():
            if (TOOLATHLON_ROOT / relative).is_file():
                for directory in GOOGLE_MCP_DIRS:
                    _docker("cp", str(TOOLATHLON_ROOT / relative), f"{task_container}:{directory}/{name}")
        _release_container_lock(container_lock)
        container_lock_held = False

        print("Running Toolathlon preprocess...", flush=True)
        attempts = K8S_PREPROCESS_ATTEMPTS if args.task in K8S_TASK_CLEANUP_COMMANDS else 1
        for attempt in range(1, attempts + 1):
            preprocess = _exec(
                task_container, "uv", "run", "python", "-m", "scripts.decoupled.container_preprocess",
                "--eval_config", EVAL_CONFIG,
                "--task_dir", f"finalpool/{args.task}",
                "--max_steps_under_single_turn_mode", str(MAX_STEPS),
                "--model_short_name", "decomposer",
                "--provider", "unified",
                "--bundle_file", CONTAINER_BUNDLE,
                "--host_output_folder", str(episode_dir),
                check=False,
            )
            with (episode_dir / "preprocess.log").open("a", encoding="utf-8") as log:
                log.write(f"=== attempt {attempt} ===\n{preprocess.stdout}{preprocess.stderr}")
            failure = _preprocess_failure(preprocess)
            if failure is None:
                break
            if attempt < attempts:
                _exec(task_container, *K8S_TASK_CLEANUP_COMMANDS[args.task], check=False)
                _exec(task_container, "rm", "-rf", "--", "/workspace/dumps/workspace", check=False)
        else:
            raise RuntimeError(f"{failure}; inspect {episode_dir / 'preprocess.log'}")
        _docker("cp", f"{task_container}:{CONTAINER_BUNDLE}", str(trusted_bundle))
        bundle = json.loads(trusted_bundle.read_text(encoding="utf-8"))

        print("Hiding evaluator and ground-truth artifacts...", flush=True)
        stash_dir = _artifact_guard(
            "stash", "--runtime", "docker", "--container", task_container,
            "--task-path", task_path, "--stash-root", str(private_dir / "stash"),
        ).stdout.strip()

        print("Starting Agent Server...", flush=True)
        gateway_port, reservation = _reserve_port()
        port_reservations.append(reservation)
        agent_port, reservation = _reserve_port()
        port_reservations.append(reservation)
        agent_server_log = (episode_dir / "agent_server.log").open("wb")
        agent_server = subprocess.Popen(
            [
                "docker", "exec", "--workdir", "/workspace",
                "--env", f"TOOLATHLON_BUNDLE={CONTAINER_BUNDLE}",
                "--env", f"GATEWAY_PORT={gateway_port}",
                "--env", f"AGENT_SERVER_PORT={agent_port}",
                "--env", f"N_JOBS_PER_WORKER={args.n_jobs_per_worker}",
                "--env", "TOOLATHLON_AGENT_CALL_LOG=/workspace/dumps/agent_model_calls.jsonl",
                task_container,
                "/opt/agents/bin/python", "/opt/decomposer/gyms/toolathlon_bench/task.py",
            ],
            stdout=agent_server_log,
            stderr=subprocess.STDOUT,
        )
        agent_url = f"http://127.0.0.1:{agent_port}"
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if agent_server.poll() is not None:
                raise RuntimeError(
                    "Agent Server exited before becoming ready; inspect "
                    f"{episode_dir / 'agent_server.log'} and {episode_dir / 'gateway.log'}"
                )
            try:
                with urllib.request.urlopen(f"{agent_url}/ok", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError(
                f"{agent_url}/ok did not become ready within {args.startup_timeout:g}s"
            )

        runtime = json.loads((episode_dir / "runtime.json").read_text(encoding="utf-8"))
        print(f"Running {args.harness}...", flush=True)
        agent_model = runtime["agent_model"]
        decomposer_model = runtime["decomposer_model"] if args.harness == "decomposer" else None
        selected_model = decomposer_model or agent_model
        thread_id = str(uuid.uuid5(uuid.NAMESPACE_URL, episode_id))
        state, agent_exception = asyncio.run(invoke_and_capture(
            agent_url,
            args.agent,
            {"messages": [{"role": "user", "content": bundle["task_str"]}]},
            thread_id=thread_id,
            timeout=args.agent_timeout,
            config={"recursion_limit": 410},
        ))
        agent_error = repr(agent_exception) if agent_exception is not None else None
        messages = state.get("messages", [])
        serialized_messages = serialize_messages(messages)
        agent_runs = state.get("agent_runs", {})
        agents = state.get("agents", {})
        usage = build_usage_summary(serialized_messages, agent_runs, agents)
        if args.harness == "react":
            usage["react"] = usage.pop("decomposer")
        (episode_dir / "trace.json").write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "thread_id": thread_id,
                    "run_id": args.run_id,
                    "task": args.task,
                    "repetition": args.repetition,
                    "attempt": args.attempt,
                    "purpose": args.purpose,
                    "harness": args.harness,
                    "assistant_id": args.agent,
                    "model": selected_model["model_id"],
                    "decomposer_model": decomposer_model["model_id"] if decomposer_model else None,
                    "teacher_backend": decomposer_model["model_id"].split("/", 1)[0] if decomposer_model else "agent",
                    "model_proxy_unix_socket": proxy_socket,
                    "decomposer_generation_config": decomposer_model["generation_config"] if decomposer_model else None,
                    "agent_model_id": agent_model["model_id"],
                    "agent_model": agent_model["api_model"],
                    "agent_api_model": agent_model["api_model"],
                    "agent_base_url": agent_model["base_url"],
                    "agent_generation_config": agent_model["generation_config"],
                    "agent_shutdown": state.get("agent_shutdown"),
                    "agent_shutdown_error": state.get("agent_shutdown_error"),
                    "agent_capture_error": state.get("agent_capture_error"),
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "decomposer_agent_runs": state.get("decomposer_agent_runs", []),
                    "agent_error": agent_error,
                    "messages": serialized_messages,
                    "agents": agents,
                    "agent_runs": agent_runs,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        (episode_dir / "usage.json").write_text(
            json.dumps(usage, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        last = serialized_messages[-1] if serialized_messages else {}
        answer = str(last.get("data", last).get("content", ""))
        (episode_dir / "answer.txt").write_text(answer, encoding="utf-8")
        _stop_agent_server(task_container, agent_server)
        trajectory_path = episode_dir / "traj_log.json"
        trajectory_path.write_text(
            json.dumps(
                trajectory(state, bundle=bundle, episode_id=episode_id,
                           started_at=started_at, error=agent_error),
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )

        print("Running native evaluation...", flush=True)
        if state.get("agent_shutdown_error"):
            evaluation = {
                "episode_id": episode_id, "task": args.task, "pass": False,
                "native_pass": None, "agent_error": agent_error,
                "details": "Evaluation skipped: agents could not be confirmed stopped",
                "agent_shutdown_error": state["agent_shutdown_error"],
            }
        else:
            _artifact_guard(
                "restore", "--runtime", "docker", "--container", task_container,
                "--task-path", task_path, "--stash-dir", stash_dir,
            )
            _docker("cp", str(trusted_bundle), f"{task_container}:{CONTAINER_BUNDLE}")
            # Agents could write this file; only the evaluator may create it.
            _exec(task_container, "rm", "-rf", "--", "/workspace/dumps/eval_res.json")
            # Exit code 0 grades the artifacts even after an agent failure; that
            # failure still keeps the strict pass below false.
            completed = _exec(
                task_container, "uv", "run", "python", "-m", "scripts.decoupled.container_eval",
                "--bundle_file", CONTAINER_BUNDLE, "--require_resolved_task_config",
                "--consume_bundle", "--agent_exit_code", "0",
                check=False,
            )
            if agent_error:
                # The evaluator persisted the forced success status; restore the real one.
                trajectory_log = json.loads(trajectory_path.read_text(encoding="utf-8"))
                trajectory_log["status"] = "failed"
                trajectory_path.write_text(
                    json.dumps(trajectory_log, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            try:
                native_result = json.loads((episode_dir / "eval_res.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                native_result = None
            evaluation = {
                "episode_id": episode_id,
                "task": args.task,
                "pass": completed.returncode == 0 and agent_exception is None,
                "native_pass": completed.returncode == 0,
                "agent_error": agent_error,
                "returncode": completed.returncode,
                "native_result": native_result,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_trace_html(state, episode_dir / "trace.html")

        # Preserve diagnostic scores, but keep interrupted agents unsuccessful.
        if agent_exception is not None:
            raise RuntimeError(f"Agent loop failed: {agent_error}") from agent_exception

        print(answer)
        print(f"\nArtifacts: {episode_dir}")
        print(f"Evaluation: {evaluation['pass']} ({evaluation_path})")
    finally:
        if container_lock_held:
            _release_container_lock(container_lock)
            container_lock_held = False
        _acquire_container_lock(container_lock)
        print("Cleaning up...", flush=True)
        try:
            _cleanup_episode(
                episode_dir=episode_dir,
                task_container=task_container,
                task=args.task,
                dumps_owner=dumps_owner,
            )
        finally:
            _release_container_lock(container_lock)
            if container_lock is not None:
                container_lock.close()
            if agent_server is not None and agent_server.poll() is None:
                agent_server.kill()
                agent_server.wait()
            if agent_server_log is not None:
                agent_server_log.close()
            for reservation in port_reservations:
                reservation.close()
            shutil.rmtree(private_dir, ignore_errors=True)



def serialize_messages(messages):
    return [message if isinstance(message, dict) else message_to_dict(message) for message in messages]


def main(argv=None):
    args = create_parser().parse_args(argv)
    if args.episode_id:
        return run_episode(args)
    from gyms.toolathlon_bench.parallel import run
    return run(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_termination)
    main()
