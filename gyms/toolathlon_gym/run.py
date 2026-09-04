from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import message_to_dict
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver

from decomposer.chat_vllm import ChatVLLM
from decomposer.core import create_decomposer_agent
from decomposer.prompts import DECOMPOSER_TEACHER_SYSTEM_PROMPT

try:
    from .usage import build_usage_summary
except ImportError:  # Executed directly as a script.
    from usage import build_usage_summary


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLATHLON_ROOT = REPO_ROOT / "external" / "toolathlon_gym"
DEFAULT_GYM_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "gyms" / "toolathlon_gym"
DEFAULT_ARTIFACTS_DIR = DEFAULT_GYM_ARTIFACTS_DIR / "traces"
DEFAULT_EVALS_DIR = DEFAULT_GYM_ARTIFACTS_DIR / "evals"
DEFAULT_IMAGE = "decomposer-toolathlon:latest"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_SUBAGENT_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_SUBAGENT_API_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_SUBAGENT_PORT = 8023
POSTGRES_IMAGE = "docker.io/library/postgres:15"
POSTGRES_ENV = {
    "PGHOST": "postgres",
    "PG_HOST": "postgres",
    "PGPORT": "5432",
    "PGUSER": "eigent",
    "PGPASSWORD": "camel",
    "PGDATABASE": "toolathlon_gym",
}
SUBAGENT_TYPE_ID = "configured_non_thinking"
# Retained for compatibility with callers that inspect the historical default.
SUBAGENT_TYPES = ((
    "gemma_4_26b_a4b_non_thinking",
    "gemma_4_26b_a4b_non_thinking",
    "Gemma-4-26B-A4B non-thinking",
),)


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


def _postgres_environment(pg_container: str) -> dict[str, str]:
    address = _docker(
        "inspect",
        "--format",
        "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
        pg_container,
    ).stdout.strip()
    if not address:
        raise RuntimeError(f"PostgreSQL container has no network address: {pg_container}")
    return {**POSTGRES_ENV, "PGHOST": address, "PG_HOST": address}


def _cleanup_episode(
    *,
    episode_dir: Path,
    task_container: str,
    pg_container: str,
    network: str,
) -> None:
    """Capture raw container state and remove all per-episode resources."""
    cleanup: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "captures": [],
        "removals": [],
    }
    for label, container in (("task", task_container), ("postgres", pg_container)):
        for kind, command in (
            ("log", ("logs", container)),
            ("inspect.json", ("inspect", container)),
        ):
            try:
                completed = _docker(*command, check=False)
                content = completed.stdout + completed.stderr
                if content:
                    (episode_dir / f"{label}.{kind}").write_text(
                        content, encoding="utf-8"
                    )
                cleanup["captures"].append(
                    {"container": label, "kind": kind, "returncode": completed.returncode}
                )
            except BaseException as error:
                cleanup["captures"].append(
                    {"container": label, "kind": kind, "error": repr(error)}
                )
    for command in (
        ("rm", "--force", task_container),
        ("rm", "--force", pg_container),
        ("network", "rm", network),
    ):
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


def vllm_command(
    model: str,
    port: int,
    *,
    served_model_name: str = DEFAULT_SUBAGENT_API_MODEL,
    max_model_len: int,
    gpu_memory_utilization: float,
    data_parallel_size: int = 1,
) -> list[str]:
    command = [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        model,
        "--served-model-name",
        served_model_name,
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
        "gemma4",
        "--reasoning-parser",
        "gemma4",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    if data_parallel_size > 1:
        command.extend(
            [
                "--data-parallel-size",
                str(data_parallel_size),
                # Keep one frontend so turns from persistent episode clients
                # can be dispatched across all data-parallel engines.
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
    served_model_name: str = DEFAULT_SUBAGENT_API_MODEL,
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
            expected_model=served_model_name,
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
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        data_parallel_size=data_parallel_size,
    )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "VLLM_ENGINE_READY_TIMEOUT_S": str(max(1, int(timeout))),
    }
    # vLLM and FlashInfer invoke helpers such as ``ninja`` by name. Preserve
    # the virtualenv tool directory in detached/non-interactive launches.
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
            expected_model=served_model_name,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--episode-id", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--purpose", choices=("trace-generation",), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--subagent-model", default=DEFAULT_SUBAGENT_MODEL)
    parser.add_argument("--subagent-api-model", default=DEFAULT_SUBAGENT_API_MODEL)
    parser.add_argument("--subagent-port", type=int, default=DEFAULT_SUBAGENT_PORT)
    parser.add_argument("--subagent-gpu", default="0")
    parser.add_argument("--vllm-max-model-len", type=int, default=256000)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-data-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-startup-timeout", type=float, default=1800)
    parser.add_argument("--reuse-vllm", action="store_true")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--evals-dir", type=Path, default=DEFAULT_EVALS_DIR)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--n-jobs-per-worker", type=int, default=1000)
    parser.add_argument("--agent-timeout", type=float, default=2700)
    parser.add_argument("--container-lock-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.n_jobs_per_worker < 1:
        parser.error("--n-jobs-per-worker must be at least 1")
    if args.agent_timeout <= 0:
        parser.error("--agent-timeout must be positive")
    if args.vllm_data_parallel_size < 1:
        parser.error("--vllm-data-parallel-size must be at least 1")
    visible_gpus = [item for item in args.subagent_gpu.split(",") if item]
    if len(visible_gpus) != args.vllm_data_parallel_size:
        parser.error(
            "--subagent-gpu must list exactly --vllm-data-parallel-size GPU IDs"
        )

    tasks_dir = (TOOLATHLON_ROOT / "tasks" / "finalpool").resolve()
    task_dir = (tasks_dir / args.task).resolve()
    if task_dir.parent != tasks_dir or not task_dir.is_dir():
        raise ValueError(f"Unknown Toolathlon task: {args.task!r}")
    decomposer_vllm_url = os.environ.get("DECOMPOSER_VLLM_BASE_URL")
    llm_proxy_url = os.environ.get("LLM_PROXY_URL")
    llm_proxy_key = os.environ.get("LLM_PROXY_MASTER_KEY")
    if decomposer_vllm_url:
        pass
    elif llm_proxy_url:
        if not llm_proxy_key:
            raise RuntimeError(
                "Set LLM_PROXY_MASTER_KEY when LLM_PROXY_URL is configured"
            )
    elif not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "Set OPENROUTER_API_KEY, or configure LLM_PROXY_URL and "
            "LLM_PROXY_MASTER_KEY for the Decomposer model"
        )

    _docker("image", "inspect", args.image)

    episode_id = args.episode_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:8]
    )
    episode_dir = args.artifacts_dir.resolve() / args.task / episode_id
    evaluation_path = args.evals_dir.resolve() / args.task / episode_id / "result.json"
    episode_dir.mkdir(parents=True)
    network = f"decomposer-toolathlon-{uuid.uuid4().hex[:16]}"
    pg_container = f"{network}-pg"
    task_container = f"{network}-task"
    started_at = datetime.now(timezone.utc).isoformat()
    vllm_log = (
        args.artifacts_dir.resolve().parent
        / "logs"
        / args.task
        / episode_id
        / "vllm.log"
    )
    print(f"Starting vLLM on GPU {args.subagent_gpu}...", flush=True)
    vllm_process = start_vllm(
        model=args.subagent_model,
        served_model_name=args.subagent_api_model,
        port=args.subagent_port,
        gpu=args.subagent_gpu,
        max_model_len=args.vllm_max_model_len,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        timeout=args.vllm_startup_timeout,
        log_path=vllm_log,
        reuse=args.reuse_vllm,
        data_parallel_size=args.vllm_data_parallel_size,
    )

    container_lock = _open_container_lock(args.container_lock_file)
    container_lock_held = False
    try:
        _acquire_container_lock(container_lock)
        container_lock_held = container_lock is not None
        print("Starting PostgreSQL...", flush=True)
        _docker("network", "create", network)
        dump = (TOOLATHLON_ROOT / "db" / "init.sql.gz").resolve()
        _docker(
            "run",
            "--http-proxy=false",
            "--detach",
            "--name",
            pg_container,
            "--network",
            network,
            "--network-alias",
            "postgres",
            "--env",
            "POSTGRES_DB=toolathlon_gym",
            "--env",
            "POSTGRES_USER=eigent",
            "--env",
            "POSTGRES_PASSWORD=camel",
            "--volume",
            f"{dump}:/docker-entrypoint-initdb.d/init.sql.gz:ro",
            "--health-cmd",
            "pg_isready -U eigent -d toolathlon_gym",
            "--health-interval",
            "2s",
            "--health-timeout",
            "5s",
            "--health-retries",
            "60",
            POSTGRES_IMAGE,
        )
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            status = _docker(
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                pg_container,
                check=False,
            ).stdout.strip()
            final_server = False
            schema_ready = False
            if status == "healthy":
                process_check = _docker(
                    "exec",
                    pg_container,
                    "cat",
                    "/proc/1/comm",
                    check=False,
                )
                final_server = (
                    process_check.returncode == 0
                    and process_check.stdout.strip() == "postgres"
                )
            if status == "healthy" and final_server:
                schema_check = _docker(
                    "exec",
                    pg_container,
                    "psql",
                    "--username",
                    "eigent",
                    "--dbname",
                    "toolathlon_gym",
                    "--tuples-only",
                    "--no-align",
                    "--command",
                    "SELECT to_regclass('email.messages') IS NOT NULL;",
                    check=False,
                )
                schema_ready = (
                    schema_check.returncode == 0
                    and schema_check.stdout.strip() == "t"
                )
            if status == "healthy" and final_server and schema_ready:
                break
            time.sleep(1)
        else:
            raise TimeoutError(
                f"PostgreSQL did not become healthy within {args.startup_timeout:g}s"
            )
        _docker(
            "exec",
            pg_container,
            "psql",
            "--username",
            "eigent",
            "--dbname",
            "toolathlon_gym",
            "--command",
            (
                "ALTER TABLE email.sent_log DROP CONSTRAINT IF EXISTS "
                "sent_log_message_id_fkey; "
                "ALTER TABLE email.sent_log ADD CONSTRAINT "
                "sent_log_message_id_fkey FOREIGN KEY (message_id) "
                "REFERENCES email.messages(id) ON DELETE CASCADE;"
            ),
        )

        print("Starting task environment...", flush=True)
        postgres_env = [
            item
            for pair in _postgres_environment(pg_container).items()
            for item in ("--env", "=".join(pair))
        ]
        _docker(
            "run",
            "--http-proxy=false",
            "--detach",
            "--name",
            task_container,
            "--network",
            network,
            "--add-host",
            "host.docker.internal:host-gateway",
            "--publish",
            "127.0.0.1::2024",
            "--env",
            f"TOOLATHLON_TASK={args.task}",
            "--env",
            f"N_JOBS_PER_WORKER={args.n_jobs_per_worker}",
            "--env",
            "TOOLATHLON_SUBAGENT_CALL_LOG=/artifacts/data/subagent_model_calls.jsonl",
            "--env",
            f"DECOMPOSER_SUBAGENT_MODEL={args.subagent_api_model}",
            "--env",
            (
                "DECOMPOSER_SUBAGENT_BASE_URL="
                f"http://host.docker.internal:{args.subagent_port}/v1"
            ),
            *postgres_env,
            "--volume",
            f"{episode_dir.resolve()}:/artifacts/data",
            args.image,
        )
        mapping = _docker("port", task_container, "2024/tcp").stdout.strip()
        if ":" not in mapping:
            status = _docker(
                "inspect",
                "--format",
                "{{.State.Status}}",
                task_container,
                check=False,
            ).stdout.strip()
            logs = _docker("logs", task_container, check=False)
            raise RuntimeError(
                "Task container did not publish port 2024 "
                f"(status={status or 'unknown'}):\n"
                + logs.stdout
                + logs.stderr
            )
        subagent_url = f"http://127.0.0.1:{mapping.rsplit(':', 1)[1]}"
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{subagent_url}/ok", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            status = _docker(
                "inspect",
                "--format",
                "{{.State.Status}}",
                task_container,
                check=False,
            ).stdout.strip()
            if status in {"dead", "exited"}:
                logs = _docker("logs", task_container, check=False)
                raise RuntimeError(
                    "Task container exited before becoming ready:\n"
                    + logs.stdout
                    + logs.stderr
                )
            time.sleep(1)
        else:
            raise TimeoutError(
                f"{subagent_url}/ok did not become ready within "
                f"{args.startup_timeout:g}s"
            )

        _release_container_lock(container_lock)
        container_lock_held = False

        runtime = json.loads((episode_dir / "runtime.json").read_text(encoding="utf-8"))
        print("Running Decomposer...", flush=True)
        checkpointer = InMemorySaver()
        openrouter_provider = os.environ.get("DECOMPOSER_OPENROUTER_PROVIDER")
        reasoning_effort = os.environ.get(
            "DECOMPOSER_REASONING_EFFORT", "max"
        ).lower()
        configured_max_tokens = os.environ.get("DECOMPOSER_MAX_TOKENS")
        decomposer_max_tokens = (
            int(configured_max_tokens) if configured_max_tokens else None
        )
        max_tokens_kwargs = (
            {"max_tokens": decomposer_max_tokens}
            if decomposer_max_tokens is not None
            else {}
        )
        request_timeout_seconds = float(
            os.environ.get("DECOMPOSER_OPENROUTER_TIMEOUT", "600")
        )
        openrouter_max_retries = int(
            os.environ.get("DECOMPOSER_OPENROUTER_MAX_RETRIES", "5")
        )
        reasoning = (
            {"enabled": False}
            if reasoning_effort in {"none", "off", "disabled"}
            else {"effort": reasoning_effort}
        )
        if decomposer_vllm_url:
            decomposer_model = ChatVLLM(
                model=args.model,
                base_url=decomposer_vllm_url,
                api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
                temperature=1.0,
                top_p=0.95,
                **max_tokens_kwargs,
                timeout=request_timeout_seconds,
                max_retries=openrouter_max_retries,
                disable_streaming=True,
                use_responses_api=False,
                preserve_reasoning=True,
                preserve_reasoning_on_tool_calls_only=True,
                extra_body={
                    "top_k": 64,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            teacher_backend = "vllm"
        elif llm_proxy_url:
            decomposer_model = ChatVLLM(
                model=args.model,
                base_url=llm_proxy_url,
                api_key=llm_proxy_key,
                temperature=1.0,
                top_p=0.95,
                **max_tokens_kwargs,
                timeout=request_timeout_seconds,
                max_retries=openrouter_max_retries,
                disable_streaming=True,
                use_responses_api=False,
                preserve_reasoning=True,
                parse_qwen_xml_tool_calls=True,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "include_reasoning": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            teacher_backend = "llm_proxy"
        else:
            decomposer_model = ChatOpenRouter(
                model=args.model,
                temperature=1.0,
                top_p=0.95,
                reasoning=reasoning,
                **max_tokens_kwargs,
                # DeepSeek V4 high-reasoning decompositions can legitimately
                # take several minutes. Preserve that response window while
                # allowing transient backend failures to retry; the outer
                # agent timeout still caps total episode wall-clock time.
                # langchain-openrouter/OpenRouter SDK timeouts are milliseconds.
                timeout=max(1, int(request_timeout_seconds * 1000)),
                max_retries=openrouter_max_retries,
                openrouter_provider=(
                    {
                        "order": [openrouter_provider],
                        "allow_fallbacks": False,
                    }
                    if openrouter_provider
                    else None
                ),
            )
            teacher_backend = "openrouter"
        agent = create_decomposer_agent(
            decomposer_model=decomposer_model,
            subagent_types=[{
                "subagent_type_id": SUBAGENT_TYPE_ID,
                "description": (
                    f"Tool-calling agent based on {args.subagent_api_model} in "
                    "non-thinking mode. Has access to all the available tools."
                ),
                "assistant_id": SUBAGENT_TYPE_ID,
                "url": subagent_url,
            }],
            decomposer_system_prompt=DECOMPOSER_TEACHER_SYSTEM_PROMPT,
            checkpointer=checkpointer,
            subagent_recursion_limit=410,
        )
        agent_config = {
            "recursion_limit": 410,
            "configurable": {"thread_id": episode_id},
        }
        agent_error: str | None = None
        agent_exception: BaseException | None = None
        try:
            state = asyncio.run(
                asyncio.wait_for(
                    agent.ainvoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": runtime["task_config"]["task_str"],
                                }
                            ]
                        },
                        config=agent_config,
                    ),
                    timeout=args.agent_timeout,
                )
            )
        except BaseException as error:
            agent_exception = error
            agent_error = repr(error)
            try:
                snapshot = asyncio.run(agent.aget_state(agent_config))
                state = dict(snapshot.values)
            except BaseException:
                state = {}
        messages = state.get("messages", [])
        serialized_messages = [message_to_dict(message) for message in messages]
        subagent_runs = state.get("subagent_runs", {})
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
                    "decomposer_model": args.model,
                    "teacher_backend": teacher_backend,
                    "openrouter_provider": openrouter_provider,
                    "reasoning_effort": reasoning_effort,
                    "decomposer_generation_config": {
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "reasoning": reasoning,
                    },
                    "decomposer_max_tokens": decomposer_max_tokens,
                    "request_timeout_seconds": request_timeout_seconds,
                    "openrouter_max_retries": openrouter_max_retries,
                    "subagent_model": args.subagent_model,
                    "subagent_api_model": args.subagent_api_model,
                    "subagent_generation_config": {
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 64,
                        "reasoning_effort": "none",
                        "chat_template_kwargs": {"enable_thinking": False},
                        "preserve_reasoning": False,
                    },
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "agent_error": agent_error,
                    "messages": serialized_messages,
                    "subagent_runs": subagent_runs,
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
        answer = str(messages[-1].content) if messages else ""
        (episode_dir / "answer.txt").write_text(answer, encoding="utf-8")

        if agent_exception is not None:
            raise RuntimeError(f"Decomposer agent loop failed: {agent_error}") from agent_exception

        print("Running native evaluation...", flush=True)
        config = runtime["task_config"]
        command = config["evaluation"]["evaluation_command"]
        if command is None:
            evaluation = {
                "episode_id": episode_id,
                "task": args.task,
                "pass": None,
                "details": "No native evaluator configured",
            }
        else:
            native_path = "/tmp/decomposer-evaluation.json"
            evaluation_args = [
                *shlex.split(command),
                "--agent_workspace",
                config["agent_workspace"],
            ]
            groundtruth = config["evaluation"]["groundtruth_workspace"]
            if groundtruth is not None:
                evaluation_args.extend(["--groundtruth_workspace", groundtruth])
            if config["launch_time"] is not None:
                evaluation_args.extend(["--launch_time", config["launch_time"]])
            evaluation_args.extend(["--res_log_file", native_path])
            completed = _docker(
                "exec",
                "--workdir",
                "/workspace",
                task_container,
                *evaluation_args,
                check=False,
            )
            native = _docker(
                "exec",
                task_container,
                "cat",
                native_path,
                check=False,
            )
            try:
                native_result = (
                    json.loads(native.stdout) if native.returncode == 0 else None
                )
            except json.JSONDecodeError:
                native_result = native.stdout
            evaluation = {
                "episode_id": episode_id,
                "task": args.task,
                "pass": completed.returncode == 0,
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
                pg_container=pg_container,
                network=network,
            )
        finally:
            _release_container_lock(container_lock)
            if container_lock is not None:
                container_lock.close()
            stop_vllm(vllm_process)


if __name__ == "__main__":
    import batch

    signal.signal(signal.SIGTERM, _handle_termination)
    if batch.wants_batch(sys.argv[1:]):
        batch.main(
            sys.argv[1:],
            repo_root=REPO_ROOT,
            toolathlon_root=TOOLATHLON_ROOT,
            default_artifacts_dir=DEFAULT_GYM_ARTIFACTS_DIR,
            default_image=DEFAULT_IMAGE,
            default_model=DEFAULT_MODEL,
            default_subagent_model=DEFAULT_SUBAGENT_MODEL,
            default_subagent_api_model=DEFAULT_SUBAGENT_API_MODEL,
            default_subagent_port=DEFAULT_SUBAGENT_PORT,
            start_vllm=start_vllm,
            stop_vllm=stop_vllm,
            docker=_docker,
        )
    else:
        main()
