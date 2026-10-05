from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import shlex
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import message_to_dict

from decomposer.agent_server import invoke_and_capture
from decomposer.visualization import write_trace_html

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
POSTGRES_IMAGE = "docker.io/library/postgres:15"
POSTGRES_ENV = {
    "PGHOST": "postgres",
    "PG_HOST": "postgres",
    "PGPORT": "5432",
    "PGUSER": "eigent",
    "PGPASSWORD": "camel",
    "PGDATABASE": "toolathlon_gym",
}


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
        ("rm", "--force", "--volumes", task_container),
        ("rm", "--force", "--volumes", pg_container),
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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run raw Toolathlon Gym episodes.")
    parser.add_argument("task", nargs="?")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--all", action="store_true")
    assistants = json.loads(Path(__file__).with_name("langgraph.json").read_text())["graphs"]
    parser.add_argument("--agent", choices=tuple(assistants), default="decomposer")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("-n", "--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GYM_ARTIFACTS_DIR / "raw")
    parser.add_argument("--episode-timeout", type=float, default=3300)
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
    proxy_mount = []
    proxy_socket = os.environ.get("LLM_PROXY_UNIX_SOCKET")
    if proxy_socket:
        socket_path = Path(proxy_socket).resolve()
        if not socket_path.is_socket():
            raise RuntimeError(f"Model proxy socket is missing: {socket_path}")
        proxy_mount = ["--volume", f"{socket_path.parent}:/run/model-proxy:ro",
                       "--env", f"LLM_PROXY_UNIX_SOCKET=/run/model-proxy/{socket_path.name}"]

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
            "VLLM_HOST=host.docker.internal",
            "--env",
            f"TOOLATHLON_TASK={args.task}",
            "--env",
            f"N_JOBS_PER_WORKER={args.n_jobs_per_worker}",
            "--env",
            "TOOLATHLON_AGENT_CALL_LOG=/artifacts/data/agent_model_calls.jsonl",
            "--env",
            "LLM_PROXY_MASTER_KEY",
            "--env",
            "OPENROUTER_API_KEY",
            *proxy_mount,
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
        agent_url = f"http://127.0.0.1:{mapping.rsplit(':', 1)[1]}"
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{agent_url}/ok", timeout=2) as response:
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
                f"{agent_url}/ok did not become ready within "
                f"{args.startup_timeout:g}s"
            )

        _release_container_lock(container_lock)
        container_lock_held = False

        runtime = json.loads((episode_dir / "runtime.json").read_text(encoding="utf-8"))
        print(f"Running {args.harness}...", flush=True)
        agent_model = runtime["agent_model"]
        decomposer_model = runtime["decomposer_model"] if args.harness == "decomposer" else None
        selected_model = decomposer_model or agent_model
        thread_id = str(uuid.uuid5(uuid.NAMESPACE_URL, episode_id))
        state, agent_exception = asyncio.run(invoke_and_capture(
            agent_url,
            args.agent,
            {"messages": [{"role": "user", "content": runtime["task_config"]["task_str"]}]},
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

        print("Running native evaluation...", flush=True)
        config = runtime["task_config"]
        command = config["evaluation"]["evaluation_command"]
        if state.get("agent_shutdown_error"):
            evaluation = {
                "episode_id": episode_id, "task": args.task, "pass": False,
                "native_pass": None, "agent_error": agent_error,
                "details": "Evaluation skipped: agents could not be confirmed stopped",
                "agent_shutdown_error": state["agent_shutdown_error"],
            }
        elif command is None:
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
                pg_container=pg_container,
                network=network,
            )
        finally:
            _release_container_lock(container_lock)
            if container_lock is not None:
                container_lock.close()



def serialize_messages(messages):
    return [message if isinstance(message, dict) else message_to_dict(message) for message in messages]


def main(argv=None):
    args = create_parser().parse_args(argv)
    if args.episode_id:
        return run_episode(args)
    from gyms.toolathlon_gym.parallel import run
    return run(args)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_termination)
    main()
