from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import message_to_dict
from langchain_openrouter import ChatOpenRouter

from decomposer.core import create_decomposer_agent


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLATHLON_ROOT = REPO_ROOT / "external" / "toolathlon_gym"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "data" / "toolathlon_gym"
DEFAULT_EVALS_DIR = REPO_ROOT / "artifacts" / "evals" / "toolathlon_gym"
DEFAULT_IMAGE = "decomposer-toolathlon:latest"
DEFAULT_MODEL = "z-ai/glm-5.2"
POSTGRES_IMAGE = "postgres:15"
POSTGRES_ENV = {
    "PGHOST": "postgres",
    "PG_HOST": "postgres",
    "PGPORT": "5432",
    "PGUSER": "eigent",
    "PGPASSWORD": "camel",
    "PGDATABASE": "toolathlon_gym",
}
VLLM_MODELS = {
    8020: "google/gemma-4-E2B-it",
    8021: "google/gemma-4-E4B-it",
    8022: "google/gemma-4-12B-it",
    8023: "google/gemma-4-26B-A4B-it",
}
SUBAGENT_TYPES = (
    # subagent_type_id, assistant_id, model_description
    ("tiny_thinking", "gemma_4_2b_thinking", "tiny thinking"),
    ("tiny_non_thinking", "gemma_4_2b_non_thinking", "tiny non-thinking"),
    ("small_thinking", "gemma_4_4b_thinking", "small thinking"),
    ("small_non_thinking", "gemma_4_4b_non_thinking", "small non-thinking"),
    ("medium_thinking", "gemma_4_12b_thinking", "medium thinking"),
    ("medium_non_thinking", "gemma_4_12b_non_thinking", "medium non-thinking"),
    ("large_thinking", "gemma_4_26b_a4b_thinking", "large thinking"),
    (
        "large_non_thinking",
        "gemma_4_26b_a4b_non_thinking",
        "large non-thinking",
    ),
)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--evals-dir", type=Path, default=DEFAULT_EVALS_DIR)
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()

    tasks_dir = (TOOLATHLON_ROOT / "tasks" / "finalpool").resolve()
    task_dir = (tasks_dir / args.task).resolve()
    if task_dir.parent != tasks_dir or not task_dir.is_dir():
        raise ValueError(f"Unknown Toolathlon task: {args.task!r}")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY for the Decomposer model")

    print("Checking vLLM servers...", flush=True)
    for port, expected_model in VLLM_MODELS.items():
        url = f"http://127.0.0.1:{port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                models = json.load(response)["data"]
        except (OSError, KeyError, json.JSONDecodeError) as error:
            raise RuntimeError(f"vLLM is not ready at {url}") from error
        if expected_model not in {model["id"] for model in models}:
            raise RuntimeError(f"{url} does not serve {expected_model}")
    _docker("image", "inspect", args.image)

    episode_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:8]
    )
    episode_dir = args.artifacts_dir.resolve() / args.task / episode_id
    evaluation_path = args.evals_dir.resolve() / args.task / episode_id / "result.json"
    episode_dir.mkdir(parents=True)
    network = f"decomposer-toolathlon-{episode_id}"
    pg_container = f"{network}-pg"
    task_container = f"{network}-task"
    started_at = datetime.now(timezone.utc).isoformat()

    try:
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
            if status == "healthy":
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
            for pair in POSTGRES_ENV.items()
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
            f"TOOLATHLON_TASK={args.task}",
            *postgres_env,
            "--volume",
            f"{episode_dir.resolve()}:/artifacts/data",
            args.image,
        )
        mapping = _docker("port", task_container, "2024/tcp").stdout.strip()
        subagent_url = f"http://127.0.0.1:{mapping.rsplit(':', 1)[1]}"
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{subagent_url}/ok", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError(
                f"{subagent_url}/ok did not become ready within "
                f"{args.startup_timeout:g}s"
            )

        runtime = json.loads((episode_dir / "runtime.json").read_text(encoding="utf-8"))
        print("Running Decomposer...", flush=True)
        agent = create_decomposer_agent(
            decomposer_model=ChatOpenRouter(
                model=args.model,
                temperature=1.0,
                top_p=0.95,
                reasoning={"effort": "high"},
            ),
            subagent_types=[
                {
                    "subagent_type_id": subagent_type_id,
                    "description": (
                        f"Tool-calling agent based on a {model_description} model. "
                        "Has access to all the available tools."
                    ),
                    "assistant_id": assistant_id,
                    "url": subagent_url,
                }
                for subagent_type_id, assistant_id, model_description in SUBAGENT_TYPES
            ],
        )
        state = asyncio.run(
            agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": runtime["task_config"]["task_str"],
                        }
                    ]
                },
                config={"recursion_limit": 200},
            )
        )
        messages = state["messages"]
        (episode_dir / "trace.json").write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "task": args.task,
                    "decomposer_model": args.model,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "messages": [message_to_dict(message) for message in messages],
                    "subagent_runs": state.get("subagent_runs", {}),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        answer = str(messages[-1].content)
        (episode_dir / "answer.txt").write_text(answer, encoding="utf-8")

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
        print("Cleaning up...", flush=True)
        log_result = _docker("logs", task_container, check=False)
        logs = log_result.stdout + log_result.stderr
        if logs:
            (episode_dir / "container.log").write_text(logs, encoding="utf-8")
        _docker("rm", "--force", task_container, check=False)
        _docker("rm", "--force", pg_container, check=False)
        _docker("network", "rm", network, check=False)


if __name__ == "__main__":
    main()
