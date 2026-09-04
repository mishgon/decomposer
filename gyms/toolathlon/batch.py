from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__:
    from .settings import SUBAGENT_CONTEXT_TOKENS, SUBAGENT_RECURSION_LIMIT
else:
    from settings import SUBAGENT_CONTEXT_TOKENS, SUBAGENT_RECURSION_LIMIT


SCHEMA_VERSION = 1
VALID_EVALUATION_TASKS = (
    "finalpool/academic-pdf-report",
    "finalpool/add-bibtex",
    "finalpool/apply-phd-email",
    "finalpool/arrange-workspace",
    "finalpool/canvas-arrange-exam",
    "finalpool/canvas-art-manager",
    "finalpool/canvas-art-quiz",
    "finalpool/canvas-do-quiz",
    "finalpool/canvas-homework-grader-python",
    "finalpool/canvas-list-test",
    "finalpool/canvas-new-students-notification",
    "finalpool/canvas-submit-late-work",
    "finalpool/cooking-guidance",
    "finalpool/course-assistant",
    "finalpool/course-schedule",
    "finalpool/courses-ta-hws",
    "finalpool/cvpr-research",
    "finalpool/detect-revised-terms",
    "finalpool/dietary-health",
    "finalpool/excel-data-transformation",
    "finalpool/excel-market-research",
    "finalpool/filter-low-selling-products",
    "finalpool/find-alita-paper",
    "finalpool/git-bug-hunt",
    "finalpool/hk-top-conf",
    "finalpool/identify-all-songs",
    "finalpool/imagenet",
    "finalpool/interview-report",
    "finalpool/inventory-sync",
    "finalpool/invoice-org",
    "finalpool/ipad-edu-price",
    "finalpool/k8s-deployment-cleanup",
    "finalpool/k8s-mysql",
    "finalpool/k8s-pr-preview-testing",
    "finalpool/k8s-redis-helm-upgrade",
    "finalpool/language-school",
    "finalpool/latex-prompt-box",
    "finalpool/logical-datasets-collection",
    "finalpool/meeting-assign",
    "finalpool/mrbeast-analysis",
    "finalpool/nvidia-market",
    "finalpool/paper-checker",
    "finalpool/ppt-analysis",
    "finalpool/privacy-desensitization",
    "finalpool/profile-update-online",
    "finalpool/reimbursement-form-filler",
    "finalpool/sales-accounting",
    "finalpool/shopping-helper",
    "finalpool/stock-build-position",
    "finalpool/train-ticket-plan",
    "finalpool/travel-exchange",
    "finalpool/university-course-selection",
    "finalpool/woocommerce-new-product",
    "finalpool/woocommerce-update-cover",
    "finalpool/yahoo-analysis",
)
RESUME_CONFIG_FIELDS = (
    "agent_mode",
    "simple_agent_implementation",
    "agent_system_prompt",
    "model",
    "decomposer_provider",
    "decomposer_base_url",
    "decomposer_prompt",
    "subagent_provider",
    "subagent_model",
    "subagent_port",
    "subagent_ports",
    "subagent_base_url",
    "subagent_gpu",
    "image",
    "docker_socket",
    "reuse_vllm",
    "publish_service_ports",
    "vllm_max_model_len",
    "subagent_recursion_limit",
    "vllm_gpu_memory_utilization",
    "vllm_data_parallel_size",
    "vllm_startup_timeout",
    "startup_timeout",
    "container_slots",
    "agent_timeout",
    "episode_timeout",
    "max_steps",
    "max_tool_output_chars",
    "eval_config",
)

# These services are shared mutable environments, not per-episode MCP
# processes.  Different tasks can otherwise reset the same Canvas courses or
# Kubernetes cluster while another agent is still working in it.
SHARED_MUTABLE_MCP_SERVERS = frozenset({"canvas", "k8s"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    return stamp + uuid.uuid4().hex[:8]


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_event(run_dir: Path, event: str, **fields: Any) -> None:
    record = {"timestamp": utc_now(), "event": event, **fields}
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        output.flush()
        os.fsync(output.fileno())


def wants_batch(argv: Sequence[str]) -> bool:
    flags = {
        "--all", "--all-valid", "--tasks", "--resume", "--repetitions", "-n"
    }
    prefixes = ("--tasks=", "--resume=", "--repetitions=", "-n")
    return any(
        argument in flags or argument.startswith(prefixes) for argument in argv
    )


def parse_args(argv: Sequence[str], defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable Decomposer batch on the Toolathlon benchmark."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true")
    selection.add_argument(
        "--all-valid",
        action="store_true",
        help=(
            "Run the 55 infrastructure-valid tasks, while scoring omitted "
            "tasks from the full benchmark as failures."
        ),
    )
    selection.add_argument("--tasks", nargs="+", metavar="TASK")
    parser.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("-n", "--repetitions", type=int, default=1)
    parser.add_argument(
        "--purpose", choices=("trace-generation", "evaluation"), required=True
    )
    parser.add_argument(
        "--agent-mode", choices=("simple", "decomposer"), default="decomposer"
    )
    parser.add_argument(
        "--simple-agent-implementation",
        choices=("toolathlon", "langgraph"),
        default="toolathlon",
    )
    parser.add_argument(
        "--agent-system-prompt",
        choices=("toolathlon", "generic"),
        default="toolathlon",
    )
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument(
        "--decomposer-provider",
        choices=("openrouter", "vllm", "lmrouter"),
        default="openrouter",
    )
    parser.add_argument(
        "--decomposer-base-url",
        default="http://127.0.0.1:8040/v1",
        help="OpenAI-compatible endpoint used by a local vLLM decomposer.",
    )
    parser.add_argument(
        "--decomposer-prompt",
        choices=("student", "teacher"),
        default="teacher",
        help="System prompt used by the decomposer model (default: teacher).",
    )
    parser.add_argument(
        "--subagent-provider",
        choices=("vllm", "openrouter"),
        default="vllm",
    )
    parser.add_argument("--subagent-model", default=defaults["subagent_model"])
    parser.add_argument("--subagent-port", type=int, default=defaults["subagent_port"])
    parser.add_argument("--subagent-base-url")
    parser.add_argument(
        "--subagent-ports",
        type=int,
        nargs="+",
        help=(
            "Pool of externally managed vLLM ports on this host. Episodes are "
            "assigned round-robin; implies --reuse-vllm."
        ),
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
    parser.add_argument("--publish-service-ports", action="store_true")
    parser.add_argument("--image", default=defaults["image"])
    parser.add_argument(
        "--docker-socket",
        help="Host Docker socket mounted into the task container as /var/run/docker.sock",
    )
    parser.add_argument(
        "--bench-artifacts-dir", type=Path, default=defaults["artifacts_dir"]
    )
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--n-jobs-per-worker", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-tool-output-chars", type=int, default=100_000)
    parser.add_argument(
        "--eval-config",
        default="scripts/formal_run_v0.json",
        help="Evaluation config path inside the task container.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--container-slots", type=int, default=1)
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=2700,
        help="Maximum seconds spent in the agent loop (default: 2700).",
    )
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=6000,
        help=(
            "Maximum total wall-clock seconds for one episode (default: 6000, "
            "covering container startup, preprocess, a 45-minute agent, "
            "evaluator isolation, and native evaluation)."
        ),
    )
    args = parser.parse_args(argv)
    if args.resume and (args.all or args.all_valid or args.tasks):
        parser.error("--resume cannot be combined with task selection")
    if not args.resume and not (args.all or args.all_valid or args.tasks):
        parser.error("choose --all, --all-valid, --tasks, or --resume")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.resume and args.repetitions != 1:
        parser.error("--repetitions cannot be changed on resume")
    if args.n_jobs_per_worker < 1:
        parser.error("--n-jobs-per-worker must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.container_slots < 1:
        parser.error("--container-slots must be at least 1")
    if args.agent_timeout <= 0 or args.episode_timeout <= 0:
        parser.error("--agent-timeout and --episode-timeout must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.max_tool_output_chars < 1:
        parser.error("--max-tool-output-chars must be at least 1")
    if (
        args.agent_mode == "simple"
        and args.simple_agent_implementation == "toolathlon"
        and args.subagent_provider != "vllm"
    ):
        parser.error(
            "--simple-agent-implementation toolathlon currently requires "
            "--subagent-provider vllm"
        )
    if (
        args.agent_mode == "simple"
        and args.simple_agent_implementation == "toolathlon"
        and args.agent_system_prompt != "toolathlon"
    ):
        parser.error(
            "Toolathlon's native TaskAgent always uses the task-specific "
            "system prompt"
        )
    if args.subagent_recursion_limit < 1:
        parser.error("--subagent-recursion-limit must be at least 1")
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
    if args.subagent_ports:
        if len(set(args.subagent_ports)) != len(args.subagent_ports):
            parser.error("--subagent-ports must not contain duplicates")
        args.reuse_vllm = True
    else:
        args.subagent_ports = [args.subagent_port]
    if args.subagent_base_url and len(args.subagent_ports) != 1:
        parser.error("--subagent-base-url requires exactly one subagent port")
    return args


def select_tasks(
    tasks_root: Path,
    run_all: bool,
    run_all_valid: bool,
    requested: Sequence[str] | None,
) -> list[str]:
    available = sorted(
        f"{domain.name}/{task.name}"
        for domain in tasks_root.iterdir()
        if domain.is_dir() and not domain.name.startswith(".")
        for task in domain.iterdir()
        if task.is_dir() and not task.name.startswith(".")
    )
    if run_all:
        return available
    if run_all_valid:
        missing = [task for task in VALID_EVALUATION_TASKS if task not in available]
        if missing:
            raise ValueError(
                "Infrastructure-valid Toolathlon task(s) are missing: "
                + ", ".join(missing)
            )
        return list(VALID_EVALUATION_TASKS)
    selected = list(requested or ())
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate Toolathlon tasks are not allowed")
    unknown = [task for task in selected if task not in available]
    if unknown:
        raise ValueError(f"Unknown Toolathlon task(s): {', '.join(unknown)}")
    return selected


def shared_task_resources(task: str) -> tuple[str, ...]:
    resources = []
    if task.startswith("finalpool/canvas-"):
        resources.append("canvas")
    if task.startswith("finalpool/k8s-"):
        resources.append("k8s")
    return tuple(resources)


def count_statuses(manifest: dict[str, Any]) -> dict[str, int]:
    counts = {name: 0 for name in ("pending", "running", "completed", "failed")}
    for episode in manifest["episodes"]:
        counts[episode["status"]] += 1
    counts["total"] = len(manifest["episodes"])
    return counts


def evaluation_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute pass@1, pass@3 and pass^3 without hiding missing scores."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for episode in manifest["episodes"]:
        by_task.setdefault(episode["task"], []).append(episode)
    repetitions = manifest["config"]["repetitions"]
    scored_tasks = [
        episodes
        for episodes in by_task.values()
        if len(episodes) == repetitions
        and all(isinstance(episode.get("score"), bool) for episode in episodes)
    ]
    scored_trials = sum(len(episodes) for episodes in scored_tasks)
    unscored_trials = sum(
        not isinstance(episode.get("score"), bool)
        for episode in manifest["episodes"]
    )
    passed_trials = sum(
        episode["score"] is True
        for episodes in scored_tasks
        for episode in episodes
    )
    benchmark_task_count = manifest["config"].get("benchmark_task_count", len(by_task))
    unrun_tasks_are_failures = manifest["config"].get(
        "unrun_tasks_are_failures", False
    )
    result: dict[str, Any] = {
        "task_count": len(by_task),
        "benchmark_task_count": benchmark_task_count,
        "unrun_task_count": max(benchmark_task_count - len(by_task), 0),
        "repetitions": repetitions,
        "expected_trials": benchmark_task_count * repetitions,
        "scored_task_count": len(scored_tasks),
        "scored_trials": scored_trials,
        "unscored_trials": unscored_trials,
        "pass@1": passed_trials / scored_trials if scored_trials else None,
        "pass@3": None,
        "pass^3": None,
    }
    if unrun_tasks_are_failures:
        all_passed_trials = sum(
            episode.get("score") is True for episode in manifest["episodes"]
        )
        result["assumed_failed_task_count"] = max(
            benchmark_task_count - len(by_task), 0
        )
        result["pass@1"] = all_passed_trials / (benchmark_task_count * repetitions)
        if repetitions == 3:
            result["pass@3"] = sum(
                any(episode.get("score") is True for episode in episodes)
                for episodes in by_task.values()
            ) / benchmark_task_count
            result["pass^3"] = sum(
                len(episodes) == repetitions
                and all(episode.get("score") is True for episode in episodes)
                for episodes in by_task.values()
            ) / benchmark_task_count
        return result
    if unscored_trials:
        # A partial focused run must never look better merely because an
        # incomplete task disappeared from the denominator.  Keep counts for
        # diagnostics, but withhold all aggregate rates until every selected
        # trial has a native boolean score.
        result.update({"pass@1": None, "pass@3": None, "pass^3": None})
        return result
    if repetitions == 3 and scored_tasks:
        result["pass@3"] = sum(
            any(episode["score"] is True for episode in episodes)
            for episodes in scored_tasks
        ) / len(scored_tasks)
        result["pass^3"] = sum(
            all(episode["score"] is True for episode in episodes)
            for episodes in scored_tasks
        ) / len(scored_tasks)
    return result


def save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    manifest["counts"] = count_statuses(manifest)
    write_json(run_dir / "manifest.json", manifest)


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Cannot resume: manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema_version')!r}")
    return manifest


def create_manifest(
    run_id: str, tasks: Sequence[str], repetitions: int, args: argparse.Namespace
) -> dict[str, Any]:
    episodes = [
        {
            "key": f"{task}::rep-{repetition:03d}",
            "task": task,
            "repetition": repetition,
            "status": "pending",
            "score": None,
            "artifact_path": None,
            "evaluation_path": None,
            "started_at": None,
            "finished_at": None,
            "duration_seconds": None,
            "error": None,
            "attempts": [],
        }
        # Schedule one full benchmark round at a time.  Toolathlon tasks share
        # backing services, so launching several repetitions of the same task
        # together races their preprocess/cleanup state.
        for repetition in range(1, repetitions + 1)
        for task in tasks
    ]
    config = {name: getattr(args, name) for name in RESUME_CONFIG_FIELDS}
    config.update(tasks=list(tasks), repetitions=repetitions, purpose=args.purpose)
    now = utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "finished_at": None,
        "config": config,
        "episodes": episodes,
    }
    manifest["counts"] = count_statuses(manifest)
    return manifest


def validate_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError(f"Invalid run ID: {run_id!r}")


def episode_id_for(run_id: str, task: str, repetition: int, attempt: int) -> str:
    task_digest = hashlib.sha256(task.encode()).hexdigest()[:8]
    return f"{run_id}-{task_digest}-r{repetition:03d}-a{attempt:03d}"


def episode_command(
    args: argparse.Namespace,
    runner_path: Path,
    task: str,
    run_id: str,
    repetition: int,
    attempt: int,
    episode_id: str,
    root: Path,
    subagent_port: int | None = None,
    container_slot: int = 0,
) -> list[str]:
    port = args.subagent_port if subagent_port is None else subagent_port
    container_slots = getattr(args, "container_slots", 1)
    command = [
        sys.executable, str(runner_path), task,
        "--episode-id", episode_id, "--run-id", run_id,
        "--repetition", str(repetition), "--attempt", str(attempt),
        "--purpose", args.purpose, "--model", args.model,
        "--agent-mode", args.agent_mode,
        "--simple-agent-implementation", getattr(
            args, "simple_agent_implementation", "toolathlon"
        ),
        "--agent-system-prompt", getattr(
            args, "agent_system_prompt", "toolathlon"
        ),
        "--decomposer-provider", getattr(args, "decomposer_provider", "openrouter"),
        "--decomposer-base-url",
        getattr(args, "decomposer_base_url", "http://127.0.0.1:8040/v1"),
        "--decomposer-prompt", getattr(args, "decomposer_prompt", "teacher"),
        "--subagent-provider", getattr(args, "subagent_provider", "vllm"),
        "--subagent-model", args.subagent_model,
        "--subagent-port", str(port),
        "--subagent-gpu", args.subagent_gpu,
        "--vllm-max-model-len", str(args.vllm_max_model_len),
        "--subagent-recursion-limit",
        str(getattr(args, "subagent_recursion_limit", SUBAGENT_RECURSION_LIMIT)),
        "--vllm-gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
        "--vllm-data-parallel-size", str(getattr(args, "vllm_data_parallel_size", 1)),
        "--vllm-startup-timeout", str(args.vllm_startup_timeout),
        "--reuse-vllm", "--image", args.image,
        "--artifacts-dir", str(root / "traces"),
        "--evals-dir", str(root / "evals"),
        "--stashes-dir", str(root / "stashes"),
        "--startup-timeout", str(args.startup_timeout),
        "--n-jobs-per-worker", str(args.n_jobs_per_worker),
        "--agent-timeout", str(getattr(args, "agent_timeout", 2700)),
        "--max-steps", str(args.max_steps),
        "--max-tool-output-chars", str(
            getattr(args, "max_tool_output_chars", 100_000)
        ),
        "--eval-config", args.eval_config,
        "--container-lock-file",
        str(
            root
            / "runs"
            / run_id
            / (
                "container.lock"
                if container_slots == 1
                else f"container-{container_slot:02d}.lock"
            )
        ),
    ]
    if getattr(args, "publish_service_ports", False):
        command.append("--publish-service-ports")
    if getattr(args, "subagent_base_url", None) is not None:
        command.extend(["--subagent-base-url", args.subagent_base_url])
    if args.docker_socket is not None:
        command.extend(["--docker-socket", args.docker_socket])
    return command


def execute_episode(
    args: argparse.Namespace,
    runner_path: Path,
    root: Path,
    run_dir: Path,
    episode: dict[str, Any],
    attempt: int,
    stop_event: threading.Event | None = None,
    subagent_port: int | None = None,
    container_slot: int = 0,
) -> dict[str, Any]:
    task, repetition = episode["task"], episode["repetition"]
    attempt_dir = run_dir / "attempts" / task / f"rep-{repetition:03d}" / f"attempt-{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    episode_id = episode_id_for(run_dir.name, task, repetition, attempt)
    command = episode_command(
        args,
        runner_path,
        task,
        run_dir.name,
        repetition,
        attempt,
        episode_id,
        root,
        subagent_port,
        container_slot,
    )
    started_at, started = utc_now(), time.monotonic()
    timed_out = False
    write_json(attempt_dir / "attempt.json", {
        "status": "running", "task": task, "repetition": repetition,
        "attempt": attempt, "command": command, "started_at": started_at,
    })
    with (attempt_dir / "runner.stdout.log").open("wb") as stdout, (
        attempt_dir / "runner.stderr.log"
    ).open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        try:
            while True:
                try:
                    returncode = process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= getattr(
                        args, "episode_timeout", 6000
                    ):
                        timed_out = True
                        process.send_signal(signal.SIGINT)
                        try:
                            returncode = process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            returncode = process.wait()
                        break
                    if stop_event is not None and stop_event.is_set():
                        raise KeyboardInterrupt("batch interrupted")
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

    artifact_dir = root / "traces" / task / episode_id
    evaluation_path = root / "evals" / task / episode_id / "result.json"
    score = None
    artifact_score = None
    if evaluation_path.is_file():
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            score = evaluation.get("pass")
            artifact_score = evaluation.get("artifact_pass", score)
        except (json.JSONDecodeError, OSError):
            pass
    # A model can terminate without a textual final answer, hit its recursion
    # limit, or exhaust its completion budget after still leaving a valid
    # native evaluation result.  Those are benchmark failures, not missing
    # infrastructure episodes: retain the model's false score and complete the
    # episode.  Infrastructure failures do not produce a boolean native score.
    completed = evaluation_path.is_file() and isinstance(score, bool)
    error = None
    if not completed:
        stderr_tail = (attempt_dir / "runner.stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )[-8000:]
        error = {
            "type": "EpisodeTimeout" if timed_out else "EpisodeProcessError",
            "message": (
                f"Episode exceeded {getattr(args, 'episode_timeout', 6000):g}s"
                if timed_out
                else f"Episode runner exited with code {returncode}"
            ),
            "returncode": returncode,
            "stderr_tail": stderr_tail,
        }
    result = {
        "attempt": attempt,
        "status": "completed" if completed else "failed",
        "score": score,
        "artifact_score": artifact_score,
        "artifact_path": str(artifact_dir if artifact_dir.is_dir() else attempt_dir),
        "attempt_log_path": str(attempt_dir),
        "evaluation_path": str(evaluation_path) if evaluation_path.is_file() else None,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.monotonic() - started,
        "returncode": returncode,
        "error": error,
    }
    write_json(attempt_dir / "attempt.json", {**result, "command": command})
    return result


def next_attempt(run_dir: Path, episode: dict[str, Any]) -> tuple[int, bool]:
    """Account for attempt directories left by an abrupt parent exit."""
    parent = run_dir / "attempts" / episode["task"] / f"rep-{episode['repetition']:03d}"
    known = {item["attempt"] for item in episode["attempts"]}
    changed = False
    for path in sorted(parent.glob("attempt-*")) if parent.is_dir() else ():
        try:
            number = int(path.name.removeprefix("attempt-"))
        except ValueError:
            continue
        if number in known:
            continue
        recovered = {
            "attempt": number,
            "status": "failed",
            "score": None,
            "artifact_path": str(path),
            "attempt_log_path": str(path),
            "evaluation_path": None,
            "finished_at": utc_now(),
            "error": {
                "type": "RecoveredIncompleteAttempt",
                "message": "Parent stopped before recording this attempt",
                "interrupted": True,
            },
        }
        episode["attempts"].append(recovered)
        episode.update(recovered)
        known.add(number)
        changed = True
    episode["attempts"].sort(key=lambda item: item["attempt"])
    highest = max((item["attempt"] for item in episode["attempts"]), default=0)
    return highest + 1, changed


def failure(attempt: int, error: BaseException, started_at: str | None) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "status": "failed",
        "score": None,
        "artifact_path": None,
        "evaluation_path": None,
        "started_at": started_at,
        "finished_at": utc_now(),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    }


def main(
    argv: Sequence[str],
    *,
    repo_root: Path,
    toolathlon_root: Path,
    default_artifacts_dir: Path,
    default_image: str,
    default_model: str,
    default_subagent_model: str,
    default_subagent_port: int,
    start_vllm: Callable[..., subprocess.Popen[bytes] | None],
    stop_vllm: Callable[[subprocess.Popen[bytes] | None], None],
    docker: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    defaults = {
        "artifacts_dir": default_artifacts_dir,
        "image": default_image,
        "model": default_model,
        "subagent_model": default_subagent_model,
        "subagent_port": default_subagent_port,
    }
    args = parse_args(argv, defaults)
    if not (toolathlon_root / "tasks").is_dir():
        raise RuntimeError(
            f"Toolathlon checkout at {toolathlon_root} is incomplete (missing "
            "tasks/); run: git submodule update --init external/toolathlon"
        )
    root = args.bench_artifacts_dir.resolve()
    if args.resume:
        validate_run_id(args.resume)
        run_dir = root / "runs" / args.resume
        manifest = load_manifest(run_dir)
        if args.purpose != manifest["config"]["purpose"]:
            raise ValueError("Resume purpose does not match the manifest")
        for name in RESUME_CONFIG_FIELDS:
            if name in {
                "agent_mode", "decomposer_provider", "subagent_provider"
            } and name not in manifest["config"]:
                defaults_by_name = {
                    "agent_mode": "decomposer",
                    "decomposer_provider": "openrouter",
                    "subagent_provider": "vllm",
                }
                setattr(args, name, defaults_by_name[name])
            elif name == "agent_system_prompt" and name not in manifest["config"]:
                setattr(args, name, "generic")
            elif (
                name == "simple_agent_implementation"
                and name not in manifest["config"]
            ):
                setattr(args, name, "langgraph")
            elif name == "decomposer_base_url" and name not in manifest["config"]:
                setattr(args, name, "http://127.0.0.1:8040/v1")
            elif name == "decomposer_prompt" and name not in manifest["config"]:
                setattr(args, name, "teacher")
            elif name == "subagent_ports" and name not in manifest["config"]:
                setattr(args, name, [manifest["config"]["subagent_port"]])
            elif name == "subagent_base_url" and name not in manifest["config"]:
                setattr(args, name, None)
            elif name == "publish_service_ports" and name not in manifest["config"]:
                setattr(args, name, False)
            elif name == "subagent_recursion_limit" and name not in manifest["config"]:
                setattr(args, name, SUBAGENT_RECURSION_LIMIT)
            elif name == "container_slots" and name not in manifest["config"]:
                setattr(args, name, 1)
            elif name == "episode_timeout" and name not in manifest["config"]:
                setattr(args, name, 6000)
            elif name == "agent_timeout" and name not in manifest["config"]:
                setattr(args, name, 2700)
            elif name == "max_tool_output_chars" and name not in manifest["config"]:
                setattr(args, name, 8_000)
            else:
                setattr(args, name, manifest["config"][name])
    else:
        tasks_root = toolathlon_root / "tasks"
        tasks = select_tasks(tasks_root, args.all, args.all_valid, args.tasks)
        all_tasks = select_tasks(tasks_root, True, False, None)
        run_dir = root / "runs" / new_run_id()
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = create_manifest(run_dir.name, tasks, args.repetitions, args)
        manifest["config"].update(
            benchmark_task_count=len(all_tasks) if args.all_valid else len(tasks),
            unrun_tasks_are_failures=args.all_valid,
            assumed_failed_tasks=(
                sorted(set(all_tasks) - set(tasks)) if args.all_valid else []
            ),
        )
        save_manifest(run_dir, manifest)
        append_event(run_dir, "run_created", tasks=tasks, repetitions=args.repetitions)

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
    docker("image", "inspect", args.image)
    manifest.update(status="running", finished_at=None)
    manifest.setdefault("invocations", []).append(
        {
            "timestamp": utc_now(),
            "argv": sys.argv,
            "hostname": platform.node(),
            "python": sys.version,
            "code": git_provenance(repo_root),
        }
    )
    save_manifest(run_dir, manifest)
    append_event(run_dir, "batch_started", resume=bool(args.resume))
    print(f"Run {run_dir.name}: {manifest['counts']['total']} episode(s)", flush=True)

    processes: list[subprocess.Popen[bytes] | None] = []
    interrupted = False
    try:
        if args.subagent_provider == "vllm":
            for port in args.subagent_ports:
                processes.append(
                    start_vllm(
                        model=args.subagent_model,
                        port=port,
                        gpu=args.subagent_gpu,
                        max_model_len=args.vllm_max_model_len,
                        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                        timeout=args.vllm_startup_timeout,
                        log_path=run_dir / f"vllm-{port}.log",
                        reuse=args.reuse_vllm,
                        data_parallel_size=args.vllm_data_parallel_size,
                    )
                )
        append_event(
            run_dir,
            "vllm_ready",
            externally_managed=args.reuse_vllm,
            ports=args.subagent_ports,
        )
        work: list[tuple[int, dict[str, Any], int]] = []
        for index, episode in enumerate(manifest["episodes"], start=1):
            attempt, reconciled = next_attempt(run_dir, episode)
            if reconciled:
                save_manifest(run_dir, manifest)
                append_event(run_dir, "episode_reconciled", key=episode["key"])
            if episode["status"] == "completed":
                append_event(run_dir, "episode_skipped", key=episode["key"])
                continue
            work.append((index, episode, attempt))

        stop_event = threading.Event()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency,
            thread_name_prefix="toolathlon-episode",
        )
        active: dict[
            concurrent.futures.Future[dict[str, Any]],
            tuple[int, dict[str, Any], int, tuple[str, ...], int],
        ] = {}
        remaining = list(work)
        next_endpoint = 0
        active_tasks: set[str] = set()
        active_resources: set[str] = set()
        available_container_slots = set(range(args.container_slots))
        task_resources = {
            task: shared_task_resources(task)
            for task in {episode["task"] for episode in manifest["episodes"]}
        }

        def record_result(
            episode: dict[str, Any], result: dict[str, Any], *, interrupted_run: bool = False
        ) -> None:
            if interrupted_run:
                result["error"] = result.get("error") or {}
                result["error"]["interrupted"] = True
            episode["attempts"].append(result)
            episode.update(result)
            save_manifest(run_dir, manifest)
            append_event(
                run_dir,
                "episode_interrupted"
                if interrupted_run
                else f"episode_{result['status']}",
                key=episode["key"],
                attempt=result["attempt"],
                score=result.get("score"),
            )

        def submit_next() -> bool:
            nonlocal next_endpoint
            if not available_container_slots:
                return False
            selected = None
            for position, (index, episode, attempt) in enumerate(remaining):
                resources = task_resources[episode["task"]]
                if episode["task"] in active_tasks:
                    continue
                if active_resources.intersection(resources):
                    continue
                selected = position, index, episode, attempt, resources
                break
            if selected is None:
                return False
            position, index, episode, attempt, resources = selected
            remaining.pop(position)
            container_slot = min(available_container_slots)
            available_container_slots.remove(container_slot)
            print(
                f"[{index}/{manifest['counts']['total']}] {episode['key']} "
                f"attempt {attempt}",
                flush=True,
            )
            episode.update(status="running", started_at=utc_now(), finished_at=None, error=None)
            save_manifest(run_dir, manifest)
            append_event(run_dir, "episode_started", key=episode["key"], attempt=attempt)
            future = executor.submit(
                execute_episode,
                args,
                runner_path=repo_root / "gyms" / "toolathlon" / "run.py",
                root=root,
                run_dir=run_dir,
                episode=episode,
                attempt=attempt,
                stop_event=stop_event,
                subagent_port=args.subagent_ports[next_endpoint],
                container_slot=container_slot,
            )
            next_endpoint = (next_endpoint + 1) % len(args.subagent_ports)
            active_tasks.add(episode["task"])
            active_resources.update(resources)
            active[future] = (index, episode, attempt, resources, container_slot)
            return True

        try:
            for _ in range(min(args.concurrency, len(work))):
                submit_next()
            while active:
                done, _ = concurrent.futures.wait(
                    active,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    _index, episode, attempt, resources, container_slot = active.pop(future)
                    active_tasks.remove(episode["task"])
                    active_resources.difference_update(resources)
                    available_container_slots.add(container_slot)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = failure(attempt, error, episode["started_at"])
                    except BaseException as error:
                        result = failure(attempt, error, episode["started_at"])
                        record_result(episode, result, interrupted_run=True)
                        raise
                    record_result(episode, result)
                    submit_next()
        except BaseException:
            stop_event.set()
            for future, (
                _index,
                episode,
                attempt,
                _resources,
                _container_slot,
            ) in list(active.items()):
                try:
                    result = future.result()
                except BaseException as error:
                    result = failure(attempt, error, episode["started_at"])
                record_result(
                    episode,
                    result,
                    interrupted_run=result["status"] != "completed",
                )
            active.clear()
            raise
        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
    except BaseException as error:
        interrupted = True
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        for process in processes:
            stop_vllm(process)
        append_event(
            run_dir,
            "vllm_stopped",
            externally_managed=args.reuse_vllm,
            ports=args.subagent_ports,
        )
        incomplete = any(item["status"] != "completed" for item in manifest["episodes"])
        manifest["status"] = (
            "interrupted"
            if interrupted
            else "completed_with_errors"
            if incomplete
            else "completed"
        )
        manifest["finished_at"] = utc_now()
        manifest["metrics"] = evaluation_metrics(manifest)
        write_json(run_dir / "metrics.json", manifest["metrics"])
        save_manifest(run_dir, manifest)
        append_event(
            run_dir,
            "batch_finished",
            status=manifest["status"],
            counts=manifest["counts"],
        )

    print(f"Run manifest: {run_dir / 'manifest.json'}", flush=True)
    return manifest
