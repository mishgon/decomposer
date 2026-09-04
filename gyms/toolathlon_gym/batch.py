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

try:
    from . import adaptive_scheduler
except ImportError:  # Executed through gyms/toolathlon_gym/run.py.
    import adaptive_scheduler


SCHEMA_VERSION = 1
OPENROUTER_TRANSIENT_MARKERS = (
    "httpx.writetimeout",
    "writetimeout",
    "httpx.readtimeout",
    "readtimeout",
    "httpx.connecttimeout",
    "connecttimeout",
    "httpx.pooltimeout",
    "pooltimeout",
    "httpx.remoteprotocolerror",
    "remoteprotocolerror",
    "httpx.connecterror",
    "connecterror",
    "apiconnectionerror",
    "ratelimiterror",
    "the run is no longer active",
    "status code: 429",
    "status code: 502",
    "status code: 503",
    "status code: 504",
)
RESUME_CONFIG_FIELDS = (
    "model",
    "subagent_model",
    "subagent_api_model",
    "subagent_port",
    "subagent_ports",
    "subagent_gpu",
    "image",
    "reuse_vllm",
    "vllm_max_model_len",
    "vllm_gpu_memory_utilization",
    "vllm_data_parallel_size",
    "vllm_startup_timeout",
    "startup_timeout",
    "n_jobs_per_worker",
    "container_slots",
    "concurrency",
    "agent_timeout",
    "episode_timeout",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    return stamp + uuid.uuid4().hex[:8]


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


class ProviderBackoffBarrier:
    """Shared launch barrier for a provider outage observed by many episodes."""

    def __init__(
        self,
        *,
        initial_seconds: float = 30.0,
        maximum_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.initial_seconds = initial_seconds
        self.maximum_seconds = maximum_seconds
        self._clock = clock
        self._condition = threading.Condition()
        self._generation = 0
        self._consecutive_failures = 0
        self._blocked_until = 0.0

    def generation(self) -> int:
        with self._condition:
            return self._generation

    def register_failure(self, generation: int) -> float | None:
        """Close the barrier once per outage generation and return its delay."""
        with self._condition:
            if generation != self._generation:
                return None
            self._consecutive_failures += 1
            delay = min(
                self.maximum_seconds,
                self.initial_seconds * (2 ** (self._consecutive_failures - 1)),
            )
            self._generation += 1
            self._blocked_until = self._clock() + delay
            self._condition.notify_all()
            return delay

    def register_success(self, generation: int) -> bool:
        """Reset escalation only after work from the current generation succeeds."""
        with self._condition:
            if generation != self._generation or not self._consecutive_failures:
                return False
            self._consecutive_failures = 0
            self._blocked_until = 0.0
            self._condition.notify_all()
            return True

    def wait(self, stop_event: threading.Event | None = None) -> int:
        with self._condition:
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise KeyboardInterrupt("batch interrupted during provider backoff")
                remaining = self._blocked_until - self._clock()
                if remaining <= 0:
                    return self._generation
                self._condition.wait(timeout=min(remaining, 1.0))


def openrouter_transient_failure(result: dict[str, Any]) -> str | None:
    """Return evidence only for transient failures in the Decomposer provider path."""
    if result.get("status") == "completed":
        return None
    evidence: list[str] = []
    error = result.get("error")
    if isinstance(error, dict):
        for key in ("message", "stderr_tail", "traceback"):
            value = error.get(key)
            if isinstance(value, str):
                evidence.append(value)
    artifact = result.get("artifact_path")
    if artifact:
        trace_path = Path(artifact) / "trace.json"
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            agent_error = trace.get("agent_error")
            if isinstance(agent_error, str):
                evidence.insert(0, agent_error)
        except (OSError, json.JSONDecodeError):
            pass
    combined = "\n".join(evidence)
    lowered = combined.lower()
    teacher_path = (
        "decomposer agent loop failed" in lowered
        or "agent_error" in lowered
        or bool(artifact and (Path(artifact) / "trace.json").is_file())
    )
    if not teacher_path:
        return None
    for marker in OPENROUTER_TRANSIENT_MARKERS:
        if marker in lowered:
            return marker
    return None


def validate_teacher_credentials() -> None:
    if os.environ.get("DECOMPOSER_VLLM_BASE_URL"):
        return
    proxy_url = os.environ.get("LLM_PROXY_URL")
    proxy_key = os.environ.get("LLM_PROXY_MASTER_KEY")
    if proxy_url or proxy_key:
        if not (proxy_url and proxy_key):
            raise RuntimeError("Set both LLM_PROXY_URL and LLM_PROXY_MASTER_KEY")
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "Set OPENROUTER_API_KEY, or configure LLM_PROXY_URL and "
            "LLM_PROXY_MASTER_KEY for the Decomposer model"
        )


def wants_batch(argv: Sequence[str]) -> bool:
    flags = {"--all", "--tasks", "--resume", "--repetitions", "-n", "--adaptive"}
    prefixes = ("--tasks=", "--resume=", "--repetitions=", "-n")
    return any(
        argument in flags or argument.startswith(prefixes) for argument in argv
    )


def parse_args(argv: Sequence[str], defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable Decomposer batch on Toolathlon Gym."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--tasks", nargs="+", metavar="TASK")
    parser.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("-n", "--repetitions", type=int, default=1)
    parser.add_argument("--purpose", choices=("trace-generation",), required=True)
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--subagent-model", default=defaults["subagent_model"])
    parser.add_argument(
        "--subagent-api-model",
        default=defaults.get("subagent_api_model", defaults["subagent_model"]),
    )
    parser.add_argument("--subagent-port", type=int, default=defaults["subagent_port"])
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
    parser.add_argument("--vllm-max-model-len", type=int, default=256000)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-data-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-startup-timeout", type=float, default=1800)
    parser.add_argument("--reuse-vllm", action="store_true")
    parser.add_argument("--image", default=defaults["image"])
    parser.add_argument(
        "--gym-artifacts-dir", type=Path, default=defaults["artifacts_dir"]
    )
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--n-jobs-per-worker", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--container-slots", type=int, default=1)
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Use balanced 1+2+3 trace scheduling and task culling. Model and "
            "infrastructure settings retain their normal batch semantics."
        ),
    )
    parser.add_argument(
        "--adaptive-success-threshold",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Qualify partial scores strictly above this fraction (default: 0.90).",
    )
    parser.add_argument(
        "--adaptive-cull-fraction",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Bottom zero-success fraction culled after three launches (default: 0.10).",
    )
    parser.add_argument(
        "--adaptive-target-successes",
        type=int,
        default=None,
        metavar="COUNT",
        help="Balanced qualifying traces requested per retained task (default: 4).",
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=2700,
        help="Maximum seconds spent in the Decomposer loop (default: 2700).",
    )
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=3300,
        help="Maximum total wall-clock seconds for one episode (default: 3300).",
    )
    args = parser.parse_args(argv)
    if args.resume and (args.all or args.tasks):
        parser.error("--resume cannot be combined with --all or --tasks")
    if not args.resume and not (args.all or args.tasks):
        parser.error("choose --all, --tasks, or --resume")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.resume and args.repetitions != 1:
        parser.error("--repetitions cannot be changed on resume")
    if args.adaptive and not args.resume and args.repetitions != 1:
        parser.error("a new --adaptive run must start with one repetition")
    if args.adaptive_success_threshold is not None and not (
        0 <= args.adaptive_success_threshold <= 1
    ):
        parser.error("--adaptive-success-threshold must be in [0, 1]")
    if args.adaptive_cull_fraction is not None and not (
        0 <= args.adaptive_cull_fraction < 1
    ):
        parser.error("--adaptive-cull-fraction must be in [0, 1)")
    if (
        args.adaptive_target_successes is not None
        and args.adaptive_target_successes < 1
    ):
        parser.error("--adaptive-target-successes must be positive")
    if args.n_jobs_per_worker < 1:
        parser.error("--n-jobs-per-worker must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.container_slots < 1:
        parser.error("--container-slots must be at least 1")
    if args.agent_timeout <= 0 or args.episode_timeout <= 0:
        parser.error("--agent-timeout and --episode-timeout must be positive")
    if args.vllm_data_parallel_size < 1:
        parser.error("--vllm-data-parallel-size must be at least 1")
    visible_gpus = [item for item in args.subagent_gpu.split(",") if item]
    if len(visible_gpus) != args.vllm_data_parallel_size:
        parser.error(
            "--subagent-gpu must list exactly --vllm-data-parallel-size GPU IDs"
        )
    if args.subagent_ports:
        if len(set(args.subagent_ports)) != len(args.subagent_ports):
            parser.error("--subagent-ports must not contain duplicates")
        args.reuse_vllm = True
    else:
        args.subagent_ports = [args.subagent_port]
    return args


def select_tasks(tasks_dir: Path, run_all: bool, requested: Sequence[str] | None) -> list[str]:
    available = sorted(
        path.name
        for path in tasks_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if run_all:
        return available
    selected = list(requested or ())
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate Toolathlon tasks are not allowed")
    unknown = [task for task in selected if task not in available]
    if unknown:
        raise ValueError(f"Unknown Toolathlon task(s): {', '.join(unknown)}")
    return selected


def count_statuses(manifest: dict[str, Any]) -> dict[str, int]:
    counts = {name: 0 for name in ("pending", "running", "completed", "failed")}
    for episode in manifest["episodes"]:
        counts[episode["status"]] += 1
    counts["total"] = len(manifest["episodes"])
    return counts


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
        for task in tasks
        for repetition in range(1, repetitions + 1)
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


def configure_adaptive_scheduler(
    manifest: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    requested = {
        "success_threshold": (
            0.90
            if args.adaptive_success_threshold is None
            else args.adaptive_success_threshold
        ),
        "cull_fraction": (
            0.10
            if args.adaptive_cull_fraction is None
            else args.adaptive_cull_fraction
        ),
        "target_successes": (
            4
            if args.adaptive_target_successes is None
            else args.adaptive_target_successes
        ),
    }
    state = manifest.get("adaptive_scheduler")
    if state is None:
        state = adaptive_scheduler.new_scheduler_state(
            manifest["config"]["tasks"],
            threshold=requested["success_threshold"],
            cull_fraction=requested["cull_fraction"],
            target_successes=requested["target_successes"],
        )
        manifest["adaptive_scheduler"] = state
        return state

    if state.get("schema_version") != adaptive_scheduler.SCHEDULER_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported adaptive scheduler schema: "
            f"{state.get('schema_version')!r}"
        )
    for name, default_value in requested.items():
        if name not in state:
            state[name] = default_value
        cli_value = getattr(args, f"adaptive_{name}")
        if cli_value is not None and cli_value != state[name]:
            raise ValueError(
                f"Cannot change adaptive {name.replace('_', ' ')} after the "
                f"scheduler starts ({state[name]!r} != {cli_value!r})"
            )
    return state


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
    return [
        sys.executable, str(runner_path), task,
        "--episode-id", episode_id, "--run-id", run_id,
        "--repetition", str(repetition), "--attempt", str(attempt),
        "--purpose", args.purpose, "--model", args.model,
        "--subagent-model", args.subagent_model,
        "--subagent-api-model",
        getattr(args, "subagent_api_model", args.subagent_model),
        "--subagent-port", str(port),
        "--subagent-gpu", args.subagent_gpu,
        "--vllm-max-model-len", str(args.vllm_max_model_len),
        "--vllm-gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
        "--vllm-data-parallel-size", str(getattr(args, "vllm_data_parallel_size", 1)),
        "--vllm-startup-timeout", str(args.vllm_startup_timeout),
        "--reuse-vllm", "--image", args.image,
        "--artifacts-dir", str(root / "traces"),
        "--evals-dir", str(root / "evals"),
        "--startup-timeout", str(args.startup_timeout),
        "--n-jobs-per-worker", str(args.n_jobs_per_worker),
        "--agent-timeout", str(getattr(args, "agent_timeout", 2700)),
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
    write_json(attempt_dir / "attempt.json", {
        "status": "running", "task": task, "repetition": repetition,
        "attempt": attempt, "command": command, "started_at": started_at,
    })
    with (attempt_dir / "runner.stdout.log").open("wb") as stdout, (
        attempt_dir / "runner.stderr.log"
    ).open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        timed_out = False
        try:
            while True:
                try:
                    returncode = process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= getattr(
                        args, "episode_timeout", 2400
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
    partial_score = None
    if evaluation_path.is_file():
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            score = evaluation.get("pass")
            parsed = adaptive_scheduler.extract_partial_score(evaluation)
            if parsed is not None:
                partial_score = {
                    "passed_checks": parsed.passed_checks,
                    "total_checks": parsed.total_checks,
                    "fraction": parsed.fraction,
                    "source": parsed.source,
                }
        except (json.JSONDecodeError, OSError):
            pass
    completed = not timed_out and returncode == 0 and evaluation_path.is_file()
    error = None
    if not completed:
        stderr_tail = (attempt_dir / "runner.stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )[-8000:]
        error = {
            "type": "EpisodeTimeout" if timed_out else "EpisodeProcessError",
            "message": (
                f"Episode exceeded {getattr(args, 'episode_timeout', 2400):g}s"
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
        "partial_score": partial_score,
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
    default_subagent_api_model: str | None = None,
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
        "subagent_api_model": default_subagent_api_model or default_subagent_model,
        "subagent_port": default_subagent_port,
    }
    args = parse_args(argv, defaults)
    root = args.gym_artifacts_dir.resolve()
    if args.resume:
        validate_run_id(args.resume)
        run_dir = root / "runs" / args.resume
        manifest = load_manifest(run_dir)
        if args.purpose != manifest["config"]["purpose"]:
            raise ValueError("Resume purpose does not match the manifest")
        for name in RESUME_CONFIG_FIELDS:
            if name == "subagent_ports" and name not in manifest["config"]:
                setattr(args, name, [manifest["config"]["subagent_port"]])
            elif name == "vllm_data_parallel_size" and name not in manifest["config"]:
                setattr(args, name, 1)
            elif name == "container_slots" and name not in manifest["config"]:
                setattr(args, name, 1)
            elif name == "agent_timeout" and name not in manifest["config"]:
                setattr(args, name, 1800)
            elif name == "episode_timeout" and name not in manifest["config"]:
                setattr(args, name, 2400)
            elif (
                name in {"n_jobs_per_worker", "concurrency"}
                and name not in manifest["config"]
            ):
                # Older manifests did not persist these scheduler settings.
                # Keep the resume-time CLI value or parser default.
                continue
            else:
                setattr(args, name, manifest["config"][name])
    else:
        tasks = select_tasks(
            toolathlon_root / "tasks" / "finalpool", args.all, args.tasks
        )
        run_dir = root / "runs" / new_run_id()
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = create_manifest(run_dir.name, tasks, args.repetitions, args)
        save_manifest(run_dir, manifest)
        append_event(run_dir, "run_created", tasks=tasks, repetitions=args.repetitions)

    if args.adaptive:
        state = configure_adaptive_scheduler(manifest, args)
        save_manifest(run_dir, manifest)
        append_event(
            run_dir,
            "adaptive_scheduler_enabled",
            phase=state["phase"],
            success_threshold=state["success_threshold"],
            cull_fraction=state["cull_fraction"],
            target_successes=state["target_successes"],
        )

    validate_teacher_credentials()
    docker("image", "inspect", args.image)
    manifest.update(status="running", finished_at=None)
    manifest.setdefault("invocations", []).append(
        {
            "timestamp": utc_now(),
            "argv": sys.argv,
            "hostname": platform.node(),
            "python": sys.version,
        }
    )
    save_manifest(run_dir, manifest)
    append_event(run_dir, "batch_started", resume=bool(args.resume))
    print(f"Run {run_dir.name}: {manifest['counts']['total']} episode(s)", flush=True)

    processes: list[subprocess.Popen[bytes] | None] = []
    interrupted = False
    try:
        for port in args.subagent_ports:
            processes.append(
                start_vllm(
                    model=args.subagent_model,
                    served_model_name=args.subagent_api_model,
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
        provider_backoff = ProviderBackoffBarrier()
        next_endpoint = 0
        while True:
            work: list[tuple[int, dict[str, Any], int]] = []
            for index, episode in enumerate(manifest["episodes"], start=1):
                attempt, reconciled = next_attempt(run_dir, episode)
                if reconciled:
                    save_manifest(run_dir, manifest)
                    append_event(run_dir, "episode_reconciled", key=episode["key"])
                terminal = episode["status"] in (
                    adaptive_scheduler.TERMINAL_EPISODE_STATUSES
                    if args.adaptive
                    else {"completed"}
                )
                if terminal:
                    continue
                work.append((index, episode, attempt))

            if not work:
                if not args.adaptive:
                    break
                added = adaptive_scheduler.plan_next_wave(manifest)
                save_manifest(run_dir, manifest)
                append_event(
                    run_dir,
                    "adaptive_wave_planned",
                    phase=manifest["adaptive_scheduler"]["phase"],
                    episode_keys=added,
                    active_tasks=len(manifest["adaptive_scheduler"]["active_tasks"]),
                    culled_tasks=len(manifest["adaptive_scheduler"]["culled_tasks"]),
                )
                if added:
                    continue
                break

            stop_event = threading.Event()
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency,
                thread_name_prefix="toolathlon-episode",
            )
            active: dict[
                concurrent.futures.Future[dict[str, Any]],
                tuple[int, dict[str, Any], int, int],
            ] = {}
            remaining = iter(work)

            def record_result(
                episode: dict[str, Any],
                result: dict[str, Any],
                *,
                interrupted_run: bool = False,
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
                try:
                    index, episode, attempt = next(remaining)
                except StopIteration:
                    return False
                provider_generation = provider_backoff.wait(stop_event)
                print(
                    f"[{index}/{manifest['counts']['total']}] {episode['key']} "
                    f"attempt {attempt}",
                    flush=True,
                )
                episode.update(
                    status="running",
                    started_at=utc_now(),
                    finished_at=None,
                    error=None,
                )
                save_manifest(run_dir, manifest)
                append_event(
                    run_dir, "episode_started", key=episode["key"], attempt=attempt
                )
                future = executor.submit(
                    execute_episode,
                    args,
                    runner_path=repo_root / "gyms" / "toolathlon_gym" / "run.py",
                    root=root,
                    run_dir=run_dir,
                    episode=episode,
                    attempt=attempt,
                    stop_event=stop_event,
                    subagent_port=args.subagent_ports[next_endpoint],
                    container_slot=(index - 1) % args.container_slots,
                )
                next_endpoint = (next_endpoint + 1) % len(args.subagent_ports)
                active[future] = (index, episode, attempt, provider_generation)
                return True

            def update_provider_backoff(
                result: dict[str, Any], provider_generation: int
            ) -> None:
                marker = openrouter_transient_failure(result)
                if marker is not None:
                    delay = provider_backoff.register_failure(provider_generation)
                    if delay is not None:
                        print(
                            "OpenRouter transient failure; pausing new episodes "
                            f"for {delay:g}s ({marker})",
                            flush=True,
                        )
                        append_event(
                            run_dir,
                            "provider_backoff_started",
                            provider="openrouter",
                            delay_seconds=delay,
                            marker=marker,
                        )
                    return
                if result.get("status") == "completed" and (
                    provider_backoff.register_success(provider_generation)
                ):
                    append_event(
                        run_dir,
                        "provider_backoff_reset",
                        provider="openrouter",
                    )

            try:
                for _ in range(min(args.concurrency, len(work))):
                    submit_next()
                while active:
                    done, _ = concurrent.futures.wait(
                        active,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        _index, episode, attempt, provider_generation = active.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:
                            result = failure(attempt, error, episode["started_at"])
                        except BaseException as error:
                            result = failure(attempt, error, episode["started_at"])
                            record_result(episode, result, interrupted_run=True)
                            raise
                        record_result(episode, result)
                        update_provider_backoff(result, provider_generation)
                        submit_next()
            except BaseException:
                stop_event.set()
                for future, (
                    _index,
                    episode,
                    attempt,
                    _provider_generation,
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
            if not args.adaptive:
                break
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
        save_manifest(run_dir, manifest)
        append_event(
            run_dir,
            "batch_finished",
            status=manifest["status"],
            counts=manifest["counts"],
        )

    print(f"Run manifest: {run_dir / 'manifest.json'}", flush=True)
    return manifest
