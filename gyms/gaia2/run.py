from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gyms.gaia2.prepare import (
    DEFAULT_GAIA2_REPO,
    DOMAIN,
    DOMAINS,
    GAIA2_REVISION,
    JUDGE_MODEL,
    PROJECT_ROOT,
    RESULTS_ROOT,
    SCENARIO_COUNT,
    SCENARIO_TIMEOUT_SECONDS,
    SPLIT,
    atomic_json,
    dataset_manifest,
    dataset_revision_root,
    dataset_root,
    filesystem_dir,
    filesystem_revision_root,
    get_domain_spec,
    git_revision,
    preparation_manifest,
    sha256_file,
    validate_materialized_dataset,
    validate_materialized_filesystem,
)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


class Supervisor:
    def __init__(self, log_dir: Path, env: dict[str, str]) -> None:
        self.log_dir = log_dir
        self.env = env
        self.processes: list[tuple[str, subprocess.Popen[Any], Any]] = []

    def start(
        self,
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.Popen[Any]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stream = (self.log_dir / f"{name}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env={**self.env, **dict(env or {})},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.processes.append((name, process, stream))
        return process

    def assert_running(self) -> None:
        stopped = [
            f"{name} (rc={process.poll()})"
            for name, process, _ in self.processes
            if process.poll() is not None
        ]
        if stopped:
            raise RuntimeError(
                "Required service stopped during evaluation: " + ", ".join(stopped)
            )

    def stop(self) -> None:
        for _, process, _ in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 20
        for _, process, _ in reversed(self.processes):
            if process.poll() is None:
                try:
                    process.wait(max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            process.wait()
        for _, _, stream in self.processes:
            stream.close()


def wait_http(
    url: str,
    processes: Sequence[subprocess.Popen[Any]],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in processes:
            code = process.poll()
            if code is not None:
                raise RuntimeError(
                    f"Server process {process.pid} exited with code {code}"
                )
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status < 500:
                    return
        except (OSError, TimeoutError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: expected an object")
        values.append(value)
    return values


def validate_result(
    directory: Path,
    *,
    num_repeats: int,
    limit: int | None,
    scenario_count: int = SCENARIO_COUNT,
    scenario_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(directory / "output.jsonl")
    task_count = min(scenario_count, limit) if limit is not None else scenario_count
    expected = task_count * num_repeats
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} Gaia2 rollout records, found {len(rows)}"
        )
    scores = [float(row.get("score") or 0.0) for row in rows]
    if any(score not in (0.0, 1.0) for score in scores):
        raise ValueError("GAIA2 evaluation scores must be binary")
    if scenario_ids is not None:
        selected_ids = tuple(scenario_ids[:task_count])
        expected_counts = Counter(
            {scenario_id: num_repeats for scenario_id in selected_ids}
        )
        observed_counts = Counter(str(row.get("task_id")) for row in rows)
        if observed_counts != expected_counts:
            raise ValueError("GAIA2 rollout coverage does not match the selected tasks")
        expected_runs = set(range(1, num_repeats + 1))
        observed_runs: dict[str, set[int]] = {
            scenario_id: set() for scenario_id in selected_ids
        }
        for row in rows:
            task_id = str(row.get("task_id"))
            metadata = row.get("metadata") or {}
            run_number = metadata.get("run_number")
            if run_number is None and num_repeats == 1:
                # ARE records no run number when it runs each scenario once.
                run_number = 1
            if isinstance(run_number, bool) or not isinstance(run_number, int):
                raise ValueError("GAIA2 rollout run_number must be an integer")
            if run_number in observed_runs[task_id]:
                raise ValueError(f"Duplicate GAIA2 rollout ({task_id}, {run_number})")
            observed_runs[task_id].add(run_number)
        if any(run_numbers != expected_runs for run_numbers in observed_runs.values()):
            raise ValueError("GAIA2 logical rollout numbers are incomplete")
    exceptions = Counter(
        str((row.get("metadata") or {}).get("exception_type") or "unknown")
        for row in rows
        if (row.get("metadata") or {}).get("has_exception")
    )
    lite_files = sorted((directory / "lite").glob("*.json"))
    judge_rationales = 0
    decisions = Counter()
    for path in lite_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        decisions[str(value.get("validation_decision") or "missing")] += 1
        if value.get("validation_rationale") is not None:
            judge_rationales += 1
    passed = sum(1 for score in scores if score > 0)
    return {
        "task_rows": task_count,
        "rollout_rows": len(rows),
        "expected_rollout_rows": expected,
        "passed_rollouts": passed,
        "failed_rollouts": expected - passed,
        "fixed_denominator_score": sum(scores) / expected,
        "score_sum": sum(scores),
        "lite_traces": len(lite_files),
        "judge_rationales": judge_rationales,
        "validation_decisions": dict(sorted(decisions.items())),
        "exception_types": dict(sorted(exceptions.items())),
        "output_jsonl": str(directory / "output.jsonl"),
        "lite_dir": str(directory / "lite"),
        "hf_dir": str(directory / "hf"),
    }


def validate_preparation(
    *,
    domain: str = DOMAIN,
) -> dict[str, Any]:
    spec = get_domain_spec(domain)
    path = preparation_manifest(spec.name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Run `python -m gyms.gaia2.prepare eval --domain {spec.name}` first: {path}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split") != SPLIT or manifest.get("domain") != spec.name:
        raise ValueError(f"Preparation manifest does not match validation/{spec.name}")
    dataset = validate_materialized_dataset(
        dataset_revision_root(spec.name), domain=spec.name
    )
    if dataset.get("aggregate_sha256") != spec.dataset_aggregate_sha256:
        raise ValueError(
            f"Prepared Gaia2 {spec.name} checksum does not match the pinned "
            "dataset revision"
        )
    if (
        manifest.get("dataset", {}).get("aggregate_sha256")
        != dataset["aggregate_sha256"]
    ):
        raise ValueError("Preparation manifest dataset checksum changed")
    filesystem = validate_materialized_filesystem(filesystem_revision_root())
    prepared_filesystem = manifest.get("filesystem") or {}
    if prepared_filesystem.get("aggregate_sha256") != filesystem["aggregate_sha256"]:
        raise ValueError("Preparation manifest filesystem checksum changed")
    gaia2 = manifest.get("gaia2") or {}
    if gaia2.get("commit") != GAIA2_REVISION:
        raise ValueError("Preparation manifest points at an unexpected Gaia2 revision")
    gaia2_source = DEFAULT_GAIA2_REPO
    if not gaia2_source.is_dir() or git_revision(gaia2_source) != GAIA2_REVISION:
        raise ValueError("Prepared Gaia2 source tree is missing or changed")
    runtime = gaia2.get("runtime") or {}
    benchmark = Path(runtime.get("are_benchmark", ""))
    if not benchmark.is_file():
        raise FileNotFoundError(f"Prepared are-benchmark is missing: {benchmark}")
    if sha256_file(gaia2_source / "uv.lock") != runtime.get("uv_lock_sha256"):
        raise ValueError("Prepared Gaia2 runtime lock checksum changed")
    return manifest


def selected_scenario_ids(args: argparse.Namespace) -> tuple[str, ...]:
    path = dataset_manifest(args.domain)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset is missing: {path}; run gyms.gaia2.prepare eval --domain {args.domain}"
        )
    manifest = json.loads(path.read_text())
    available = tuple(sorted(item["scenario_id"] for item in manifest["scenarios"]))
    if args.task:
        unknown = set(args.task) - set(available)
        if unknown:
            raise ValueError(f"Unknown tasks in the selected domain: {sorted(unknown)}")
        available = tuple(task for task in available if task in args.task)
    return tuple(available[: args.limit] if args.limit else available)


def are_command(
    args: argparse.Namespace,
    *,
    benchmark: Path,
    dataset: Path,
    directory: Path,
    judge_endpoint: str,
) -> list[str]:
    command = [
        str(benchmark),
        "run",
        "-d",
        str(dataset),
        "--config",
        args.domain,
        "--judge_model",
        args.judge_model,
        "--judge_provider",
        "local",
        "--judge_endpoint",
        judge_endpoint,
        "--num_runs",
        str(args.num_repeats),
        "--max_concurrent_scenarios",
        str(args.concurrency),
        "--scenario_timeout",
        str(args.episode_timeout),
        "--output_dir",
        str(directory),
        "--trace_dump_format",
        "both",
        "--log-level",
        "WARNING",
        "--executor_type",
        "thread",
    ]
    command += [
        "--model",
        args.agent,
        "--provider",
        "local",
        "--agent",
        "agent_server",
        "--agent-plugin",
        "gyms.gaia2.are:create_plugin",
        "--agent-config",
        str(directory / "configuration" / "are_plugin.json"),
    ]
    return command


def execute(args: argparse.Namespace) -> Path:
    if not 0 <= args.port_offset <= 65535 - 8123:
        raise ValueError("port-offset places service ports outside the TCP port range")
    scenario_ids = selected_scenario_ids(args)
    directory = args.output_dir or RESULTS_ROOT / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    directory = directory.expanduser().resolve()
    identity = {
        "agent": args.agent,
        "decomposer_model_id": args.decomposer_model_id,
        "decomposer_base_url": args.decomposer_base_url,
        "agent_base_url": args.agent_base_url,
        "agent_model_id": args.agent_model_id,
        "domain": args.domain,
        "num_repeats": args.num_repeats,
        "scenario_ids": scenario_ids,
        "concurrency": args.concurrency,
        "episode_timeout": args.episode_timeout,
        "agent_run_budget_seconds": args.agent_run_budget_seconds,
        "agent_shutdown_grace_seconds": args.agent_shutdown_grace_seconds,
        "judge_model": args.judge_model,
        "gaia2_revision": GAIA2_REVISION,
        "decomposer_revision": git_revision(PROJECT_ROOT),
    }
    if args.dry:
        print(json.dumps({"output_dir": str(directory), **identity}, indent=2))
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    manifest = validate_preparation(domain=args.domain)
    judge_endpoint = os.environ.get("LLM_PROXY_URL", "")
    judge_key = os.environ.get("LLM_PROXY_MASTER_KEY", "")
    if not judge_endpoint or not judge_key:
        raise ValueError(
            "LLM_PROXY_URL and LLM_PROXY_MASTER_KEY are required for the ARE judge"
        )
    gaia2_source = DEFAULT_GAIA2_REPO
    directory.mkdir(parents=True, exist_ok=True)
    selected_dataset = directory / "inputs"
    selected_domain = selected_dataset / args.domain
    selected_domain.mkdir(parents=True)
    source = dataset_root(args.domain) / args.domain
    for scenario_id in scenario_ids:
        shutil.copyfile(
            source / f"{scenario_id}.json", selected_domain / f"{scenario_id}.json"
        )
    no_proxy = ",".join(
        filter(None, (os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"))
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(gaia2_source))
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "DEMO_FS_PATH": str(filesystem_dir()),
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
        "OPENAI_API_KEY": judge_key,
        "ARE_JUDGE_SAMPLING_PARAMS": '{"temperature":0.0}',
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "GAIA2_DECOMPOSER_MODEL_ID": args.decomposer_model_id,
        "GAIA2_DECOMPOSER_BASE_URL": args.decomposer_base_url or "",
        "GAIA2_AGENT_SERVER_URL": f"http://127.0.0.1:{8123 + args.port_offset}",
        "GAIA2_AGENT_RUN_BUDGET_SECONDS": str(args.agent_run_budget_seconds),
        "GAIA2_AGENT_SHUTDOWN_GRACE_SECONDS": str(args.agent_shutdown_grace_seconds),
        "GAIA2_AGENT_MODEL_ID": args.agent_model_id,
        "GAIA2_AGENT_BASE_URL": args.agent_base_url or "",
    }
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(name, None)
    supervisor = Supervisor(directory / "logs", env)
    status = {
        **identity,
        "preparation_manifest": str(preparation_manifest(args.domain)),
        "state": "starting",
        "started_at": datetime.now(UTC).isoformat(),
    }
    started = time.monotonic()
    atomic_json(directory / "run_status.json", status)
    previous_sigterm = signal.signal(
        signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum)
    )
    try:
        config_dir = directory / "configuration"
        atomic_json(
            config_dir / "are_plugin.json",
            {
                "assistant_id": args.agent,
                "agent_server_url": env["GAIA2_AGENT_SERVER_URL"],
                "request_timeout_seconds": args.episode_timeout + 60,
                "sidecar_root": str(directory / "decomposer_sidecars"),
                "gaia2_revision": GAIA2_REVISION,
                "decomposer_revision": identity["decomposer_revision"],
                "model_configuration": {
                    "decomposer_model_id": args.decomposer_model_id,
                    "agent_model_id": args.agent_model_id,
                    "decomposer_base_url": args.decomposer_base_url,
                    "agent_base_url": args.agent_base_url,
                },
            },
        )
        runtime_dir = directory / "cache" / "langgraph_runtime"
        runtime_dir.mkdir(parents=True)
        worker = supervisor.start(
            "agent_server",
            [
                sys.executable,
                "-m",
                "gyms.gaia2.agents",
                "--port",
                str(8123 + args.port_offset),
                "--n-jobs-per-worker",
                str(max(16, 2 * args.concurrency)),
            ],
            cwd=runtime_dir,
        )
        wait_http(f"http://127.0.0.1:{8123 + args.port_offset}/ok", [worker], 300)
        command = are_command(
            args,
            benchmark=Path(manifest["gaia2"]["runtime"]["are_benchmark"]),
            dataset=selected_dataset,
            directory=directory,
            judge_endpoint=judge_endpoint,
        )
        status["state"] = "evaluation"
        atomic_json(directory / "run_status.json", status)
        process = supervisor.start("are_benchmark", command, cwd=gaia2_source)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        metrics = validate_result(
            directory,
            num_repeats=args.num_repeats,
            limit=None,
            scenario_count=len(scenario_ids),
            scenario_ids=scenario_ids,
        )
        atomic_json(directory / "metrics.json", metrics)
        status.update(state="complete", metrics=metrics)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        supervisor.stop()
        status.update(
            finished_at=datetime.now(UTC).isoformat(),
            total_seconds=round(time.monotonic() - started, 3),
        )
        atomic_json(directory / "run_status.json", status)
    atomic_json(directory / ".eval_done.json", status)
    print(json.dumps({"output_dir": str(directory), "metrics": metrics}, indent=2))
    return directory


def build_parser(*, evaluation: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate existing model endpoints on GAIA2."
    )
    parser.add_argument(
        "--agent", choices=("decomposer", "qwen_3_5_4b_thinking"), default="decomposer"
    )
    parser.add_argument(
        "--decomposer-model-id", default="vllm/qwen_3_8_flash_next_non_thinking"
    )
    parser.add_argument("--agent-model-id", default="vllm/qwen_3_5_4b_thinking")
    parser.add_argument("--decomposer-base-url")
    parser.add_argument("--agent-base-url")
    if evaluation:
        parser.add_argument(
            "--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS)
        )
    else:
        parser.add_argument("--domain", choices=DOMAINS, default=DOMAIN)
    parser.add_argument("--task", action="append")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--num-repeats", type=positive_int, default=3)
    parser.add_argument("--concurrency", type=positive_int, default=8)
    parser.add_argument(
        "--episode-timeout", type=positive_int, default=SCENARIO_TIMEOUT_SECONDS
    )
    parser.add_argument("--agent-run-budget-seconds", type=positive_int, default=60)
    parser.add_argument("--agent-shutdown-grace-seconds", type=positive_int, default=60)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--port-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    execute(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
