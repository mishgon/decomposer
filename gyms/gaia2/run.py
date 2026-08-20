"""Run one prepared Gaia2 execution evaluation locally, without a job."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gyms.gaia2.dataset import (  # noqa: E402
    sha256_file,
    validate_materialized_dataset,
    validate_materialized_filesystem,
)
from gyms.gaia2.experiments import (  # noqa: E402
    DOMAIN,
    GAIA2_REVISION,
    JUDGE_MODEL,
    PROJECT_VENV,
    SCENARIO_COUNT,
    SCENARIO_TIMEOUT_SECONDS,
    SPLIT,
    DecomposerExperiment,
    Experiment,
    SimpleExperiment,
    dataset_revision_root,
    filesystem_dir,
    filesystem_revision_root,
    get_experiment,
    output_dir,
    preparation_manifest,
)
from gyms.gaia2.staging import git  # noqa: E402


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_cuda_visible_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(","))
    if not devices or any(not item for item in devices):
        raise argparse.ArgumentTypeError(
            "CUDA devices must be a comma-separated list without empty entries"
        )
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("CUDA devices must be unique")
    return devices


def selected_cuda_devices(
    experiment: Experiment, requested: tuple[str, ...] | None
) -> tuple[str, ...]:
    devices = requested or tuple(str(index) for index in range(experiment.num_gpus))
    if len(devices) != experiment.num_gpus:
        raise ValueError(
            f"{experiment.name} requires {experiment.num_gpus} CUDA device(s), "
            f"but {len(devices)} were selected"
        )
    return devices


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


def check_judge(endpoint: str, api_key: str) -> None:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    context = ssl._create_unverified_context()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
    )
    try:
        with opener.open(request, timeout=20) as response:
            if response.status >= 500:
                raise RuntimeError(f"Judge endpoint returned HTTP {response.status}")
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise RuntimeError(
            f"Judge endpoint is unavailable: {endpoint}: {error}"
        ) from error


def _common_vllm_command(
    checkpoint: Path,
    served_name: str,
    port: int,
    *,
    thinking: bool,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
) -> list[str]:
    return [
        str(PROJECT_VENV / "bin" / "vllm"),
        "serve",
        str(checkpoint),
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "gemma4",
        "--reasoning-parser",
        "gemma4",
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": thinking}, separators=(",", ":")),
    ]


def simple_vllm_command(experiment: SimpleExperiment) -> list[str]:
    return _common_vllm_command(
        experiment.checkpoint,
        experiment.served_name,
        experiment.port,
        thinking=experiment.thinking,
        max_model_len=experiment.max_model_len,
        max_num_seqs=experiment.max_num_seqs,
        gpu_memory_utilization=experiment.gpu_memory_utilization,
    )


def decomposer_vllm_commands(
    experiment: DecomposerExperiment,
) -> tuple[list[str], list[str]]:
    manager = _common_vllm_command(
        experiment.manager_checkpoint,
        experiment.manager_served_name,
        experiment.manager_port,
        thinking=experiment.manager_thinking,
        max_model_len=experiment.max_model_len,
        max_num_seqs=experiment.max_num_seqs,
        gpu_memory_utilization=experiment.gpu_memory_utilization,
    )
    worker = _common_vllm_command(
        experiment.worker_checkpoint,
        experiment.worker_served_name,
        experiment.worker_port,
        thinking=experiment.worker_thinking,
        max_model_len=experiment.max_model_len,
        max_num_seqs=experiment.max_num_seqs,
        gpu_memory_utilization=experiment.gpu_memory_utilization,
    )
    return manager, worker


def langgraph_command(
    local_repo: Path, experiment: DecomposerExperiment
) -> tuple[list[str], Path]:
    directory = local_repo / "gyms" / "gaia2" / "subagents"
    return (
        [
            str(PROJECT_VENV / "bin" / "langgraph"),
            "dev",
            "--config",
            str(directory / "langgraph.json"),
            "--host",
            "127.0.0.1",
            "--port",
            str(experiment.subagent_port),
            "--n-jobs-per-worker",
            "16",
            "--no-browser",
            "--no-reload",
            "--allow-blocking",
        ],
        directory,
    )


def service_command(
    experiment: DecomposerExperiment,
    config_path: Path,
) -> list[str]:
    return [
        str(PROJECT_VENV / "bin" / "python"),
        "-m",
        "gyms.gaia2.service",
        "--config",
        str(config_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(experiment.service_port),
    ]


def are_command(
    experiment: Experiment,
    *,
    benchmark: Path,
    dataset_root: Path,
    output: Path,
    judge_endpoint: str,
    num_repeats: int,
    limit: int | None,
    plugin_config: Path | None,
) -> list[str]:
    command = [
        str(benchmark),
        "run",
        "-d",
        str(dataset_root),
        "--config",
        DOMAIN,
        "--judge_model",
        JUDGE_MODEL,
        "--judge_provider",
        "local",
        "--judge_endpoint",
        judge_endpoint,
        "--num_runs",
        str(num_repeats),
        "--max_concurrent_scenarios",
        str(experiment.concurrency),
        "--scenario_timeout",
        str(SCENARIO_TIMEOUT_SECONDS),
        "--output_dir",
        str(output),
        "--trace_dump_format",
        "both",
        "--log-level",
        "WARNING",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if isinstance(experiment, SimpleExperiment):
        command.extend(
            [
                "--model",
                f"openai/{experiment.served_name}",
                "--provider",
                "local",
                "--endpoint",
                f"http://127.0.0.1:{experiment.port}/v1",
                "--agent",
                "native_tools",
            ]
        )
    else:
        if plugin_config is None:
            raise ValueError("Decomposer evaluation requires a plugin config")
        command.extend(
            [
                "--model",
                "decomposer",
                "--provider",
                "local",
                "--agent",
                "decomposer",
                "--agent-plugin",
                "gyms.gaia2.plugin:create_plugin",
                "--agent-config",
                str(plugin_config),
                "--executor_type",
                "thread",
            ]
        )
    return command


def _validate_file_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    for filename, item in (manifest.get("files") or {}).items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != item.get("size"):
            raise ValueError(f"Prepared model file changed or is missing: {path}")
        if item.get("sha256") and sha256_file(path) != item["sha256"]:
            raise ValueError(f"Prepared model file checksum changed: {path}")


def validate_preparation(experiment: Experiment) -> dict[str, Any]:
    path = preparation_manifest(experiment)
    if not path.is_file():
        raise FileNotFoundError(
            f"Run `python -m gyms.gaia2.prepare eval --experiment {experiment.name}` first: {path}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != {
        "name": experiment.name,
        "kind": experiment.kind,
    }:
        raise ValueError(f"Preparation manifest does not match {experiment.name}")
    if manifest.get("split") != SPLIT or manifest.get("domain") != DOMAIN:
        raise ValueError("Preparation manifest does not match validation/execution")
    dataset = validate_materialized_dataset(dataset_revision_root())
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
    staged_gaia2 = Path(gaia2.get("staged_repo", ""))
    if (
        not staged_gaia2.is_dir()
        or git(staged_gaia2, "rev-parse", "HEAD") != GAIA2_REVISION
    ):
        raise ValueError("Prepared Gaia2 source tree is missing or changed")
    runtime = gaia2.get("runtime") or {}
    benchmark = Path(runtime.get("are_benchmark", ""))
    if not benchmark.is_file():
        raise FileNotFoundError(f"Prepared are-benchmark is missing: {benchmark}")
    if sha256_file(staged_gaia2 / "uv.lock") != runtime.get("uv_lock_sha256"):
        raise ValueError("Prepared Gaia2 runtime lock checksum changed")
    models = manifest.get("models") or {}
    if isinstance(experiment, SimpleExperiment):
        policy = models.get("policy") or {}
        if Path(policy.get("path", "")) != experiment.checkpoint:
            raise ValueError("Preparation manifest points at an unexpected policy")
        _validate_file_manifest(experiment.checkpoint, policy)
    else:
        manager = models.get("manager") or {}
        worker = models.get("worker") or {}
        if Path(manager.get("path", "")) != experiment.manager_checkpoint:
            raise ValueError("Preparation manifest points at an unexpected manager")
        if Path(worker.get("path", "")) != experiment.worker_checkpoint:
            raise ValueError("Preparation manifest points at an unexpected worker")
        _validate_file_manifest(experiment.manager_checkpoint, manager)
        _validate_file_manifest(experiment.worker_checkpoint, worker)
    return manifest


def archive_attempt(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    entries = [path for path in directory.iterdir() if path.name != "attempts"]
    if not entries:
        return None
    attempt_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = directory / "attempts" / attempt_id
    if archive.exists():
        archive = directory / "attempts" / f"{attempt_id}_{os.getpid()}"
    archive.mkdir(parents=True)
    for path in entries:
        shutil.move(str(path), str(archive / path.name))
    return archive


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
    directory: Path, *, num_repeats: int, limit: int | None
) -> dict[str, Any]:
    rows = _read_jsonl(directory / "output.jsonl")
    task_count = min(SCENARIO_COUNT, limit) if limit is not None else SCENARIO_COUNT
    expected = task_count * num_repeats
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} Gaia2 rollout records, found {len(rows)}"
        )
    scores = [float(row.get("score") or 0.0) for row in rows]
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


def _runtime_configs(
    local_repo: Path,
    directory: Path,
    experiment: DecomposerExperiment,
) -> tuple[Path, Path]:
    config_dir = directory / "configuration"
    service_path = config_dir / "service.json"
    plugin_path = config_dir / "are_plugin.json"
    service = {
        "manager": {
            "model": experiment.manager_served_name,
            "base_url": f"http://127.0.0.1:{experiment.manager_port}/v1",
            "api_key": "EMPTY",
            "temperature": experiment.temperature,
            "top_p": experiment.top_p,
            "max_completion_tokens": experiment.max_completion_tokens,
            "use_responses_api": False,
            "extra_body": {
                "top_k": experiment.top_k,
                "include_reasoning": experiment.manager_thinking,
                "chat_template_kwargs": {
                    "enable_thinking": experiment.manager_thinking
                },
            },
        },
        "subagent_types": [
            {
                "subagent_type_id": "gaia2_worker",
                "description": (
                    "Thinking Gemma-4 E4B worker with authenticated access to "
                    "the current Gaia2 execution scenario tools."
                ),
                "assistant_id": "gaia2_worker",
                "url": f"http://127.0.0.1:{experiment.subagent_port}",
            }
        ],
        "manager_recursion_limit": 200,
        "subagent_recursion_limit": 200,
    }
    plugin = {
        "service_url": f"http://127.0.0.1:{experiment.service_port}",
        "policy": "shared_serialized",
        "request_timeout_seconds": 3500,
        "notification_poll_seconds": 0.1,
        "sidecar_root": str(directory / "decomposer_sidecars"),
        "allow_uncollected_final": False,
        "output_dir": str(directory),
        "gaia2_revision": GAIA2_REVISION,
        "decomposer_revision": git(local_repo, "rev-parse", "HEAD"),
        "service_configuration": service,
        "model_configuration": {
            "manager": {
                "path": str(experiment.manager_checkpoint),
                "served_name": experiment.manager_served_name,
                "thinking": experiment.manager_thinking,
            },
            "subagent": {
                "path": str(experiment.worker_checkpoint),
                "served_name": experiment.worker_served_name,
                "thinking": experiment.worker_thinking,
            },
        },
    }
    atomic_json(service_path, service)
    atomic_json(plugin_path, plugin)
    return service_path, plugin_path


def _base_environment(
    local_repo: Path,
    staged_gaia2: Path,
    directory: Path,
    judge_key: str,
) -> dict[str, str]:
    cache_root = directory / "cache"
    caches = {
        "VLLM_CACHE_ROOT": cache_root / "vllm",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "FLASHINFER_WORKSPACE_BASE": cache_root / "flashinfer",
    }
    for path in caches.values():
        path.mkdir(parents=True, exist_ok=True)
    no_proxy = ",".join(
        value
        for value in (os.environ.get("NO_PROXY", ""), "127.0.0.1", "localhost")
        if value
    )
    inherited_environment = dict(os.environ)
    inherited_environment.pop("VLLM_API_KEY", None)
    return {
        **inherited_environment,
        "PATH": os.pathsep.join(
            (str(PROJECT_VENV / "bin"), os.environ.get("PATH", ""))
        ),
        "PYTHONPATH": os.pathsep.join(
            (
                str(local_repo),
                str(local_repo / "src"),
                str(staged_gaia2),
                os.environ.get("PYTHONPATH", ""),
            )
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HOME": os.environ.get(
            "HF_HOME", "/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface"
        ),
        "DEMO_FS_PATH": str(filesystem_dir()),
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
        "OPENAI_API_KEY": judge_key,
        "ARE_JUDGE_SAMPLING_PARAMS": '{"temperature":0.0}',
        "ARE_LITELLM_SSL_VERIFY": "0",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        **{name: str(path) for name, path in caches.items()},
    }


def _dry_plan(
    local_repo: Path,
    experiment: Experiment,
    directory: Path,
    visible_devices: tuple[str, ...],
    num_repeats: int,
    limit: int | None,
) -> dict[str, Any]:
    manifest_path = preparation_manifest(experiment)
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    benchmark = Path(
        ((manifest or {}).get("gaia2") or {})
        .get("runtime", {})
        .get("are_benchmark", "<prepared-gaia2-venv>/bin/are-benchmark")
    )
    services: list[list[str]] = []
    plugin_config: Path | None = None
    if isinstance(experiment, SimpleExperiment):
        services.append(simple_vllm_command(experiment))
        gpu_assignments = {"policy_vllm": visible_devices[0]}
    else:
        services.extend(decomposer_vllm_commands(experiment))
        service_config = directory / "configuration" / "service.json"
        plugin_config = directory / "configuration" / "are_plugin.json"
        services.append(langgraph_command(local_repo, experiment)[0])
        services.append(service_command(experiment, service_config))
        gpu_assignments = {
            "manager_vllm": visible_devices[0],
            "worker_vllm": visible_devices[1],
        }
    return {
        "experiment": experiment.name,
        "kind": experiment.kind,
        "split": SPLIT,
        "domain": DOMAIN,
        "num_repeats": num_repeats,
        "limit": limit,
        "student_prompt": isinstance(experiment, DecomposerExperiment),
        "gpu_assignments": gpu_assignments,
        "preparation_manifest": str(manifest_path),
        "services": [shlex.join(command) for command in services],
        "are_benchmark": shlex.join(
            are_command(
                experiment,
                benchmark=benchmark,
                dataset_root=dataset_revision_root() / SPLIT,
                output=directory,
                judge_endpoint=os.environ.get("LLM_PROXY_URL", "<LLM_PROXY_URL>"),
                num_repeats=num_repeats,
                limit=limit,
                plugin_config=plugin_config,
            )
        ),
        "output_dir": str(directory),
    }


def execute(local_repo: Path, args: argparse.Namespace) -> int:
    experiment = get_experiment(args.experiment)
    visible_devices = selected_cuda_devices(experiment, args.cuda_visible_devices)
    directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else output_dir(experiment, args.num_repeats, args.limit)
    )
    if args.dry:
        print(
            json.dumps(
                _dry_plan(
                    local_repo,
                    experiment,
                    directory,
                    visible_devices,
                    args.num_repeats,
                    args.limit,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    marker = directory / ".eval_done.json"
    if marker.is_file() and not args.force:
        print(f"Skip (completed): {marker}")
        return 0
    manifest = validate_preparation(experiment)
    judge_endpoint = os.environ.get("LLM_PROXY_URL", "").rstrip("/")
    judge_key = os.environ.get("LLM_PROXY_MASTER_KEY", "")
    if not judge_endpoint or not judge_key:
        raise RuntimeError("LLM_PROXY_URL and LLM_PROXY_MASTER_KEY are required")
    check_judge(judge_endpoint, judge_key)

    archived = archive_attempt(directory) if directory.exists() else None
    directory.mkdir(parents=True, exist_ok=True)
    logs = directory / "logs"
    status_path = directory / "run_status.json"
    started = time.monotonic()
    status: dict[str, Any] = {
        "schema_version": 1,
        "state": "starting",
        "experiment": experiment.name,
        "kind": experiment.kind,
        "split": SPLIT,
        "domain": DOMAIN,
        "num_repeats": args.num_repeats,
        "limit": args.limit,
        "student_prompt": isinstance(experiment, DecomposerExperiment),
        "cuda_visible_devices": list(visible_devices),
        "output_dir": str(directory),
        "preparation_manifest": str(preparation_manifest(experiment)),
        "started_at": utc_now(),
    }
    if archived is not None:
        status["archived_attempt"] = str(archived)
    atomic_json(status_path, status)

    staged_gaia2 = Path(manifest["gaia2"]["staged_repo"])
    benchmark = Path(manifest["gaia2"]["runtime"]["are_benchmark"])
    env = _base_environment(local_repo, staged_gaia2, directory, judge_key)
    supervisor = Supervisor(logs, env)
    plugin_config: Path | None = None
    try:
        status["state"] = "agent_services_startup"
        atomic_json(status_path, status)
        if isinstance(experiment, SimpleExperiment):
            process = supervisor.start(
                "policy_vllm",
                simple_vllm_command(experiment),
                cwd=local_repo,
                env={"CUDA_VISIBLE_DEVICES": visible_devices[0]},
            )
            wait_http(
                f"http://127.0.0.1:{experiment.port}/v1/models",
                [process],
                1800,
            )
            env["ARE_ENABLE_THINKING"] = "1" if experiment.thinking else "0"
            env["ARE_SAMPLING_PARAMS"] = json.dumps(
                {
                    "temperature": experiment.temperature,
                    "top_p": experiment.top_p,
                    "top_k": experiment.top_k,
                    "max_tokens": experiment.max_completion_tokens,
                },
                separators=(",", ":"),
            )
        else:
            manager_command, worker_command = decomposer_vllm_commands(experiment)
            manager_process = supervisor.start(
                "manager_vllm",
                manager_command,
                cwd=local_repo,
                env={"CUDA_VISIBLE_DEVICES": visible_devices[0]},
            )
            worker_process = supervisor.start(
                "worker_vllm",
                worker_command,
                cwd=local_repo,
                env={"CUDA_VISIBLE_DEVICES": visible_devices[1]},
            )
            wait_http(
                f"http://127.0.0.1:{experiment.manager_port}/v1/models",
                [manager_process],
                1800,
            )
            wait_http(
                f"http://127.0.0.1:{experiment.worker_port}/v1/models",
                [worker_process],
                1800,
            )
            service_config, plugin_config = _runtime_configs(
                local_repo, directory, experiment
            )
            subagent_env = {
                "GAIA2_SUBAGENT_MODEL": experiment.worker_served_name,
                "GAIA2_SUBAGENT_ENDPOINT": (
                    f"http://127.0.0.1:{experiment.worker_port}/v1"
                ),
                "GAIA2_SUBAGENT_API_KEY": "EMPTY",
                "GAIA2_SUBAGENT_TEMPERATURE": str(experiment.temperature),
                "GAIA2_SUBAGENT_TOP_P": str(experiment.top_p),
                "GAIA2_SUBAGENT_TOP_K": str(experiment.top_k),
                "GAIA2_SUBAGENT_MAX_COMPLETION_TOKENS": str(
                    experiment.max_completion_tokens
                ),
                "GAIA2_SUBAGENT_THINKING": ("1" if experiment.worker_thinking else "0"),
            }
            langgraph_argv, langgraph_cwd = langgraph_command(local_repo, experiment)
            langgraph_process = supervisor.start(
                "langgraph_subagent",
                langgraph_argv,
                cwd=langgraph_cwd,
                env=subagent_env,
            )
            service_process = supervisor.start(
                "decomposer_service",
                service_command(experiment, service_config),
                cwd=local_repo,
            )
            wait_http(
                f"http://127.0.0.1:{experiment.subagent_port}/ok",
                [langgraph_process],
                300,
            )
            wait_http(
                f"http://127.0.0.1:{experiment.service_port}/health",
                [service_process],
                300,
            )

        status["state"] = "evaluation"
        atomic_json(status_path, status)
        command = are_command(
            experiment,
            benchmark=benchmark,
            dataset_root=dataset_revision_root() / SPLIT,
            output=directory,
            judge_endpoint=judge_endpoint,
            num_repeats=args.num_repeats,
            limit=args.limit,
            plugin_config=plugin_config,
        )
        are_env = dict(env)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            are_env.pop(name, None)
        with (logs / "are_benchmark.log").open("a", encoding="utf-8") as stream:
            subprocess.run(
                command,
                check=True,
                cwd=staged_gaia2,
                env=are_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        supervisor.assert_running()
        status["state"] = "validation"
        atomic_json(status_path, status)
        metrics = validate_result(
            directory,
            num_repeats=args.num_repeats,
            limit=args.limit,
        )
        atomic_json(directory / "metrics.json", metrics)
        gpu_metadata = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except Exception:
        status.update(
            {
                "state": "failed",
                "finished_at": utc_now(),
                "total_seconds": round(time.monotonic() - started, 3),
            }
        )
        atomic_json(status_path, status)
        raise
    finally:
        supervisor.stop()

    status.update(
        {
            "state": "complete",
            "finished_at": utc_now(),
            "total_seconds": round(time.monotonic() - started, 3),
            "metrics": metrics,
            "gpu_metadata": gpu_metadata,
            "decomposer_commit": git(local_repo, "rev-parse", "HEAD"),
            "gaia2_commit": git(staged_gaia2, "rev-parse", "HEAD"),
        }
    )
    atomic_json(status_path, status)
    atomic_json(marker, status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", choices=(SPLIT,), default=SPLIT)
    parser.add_argument("--domain", choices=(DOMAIN,), default=DOMAIN)
    parser.add_argument("--num-repeats", type=positive_int, default=3)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--dry", "--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--cuda-visible-devices",
        type=parse_cuda_visible_devices,
        help="comma-separated physical GPU IDs mapped to logical model slots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the result, logs, status, and marker directory",
    )
    parser.add_argument("--workdir", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.workdir or Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return execute(repo_root.resolve(), args)


if __name__ == "__main__":
    raise SystemExit(main())
