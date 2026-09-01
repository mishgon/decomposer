"""Run one prepared Gaia2 evaluation locally, without a job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
from dataclasses import replace
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
    DOMAINS,
    GAIA2_REVISION,
    JUDGE_MODEL,
    PARTITIONS,
    PROJECT_VENV,
    SCENARIO_COUNT,
    SCENARIO_TIMEOUT_SECONDS,
    SPLIT,
    Gaia2Domain,
    DecomposerExperiment,
    Experiment,
    SimpleExperiment,
    dataset_revision_root,
    filesystem_dir,
    filesystem_revision_root,
    get_domain_spec,
    get_experiment,
    output_dir,
    partition_dataset_root,
    preparation_manifest,
    trace_output_dir,
)
from gyms.gaia2.partition import (  # noqa: E402
    partition_scenario_ids,
    validate_partition_view,
)
from gyms.gaia2.prompts import compose_decomposer_system_prompt  # noqa: E402
from gyms.gaia2.staging import git  # noqa: E402
from decomposer.prompts import (  # noqa: E402
    DECOMPOSER_PROMPT_PROFILES,
)


def select_prompt_profile(
    experiment: Experiment, requested_profile: str | None
) -> Experiment:
    if requested_profile is None:
        return experiment
    if not isinstance(experiment, DecomposerExperiment):
        raise ValueError("--prompt-profile is only valid for Decomposer experiments")
    return replace(experiment, prompt_profile=requested_profile)


def prompt_sha256(experiment: Experiment) -> str | None:
    if not isinstance(experiment, DecomposerExperiment):
        return None
    prompt = compose_decomposer_system_prompt(
        experiment.prompt_profile,
        experiment.manager_prompt_addendum_profile,
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def runtime_configuration(experiment: Experiment) -> dict[str, Any]:
    sampling = {
        name: getattr(experiment, name)
        for name in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
        )
        if getattr(experiment, name) is not None
    }
    configuration: dict[str, Any] = {
        "max_model_len": experiment.max_model_len,
        "max_completion_tokens": experiment.max_completion_tokens,
        "sampling": sampling,
    }
    if isinstance(experiment, SimpleExperiment):
        configuration.update(
            {
                "thinking": experiment.thinking,
                "language_model_only": experiment.language_model_only,
                "max_model_calls": experiment.max_model_calls,
            }
        )
    else:
        configuration.update(
            {
                "manager": {
                    "backend": experiment.manager_backend,
                    "thinking": experiment.manager_thinking,
                    "reasoning_mode": experiment.manager_reasoning_mode,
                    "max_model_calls": experiment.manager_max_model_calls,
                    "recursion_limit": experiment.manager_recursion_limit,
                },
                "subagent": {
                    "thinking": experiment.worker_thinking,
                    "language_model_only": experiment.worker_language_model_only,
                    "max_model_calls": experiment.subagent_max_model_calls,
                    "recursion_limit": experiment.subagent_recursion_limit,
                },
            }
        )
    return configuration


def run_identity(
    experiment: Experiment,
    *,
    domain: Gaia2Domain,
    purpose: str,
    partition: str,
    num_repeats: int,
    concurrency: int,
    limit: int | None,
    rollout_offset: int = 0,
) -> dict[str, Any]:
    """Return the fields that make an output directory safe to reuse."""

    spec = get_domain_spec(domain)
    identity: dict[str, Any] = {
        "purpose": purpose,
        "experiment": experiment.name,
        "kind": experiment.kind,
        "split": SPLIT,
        "domain": spec.name,
        "partition": partition,
        "num_repeats": num_repeats,
        "concurrency": concurrency,
        "limit": limit,
        "decomposer_system_prompt_profile": (
            experiment.prompt_profile
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_system_prompt_addendum_profile": (
            experiment.manager_prompt_addendum_profile
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_system_prompt_sha256": prompt_sha256(experiment),
        "runtime_configuration": runtime_configuration(experiment),
    }
    if partition != "full":
        identity["split_manifest"] = {
            "name": spec.split_manifest_name,
            "path": spec.split_manifest_relpath,
            "sha256": spec.split_manifest_sha256,
        }
    if purpose == "trace-generation":
        identity["rollout_offset"] = rollout_offset
    return identity


def validate_run_identity(
    path: Path,
    expected: Mapping[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate a marker/status before skipping or resuming its artifacts."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Gaia2 run identity file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Gaia2 run identity is not an object: {path}")
    legacy_optional_fields = {
        "decomposer_system_prompt_sha256",
        "runtime_configuration",
    }
    mismatches = {
        key: {"found": value.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
        and not (key in legacy_optional_fields and key not in value)
    }
    if require_complete and value.get("state") != "complete":
        mismatches["state"] = {
            "found": value.get("state"),
            "expected": "complete",
        }
    if mismatches:
        raise ValueError(
            f"Gaia2 output identity mismatch in {path}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    return value


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
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


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
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
    tool_call_parser: str = "gemma4",
    reasoning_parser: str | None = "gemma4",
    language_model_only: bool = True,
    trust_remote_code: bool = False,
    gdn_prefill_backend: str | None = None,
) -> list[str]:
    command = [
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
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_call_parser,
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": thinking}, separators=(",", ":")),
    ]
    if reasoning_parser is not None:
        command.extend(["--reasoning-parser", reasoning_parser])
    if language_model_only:
        command.append("--language-model-only")
    if trust_remote_code:
        command.append("--trust-remote-code")
    if gdn_prefill_backend is not None:
        command.extend(["--gdn-prefill-backend", gdn_prefill_backend])
    return command


def simple_vllm_command(experiment: SimpleExperiment) -> list[str]:
    if experiment.requires_openrouter or experiment.checkpoint is None:
        raise ValueError("Remote simple agents do not start a local vLLM server")
    return _common_vllm_command(
        experiment.checkpoint,
        experiment.served_name,
        experiment.port,
        thinking=experiment.thinking,
        max_model_len=experiment.max_model_len,
        max_num_seqs=experiment.max_num_seqs,
        gpu_memory_utilization=experiment.gpu_memory_utilization,
        tool_call_parser=experiment.tool_call_parser,
        reasoning_parser=experiment.reasoning_parser,
        language_model_only=experiment.language_model_only,
        trust_remote_code=experiment.trust_remote_code,
        gdn_prefill_backend=experiment.gdn_prefill_backend,
    )


def simple_sampling_parameters(
    experiment: SimpleExperiment,
) -> dict[str, int | float]:
    parameters: dict[str, int | float] = {
        "temperature": experiment.temperature,
        "top_p": experiment.top_p,
        "max_tokens": experiment.max_completion_tokens,
    }
    for name in ("top_k", "min_p", "presence_penalty", "repetition_penalty"):
        value = getattr(experiment, name)
        if value is not None:
            parameters[name] = value
    return parameters


def openrouter_proxy_command(experiment: SimpleExperiment) -> list[str]:
    if (
        not experiment.requires_openrouter
        or experiment.base_url is None
        or experiment.api_key_env is None
    ):
        raise ValueError("OpenRouter proxy requires a remote simple experiment")
    return [
        str(PROJECT_VENV / "bin" / "python"),
        "-m",
        "gyms.gaia2.openrouter_proxy",
        "--host",
        "127.0.0.1",
        "--port",
        str(experiment.port),
        "--upstream-url",
        experiment.base_url,
        "--api-key-env",
        experiment.api_key_env,
        "--extra-body-json",
        json.dumps(experiment.remote_extra_body, separators=(",", ":")),
        "--timeout-seconds",
        "3300",
        "--max-retries",
        "2",
    ]


def remote_manager_proxy_command(experiment: DecomposerExperiment) -> list[str]:
    if not experiment.requires_llm_proxy:
        raise ValueError(f"{experiment.name} does not use the LLM proxy manager")
    required = {
        "manager_upstream_url_env": experiment.manager_upstream_url_env,
        "manager_api_key_env": experiment.manager_api_key_env,
        "manager_response_tool_parser": experiment.manager_response_tool_parser,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"{experiment.name}: missing remote manager fields: {missing}")
    command = [
        str(PROJECT_VENV / "bin" / "python"),
        "-m",
        "gyms.remote_model_proxy",
        "--host",
        "127.0.0.1",
        "--port",
        str(experiment.manager_port),
        "--upstream-url-env",
        str(experiment.manager_upstream_url_env),
        "--api-key-env",
        str(experiment.manager_api_key_env),
        "--response-tool-parser",
        str(experiment.manager_response_tool_parser),
        "--timeout-seconds",
        "3300",
        "--max-retries",
        "2",
        "--extra-body-json",
        json.dumps(experiment.remote_manager_extra_body, separators=(",", ":")),
    ]
    if not experiment.manager_verify_tls:
        command.append("--no-verify-tls")
    return command


def decomposer_vllm_commands(
    experiment: DecomposerExperiment,
) -> tuple[list[str] | None, list[str]]:
    manager = None
    if experiment.requires_local_manager:
        if experiment.manager_checkpoint is None:
            raise ValueError("Local manager requires manager_checkpoint")
        manager = _common_vllm_command(
            experiment.manager_checkpoint,
            experiment.manager_served_name,
            experiment.manager_port,
            thinking=experiment.manager_thinking,
            max_model_len=experiment.max_model_len,
            max_num_seqs=experiment.max_num_seqs,
            gpu_memory_utilization=experiment.gpu_memory_utilization,
            tool_call_parser=experiment.manager_tool_call_parser,
            reasoning_parser=experiment.manager_reasoning_parser,
            language_model_only=experiment.manager_language_model_only,
            trust_remote_code=experiment.manager_trust_remote_code,
            gdn_prefill_backend=experiment.manager_gdn_prefill_backend,
        )
    worker = _common_vllm_command(
        experiment.worker_checkpoint,
        experiment.worker_served_name,
        experiment.worker_port,
        thinking=experiment.worker_thinking,
        max_model_len=experiment.max_model_len,
        max_num_seqs=experiment.max_num_seqs,
        gpu_memory_utilization=experiment.gpu_memory_utilization,
        tool_call_parser=experiment.worker_tool_call_parser,
        reasoning_parser=experiment.worker_reasoning_parser,
        language_model_only=experiment.worker_language_model_only,
        trust_remote_code=experiment.worker_trust_remote_code,
        gdn_prefill_backend=experiment.worker_gdn_prefill_backend,
    )
    return manager, worker


def langgraph_runtime_paths(directory: Path) -> tuple[Path, Path]:
    return (
        directory / "configuration" / "langgraph.json",
        directory / "cache" / "langgraph_runtime",
    )


def langgraph_command(
    experiment: DecomposerExperiment, directory: Path
) -> tuple[list[str], Path]:
    config_path, runtime_directory = langgraph_runtime_paths(directory)
    return (
        [
            str(PROJECT_VENV / "bin" / "python"),
            "-m",
            "gyms.gaia2.langgraph_server",
            "--config",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(experiment.subagent_port),
            "--n-jobs-per-worker",
            "16",
        ],
        runtime_directory,
    )


def subagent_environment(experiment: DecomposerExperiment) -> dict[str, str]:
    environment = {
        "GAIA2_SUBAGENT_MODEL": experiment.worker_served_name,
        "GAIA2_SUBAGENT_ENDPOINT": (f"http://127.0.0.1:{experiment.worker_port}/v1"),
        "GAIA2_SUBAGENT_API_KEY": "EMPTY",
        "GAIA2_SUBAGENT_TEMPERATURE": str(experiment.temperature),
        "GAIA2_SUBAGENT_TOP_P": str(experiment.top_p),
        "GAIA2_SUBAGENT_TOP_K": str(experiment.top_k),
        "GAIA2_SUBAGENT_MAX_COMPLETION_TOKENS": str(experiment.max_completion_tokens),
        "GAIA2_SUBAGENT_THINKING": "1" if experiment.worker_thinking else "0",
    }
    if experiment.min_p is not None:
        environment["GAIA2_SUBAGENT_MIN_P"] = str(experiment.min_p)
    if experiment.presence_penalty is not None:
        environment["GAIA2_SUBAGENT_PRESENCE_PENALTY"] = str(
            experiment.presence_penalty
        )
    if experiment.repetition_penalty is not None:
        environment["GAIA2_SUBAGENT_REPETITION_PENALTY"] = str(
            experiment.repetition_penalty
        )
    if experiment.subagent_max_model_calls is not None:
        environment["GAIA2_SUBAGENT_MAX_MODEL_CALLS"] = str(
            experiment.subagent_max_model_calls
        )
    return environment


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
    concurrency: int | None = None,
    domain: Gaia2Domain = DOMAIN,
) -> list[str]:
    command = [
        str(benchmark),
        "run",
        "-d",
        str(dataset_root),
        "--config",
        domain,
        "--judge_model",
        JUDGE_MODEL,
        "--judge_provider",
        "local",
        "--judge_endpoint",
        judge_endpoint,
        "--num_runs",
        str(num_repeats),
        "--max_concurrent_scenarios",
        str(concurrency if concurrency is not None else experiment.concurrency),
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
        endpoint = f"http://127.0.0.1:{experiment.port}/v1"
        command.extend(
            [
                "--model",
                f"openai/{experiment.served_name}",
                "--provider",
                "local",
                "--endpoint",
                endpoint,
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


def validate_preparation(
    experiment: Experiment,
    *,
    partition: str = "full",
    domain: Gaia2Domain = DOMAIN,
) -> dict[str, Any]:
    spec = get_domain_spec(domain)
    path = preparation_manifest(experiment, spec.name)
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
    if manifest.get("split") != SPLIT or manifest.get("domain") != spec.name:
        raise ValueError(
            f"Preparation manifest does not match validation/{spec.name}"
        )
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
    if partition != "full":
        prepared_split = manifest.get("partition_split") or {}
        if prepared_split.get("name") != spec.split_manifest_name:
            raise ValueError("Preparation manifest does not identify the pinned split")
        if prepared_split.get("path") != spec.split_manifest_relpath:
            raise ValueError("Preparation manifest points at an unexpected split")
        if prepared_split.get("sha256") != spec.split_manifest_sha256:
            raise ValueError("Preparation manifest split checksum changed")
        validate_partition_view(partition, domain=spec.name)
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
        if experiment.requires_openrouter:
            if policy != {
                "backend": experiment.backend,
                "model": experiment.served_name,
            }:
                raise ValueError("Preparation manifest points at an unexpected policy")
        else:
            if experiment.checkpoint is None:
                raise ValueError("Local simple agent requires checkpoint")
            if Path(policy.get("path", "")) != experiment.checkpoint:
                raise ValueError("Preparation manifest points at an unexpected policy")
            _validate_file_manifest(experiment.checkpoint, policy)
    else:
        manager = models.get("manager") or {}
        worker = models.get("worker") or {}
        if Path(worker.get("path", "")) != experiment.worker_checkpoint:
            raise ValueError("Preparation manifest points at an unexpected worker")
        _validate_file_manifest(experiment.worker_checkpoint, worker)
        if experiment.requires_local_manager:
            if experiment.manager_checkpoint is None:
                raise ValueError("Local manager requires manager_checkpoint")
            if Path(manager.get("path", "")) != experiment.manager_checkpoint:
                raise ValueError("Preparation manifest points at an unexpected manager")
            _validate_file_manifest(experiment.manager_checkpoint, manager)
        elif manager != {
            "backend": experiment.manager_backend,
            "model": experiment.manager_served_name,
            **(
                {
                    "upstream_url_env": experiment.manager_upstream_url_env,
                    "api_key_env": experiment.manager_api_key_env,
                    "response_tool_parser": experiment.manager_response_tool_parser,
                    "reasoning_mode": experiment.manager_reasoning_mode,
                    "verify_tls": experiment.manager_verify_tls,
                }
                if experiment.requires_llm_proxy
                else {}
            ),
        }:
            raise ValueError("Preparation manifest points at an unexpected manager")
    return manifest


def archive_attempt(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    entries = [
        path
        for path in directory.iterdir()
        if path.name != "attempts"
        and not (path.is_dir() and re.fullmatch(r"smoke_[1-9][0-9]*", path.name))
    ]
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


def _optional_round_artifact(
    path: Path, *, round_directory: Path, trace_directory: Path
) -> str | None:
    if not path.is_file():
        return None
    resolved = path.resolve()
    try:
        resolved.relative_to(round_directory.resolve())
    except ValueError as error:
        raise ValueError(f"Round artifact escapes its raw directory: {path}") from error
    return resolved.relative_to(trace_directory.resolve()).as_posix()


def validate_trace_round(
    round_directory: Path,
    *,
    trace_directory: Path,
    logical_rollout_number: int,
    scenario_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _read_jsonl(round_directory / "output.jsonl")
    expected_ids = set(scenario_ids)
    if len(expected_ids) != len(scenario_ids):
        raise ValueError("Expected Gaia2 trace scenario IDs must be unique")
    metrics = validate_result(
        round_directory,
        num_repeats=1,
        limit=None,
        scenario_count=len(scenario_ids),
    )
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, 1):
        metadata = row.get("metadata") or {}
        scenario_id = metadata.get("scenario_id") or row.get("task_id")
        if scenario_id not in expected_ids:
            raise ValueError(
                f"Unexpected Gaia2 scenario in round {logical_rollout_number}: "
                f"{scenario_id!r}"
            )
        if scenario_id in seen:
            raise ValueError(
                f"Duplicate Gaia2 scenario in round {logical_rollout_number}: "
                f"{scenario_id}"
            )
        raw_native_run_number = metadata.get("run_number")
        if raw_native_run_number is None:
            native_run_number = 0
        elif (
            isinstance(raw_native_run_number, bool)
            or not isinstance(raw_native_run_number, int)
            or raw_native_run_number not in (0, 1)
        ):
            raise ValueError(
                f"Round {logical_rollout_number} has unexpected native "
                f"run_number={raw_native_run_number!r}"
            )
        else:
            native_run_number = raw_native_run_number
        seen.add(scenario_id)

        trace_id = row.get("trace_id")
        hf_path: Path | None = None
        if isinstance(trace_id, str) and trace_id:
            candidate = Path(trace_id)
            if not candidate.is_absolute():
                candidate = round_directory / candidate
            hf_path = candidate
        hf_trace = (
            _optional_round_artifact(
                hf_path,
                round_directory=round_directory,
                trace_directory=trace_directory,
            )
            if hf_path is not None
            else None
        )
        lite_path = (
            round_directory / "lite" / hf_path.name
            if hf_path is not None
            else round_directory / "lite" / "__missing__"
        )
        sidecar_path = (
            round_directory
            / "decomposer_sidecars"
            / f"{scenario_id}__run{native_run_number}.json"
        )
        records.append(
            {
                "schema_version": 1,
                "scenario_id": scenario_id,
                "logical_rollout_number": logical_rollout_number,
                "native_run_number": native_run_number,
                "reward": float(row.get("score") or 0.0),
                "status": str(metadata.get("status") or "unknown"),
                "has_exception": bool(metadata.get("has_exception")),
                "exception_type": metadata.get("exception_type"),
                "round": f"round_{logical_rollout_number:02d}",
                "output_jsonl": (round_directory / "output.jsonl")
                .resolve()
                .relative_to(trace_directory.resolve())
                .as_posix(),
                "output_line_number": line_number,
                "sidecar": _optional_round_artifact(
                    sidecar_path,
                    round_directory=round_directory,
                    trace_directory=trace_directory,
                ),
                "hf_trace": hf_trace,
                "lite_trace": _optional_round_artifact(
                    lite_path,
                    round_directory=round_directory,
                    trace_directory=trace_directory,
                ),
            }
        )
    missing = expected_ids - seen
    if missing:
        raise ValueError(
            f"Round {logical_rollout_number} is missing {len(missing)} scenarios"
        )
    return metrics, sorted(records, key=lambda item: item["scenario_id"])


def complete_trace_round(
    round_directory: Path,
    *,
    logical_rollout_number: int,
    scenario_count: int,
    concurrency: int,
    metrics: Mapping[str, Any],
    archived_round: Path | None = None,
    recovered_from_unmarked_artifacts: bool = False,
) -> None:
    atomic_json(round_directory / "metrics.json", metrics)
    round_status: dict[str, Any] = {
        "schema_version": 1,
        "state": "complete",
        "logical_rollout_number": logical_rollout_number,
        "native_num_runs": 1,
        "scenario_count": scenario_count,
        "attempted_rollouts": metrics["rollout_rows"],
        "concurrency": concurrency,
        "finished_at": utc_now(),
        "metrics": metrics,
    }
    if archived_round is not None:
        round_status["archived_attempt"] = str(archived_round)
    if recovered_from_unmarked_artifacts:
        round_status["recovered_from_unmarked_artifacts"] = True
    atomic_json(round_directory / ".round_done.json", round_status)


def aggregate_trace_manifest(
    trace_directory: Path,
    *,
    logical_rollout_numbers: Sequence[int],
    scenario_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    round_metrics: dict[str, Any] = {}
    for logical_rollout_number in logical_rollout_numbers:
        round_directory = trace_directory / f"round_{logical_rollout_number:02d}"
        marker = round_directory / ".round_done.json"
        if not marker.is_file():
            raise FileNotFoundError(
                f"Completed Gaia2 round marker is missing: {marker}"
            )
        marker_value = json.loads(marker.read_text(encoding="utf-8"))
        if marker_value.get("logical_rollout_number") != logical_rollout_number:
            raise ValueError(f"Gaia2 round marker changed: {marker}")
        metrics, round_records = validate_trace_round(
            round_directory,
            trace_directory=trace_directory,
            logical_rollout_number=logical_rollout_number,
            scenario_ids=scenario_ids,
        )
        records.extend(round_records)
        round_metrics[f"round_{logical_rollout_number:02d}"] = metrics

    keys = {
        (record["scenario_id"], record["logical_rollout_number"]) for record in records
    }
    expected_attempts = len(scenario_ids) * len(logical_rollout_numbers)
    if len(records) != expected_attempts or len(keys) != expected_attempts:
        raise ValueError(
            f"Expected {expected_attempts} unique attempted Gaia2 rollouts, "
            f"found {len(records)} rows and {len(keys)} unique keys"
        )
    counts = Counter(record["scenario_id"] for record in records)
    expected_per_scenario = len(logical_rollout_numbers)
    invalid_counts = {
        scenario_id: counts[scenario_id]
        for scenario_id in scenario_ids
        if counts[scenario_id] != expected_per_scenario
    }
    if invalid_counts:
        raise ValueError(f"Gaia2 logical rollout counts changed: {invalid_counts}")
    records.sort(key=lambda item: (item["logical_rollout_number"], item["scenario_id"]))
    atomic_jsonl(trace_directory / "trace_manifest.jsonl", records)
    rewards = [record["reward"] for record in records]
    metrics = {
        "scenario_count": len(scenario_ids),
        "logical_rollout_numbers": list(logical_rollout_numbers),
        "attempted_rollouts": len(records),
        "unique_attempted_rollouts": len(keys),
        "rollouts_per_scenario": expected_per_scenario,
        "passed_rollouts": sum(reward > 0 for reward in rewards),
        "failed_rollouts": sum(reward <= 0 for reward in rewards),
        "reward_sum": sum(rewards),
        "fixed_denominator_score": sum(rewards) / len(rewards),
        "rounds": round_metrics,
        "trace_manifest": str(trace_directory / "trace_manifest.jsonl"),
    }
    return metrics, records


def write_round_plugin_config(base_plugin: Path, round_directory: Path) -> Path:
    plugin = json.loads(base_plugin.read_text(encoding="utf-8"))
    plugin["sidecar_root"] = str(round_directory / "decomposer_sidecars")
    plugin["output_dir"] = str(round_directory)
    path = round_directory / "configuration" / "are_plugin.json"
    atomic_json(path, plugin)
    return path


def _runtime_configs(
    local_repo: Path,
    directory: Path,
    experiment: DecomposerExperiment,
) -> tuple[Path, Path]:
    config_dir = directory / "configuration"
    service_path = config_dir / "service.json"
    plugin_path = config_dir / "are_plugin.json"
    langgraph_path, langgraph_runtime_directory = langgraph_runtime_paths(directory)
    if experiment.requires_openrouter:
        manager = {
            "model": experiment.manager_served_name,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY_DECOMPOSER",
            "temperature": 1.0,
            "top_p": 1.0,
            "use_responses_api": True,
            "reasoning": {"effort": "high"},
            "timeout": 3300,
            "max_retries": 2,
        }
    elif experiment.requires_llm_proxy:
        manager = {
            "model": experiment.manager_served_name,
            "base_url": f"http://127.0.0.1:{experiment.manager_port}/v1",
            "api_key": "EMPTY",
            "temperature": (
                experiment.temperature
                if experiment.remote_manager_extra_body
                else 1.0
            ),
            "top_p": (
                experiment.top_p if experiment.remote_manager_extra_body else 1.0
            ),
            "use_responses_api": True,
            "timeout": 3300,
            "max_retries": 2,
        }
        if experiment.remote_manager_extra_body:
            manager.update(
                {
                    "presence_penalty": experiment.presence_penalty,
                    "max_completion_tokens": experiment.max_completion_tokens,
                    "extra_body": experiment.remote_manager_extra_body,
                }
            )
    else:
        manager_extra_body: dict[str, Any] = {
            "top_k": experiment.top_k,
            "include_reasoning": experiment.manager_thinking,
            "chat_template_kwargs": {"enable_thinking": experiment.manager_thinking},
        }
        if experiment.min_p is not None:
            manager_extra_body["min_p"] = experiment.min_p
        if experiment.repetition_penalty is not None:
            manager_extra_body["repetition_penalty"] = experiment.repetition_penalty
        manager = {
            "model": experiment.manager_served_name,
            "base_url": f"http://127.0.0.1:{experiment.manager_port}/v1",
            "api_key": "EMPTY",
            "temperature": experiment.temperature,
            "top_p": experiment.top_p,
            "max_completion_tokens": experiment.max_completion_tokens,
            "use_responses_api": False,
            "extra_body": manager_extra_body,
        }
        if experiment.presence_penalty is not None:
            manager["presence_penalty"] = experiment.presence_penalty
    manager["parallel_tool_calls"] = experiment.manager_parallel_tool_calls
    service = {
        "manager": manager,
        "decomposer_system_prompt_profile": experiment.prompt_profile,
        "decomposer_system_prompt_addendum_profile": (
            experiment.manager_prompt_addendum_profile
        ),
        "subagent_types": [
            {
                "subagent_type_id": "gaia2_worker",
                "description": (
                    f"{experiment.worker_served_name} worker with authenticated access to "
                    "the current Gaia2 scenario tools."
                ),
                "assistant_id": "gaia2_worker",
                "url": f"http://127.0.0.1:{experiment.subagent_port}",
            }
        ],
        "manager_recursion_limit": experiment.manager_recursion_limit,
        "subagent_recursion_limit": experiment.subagent_recursion_limit,
    }
    if experiment.manager_max_model_calls is not None:
        service["manager_max_model_calls"] = experiment.manager_max_model_calls
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
                "backend": experiment.manager_backend,
                "served_name": experiment.manager_served_name,
                "thinking": experiment.manager_thinking,
                "parallel_tool_calls": experiment.manager_parallel_tool_calls,
                "max_model_calls": experiment.manager_max_model_calls,
                "recursion_limit": experiment.manager_recursion_limit,
                **(
                    {"path": str(experiment.manager_checkpoint)}
                    if experiment.manager_checkpoint is not None
                    else {}
                ),
                **(
                    {
                        "endpoint": os.environ.get(
                            experiment.manager_upstream_url_env or "", ""
                        ),
                        "response_tool_parser": (
                            experiment.manager_response_tool_parser
                        ),
                        "reasoning_mode": experiment.manager_reasoning_mode,
                    }
                    if experiment.requires_llm_proxy
                    else {}
                ),
            },
            "subagent": {
                "path": str(experiment.worker_checkpoint),
                "served_name": experiment.worker_served_name,
                "thinking": experiment.worker_thinking,
                "max_model_calls": experiment.subagent_max_model_calls,
                "recursion_limit": experiment.subagent_recursion_limit,
            },
        },
    }
    atomic_json(service_path, service)
    atomic_json(plugin_path, plugin)
    atomic_json(
        langgraph_path,
        {
            "dependencies": ["."],
            "graphs": {
                "gaia2_worker": "gyms.gaia2.subagents.graphs:gaia2_worker",
            },
            "python_version": "3.12",
            "disable_persistence": True,
        },
    )
    langgraph_runtime_directory.mkdir(parents=True, exist_ok=True)
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
    *,
    purpose: str = "evaluation",
    partition: str = "full",
    concurrency: int | None = None,
    rollout_offset: int = 0,
    domain: Gaia2Domain = DOMAIN,
) -> dict[str, Any]:
    spec = get_domain_spec(domain)
    manifest_path = preparation_manifest(experiment, spec.name)
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
        if experiment.requires_openrouter:
            services.append(openrouter_proxy_command(experiment))
            gpu_assignments = {}
        else:
            services.append(simple_vllm_command(experiment))
            gpu_assignments = {"policy_vllm": visible_devices[0]}
    else:
        manager_command, worker_command = decomposer_vllm_commands(experiment)
        if experiment.requires_llm_proxy:
            services.append(remote_manager_proxy_command(experiment))
        if manager_command is not None:
            services.append(manager_command)
        services.append(worker_command)
        service_config = directory / "configuration" / "service.json"
        plugin_directory = (
            directory / f"round_{rollout_offset + 1:02d}"
            if purpose == "trace-generation"
            else directory
        )
        plugin_config = plugin_directory / "configuration" / "are_plugin.json"
        langgraph_argv, langgraph_cwd = langgraph_command(experiment, directory)
        services.append(langgraph_argv)
        services.append(service_command(experiment, service_config))
        if experiment.requires_local_manager:
            gpu_assignments = {
                "manager_vllm": visible_devices[0],
                "worker_vllm": visible_devices[1],
            }
        else:
            gpu_assignments = {"worker_vllm": visible_devices[0]}
    selected_dataset_root = partition_dataset_root(partition, spec.name)
    effective_concurrency = concurrency or experiment.concurrency
    first_round_output = (
        directory / f"round_{rollout_offset + 1:02d}"
        if purpose == "trace-generation"
        else directory
    )
    first_are_command = are_command(
        experiment,
        benchmark=benchmark,
        dataset_root=selected_dataset_root,
        output=first_round_output,
        judge_endpoint=os.environ.get("LLM_PROXY_URL", "<LLM_PROXY_URL>"),
        num_repeats=1 if purpose == "trace-generation" else num_repeats,
        limit=limit,
        plugin_config=plugin_config,
        concurrency=effective_concurrency,
        domain=spec.name,
    )
    plan = {
        "experiment": experiment.name,
        "kind": experiment.kind,
        "split": SPLIT,
        "domain": spec.name,
        "purpose": purpose,
        "partition": partition,
        "num_repeats": num_repeats,
        "rollout_offset": rollout_offset,
        "logical_rollout_numbers": list(
            range(rollout_offset + 1, rollout_offset + num_repeats + 1)
        ),
        "concurrency": effective_concurrency,
        "limit": limit,
        "decomposer_system_prompt_profile": (
            experiment.prompt_profile
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_system_prompt_addendum_profile": (
            experiment.manager_prompt_addendum_profile
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_system_prompt_sha256": prompt_sha256(experiment),
        "manager_parallel_tool_calls": (
            experiment.manager_parallel_tool_calls
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "runtime_configuration": runtime_configuration(experiment),
        "gpu_assignments": gpu_assignments,
        "preparation_manifest": str(manifest_path),
        "services": [shlex.join(command) for command in services],
        "are_benchmark": shlex.join(first_are_command),
        "output_dir": str(directory),
    }
    if isinstance(experiment, DecomposerExperiment):
        langgraph_config, langgraph_runtime_directory = langgraph_runtime_paths(
            directory
        )
        plan["langgraph_runtime"] = {
            "config": str(langgraph_config),
            "working_directory": str(langgraph_runtime_directory),
            "file_persistence": False,
        }
    if purpose == "trace-generation":
        plan["are_rounds"] = [
            {
                "logical_rollout_number": logical_rollout_number,
                "output_dir": str(directory / f"round_{logical_rollout_number:02d}"),
                "command": shlex.join(
                    are_command(
                        experiment,
                        benchmark=benchmark,
                        dataset_root=selected_dataset_root,
                        output=directory / f"round_{logical_rollout_number:02d}",
                        judge_endpoint=os.environ.get(
                            "LLM_PROXY_URL", "<LLM_PROXY_URL>"
                        ),
                        num_repeats=1,
                        limit=limit,
                        plugin_config=(
                            directory
                            / f"round_{logical_rollout_number:02d}"
                            / "configuration"
                            / "are_plugin.json"
                        ),
                        concurrency=effective_concurrency,
                        domain=spec.name,
                    )
                ),
            }
            for logical_rollout_number in range(
                rollout_offset + 1, rollout_offset + num_repeats + 1
            )
        ]
    return plan


def execute_trace_generation(local_repo: Path, args: argparse.Namespace) -> int:
    spec = get_domain_spec(getattr(args, "domain", DOMAIN))
    if not spec.supports_trace_generation:
        raise ValueError(f"Gaia2 {spec.name} does not support trace generation")
    requested_prompt_profile = getattr(args, "prompt_profile", None)
    experiment = select_prompt_profile(
        get_experiment(args.experiment), requested_prompt_profile
    )
    if not isinstance(experiment, DecomposerExperiment):
        raise ValueError("Gaia2 trace generation requires a Decomposer experiment")
    if args.partition != "train":
        raise ValueError(
            "Gaia2 trace generation is restricted to the pinned train partition"
        )
    visible_devices = selected_cuda_devices(experiment, args.cuda_visible_devices)
    directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else trace_output_dir(
            experiment,
            args.num_repeats,
            args.rollout_offset,
            args.partition,
            args.limit,
            prompt_profile=requested_prompt_profile,
            domain=spec.name,
        )
    )
    logical_rollout_numbers = tuple(
        range(
            args.rollout_offset + 1,
            args.rollout_offset + args.num_repeats + 1,
        )
    )
    all_scenario_ids = partition_scenario_ids(args.partition, domain=spec.name)
    scenario_ids = (
        all_scenario_ids[: args.limit] if args.limit is not None else all_scenario_ids
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
                    purpose=args.purpose,
                    partition=args.partition,
                    concurrency=args.concurrency,
                    rollout_offset=args.rollout_offset,
                    domain=spec.name,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    identity = run_identity(
        experiment,
        domain=spec.name,
        purpose=args.purpose,
        partition=args.partition,
        num_repeats=args.num_repeats,
        concurrency=args.concurrency or experiment.concurrency,
        limit=args.limit,
        rollout_offset=args.rollout_offset,
    )
    marker = directory / ".trace_done.json"
    if marker.is_file() and not args.force:
        validate_run_identity(marker, identity, require_complete=True)
        print(f"Skip (completed): {marker}")
        return 0
    previous_status = directory / "run_status.json"
    if directory.exists() and not args.force:
        if previous_status.is_file():
            validate_run_identity(previous_status, identity, require_complete=False)
        elif any(directory.glob("round_[0-9][0-9]")):
            raise ValueError(
                "Gaia2 trace rounds exist without a run identity; use --force to "
                f"archive them: {directory}"
            )
    manifest = validate_preparation(
        experiment, partition=args.partition, domain=spec.name
    )
    judge_endpoint = os.environ.get("LLM_PROXY_URL", "").rstrip("/")
    judge_key = os.environ.get("LLM_PROXY_MASTER_KEY", "")
    if not judge_endpoint or not judge_key:
        raise RuntimeError("LLM_PROXY_URL and LLM_PROXY_MASTER_KEY are required")
    if experiment.requires_openrouter:
        if not os.environ.get("OPENROUTER_API_KEY_DECOMPOSER"):
            raise RuntimeError("OPENROUTER_API_KEY_DECOMPOSER is required")
        if not (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")):
            raise RuntimeError("HTTPS_PROXY or https_proxy is required for OpenRouter")
    check_judge(judge_endpoint, judge_key)

    archived = archive_attempt(directory) if args.force and directory.exists() else None
    directory.mkdir(parents=True, exist_ok=True)
    logs = directory / "logs"
    status_path = directory / "run_status.json"
    started = time.monotonic()
    status: dict[str, Any] = {
        "schema_version": 1,
        "state": "starting",
        **identity,
        "scenario_count": len(scenario_ids),
        "logical_rollout_numbers": list(logical_rollout_numbers),
        "manager_parallel_tool_calls": experiment.manager_parallel_tool_calls,
        "cuda_visible_devices": list(visible_devices),
        "output_dir": str(directory),
        "preparation_manifest": str(preparation_manifest(experiment, spec.name)),
        "started_at": utc_now(),
        "completed_logical_rollouts": [],
    }
    if archived is not None:
        status["archived_attempt"] = str(archived)
    if experiment.requires_llm_proxy:
        status["manager"] = {
            "backend": experiment.manager_backend,
            "model": experiment.manager_served_name,
            "endpoint": judge_endpoint,
            "response_tool_parser": experiment.manager_response_tool_parser,
            "reasoning_mode": experiment.manager_reasoning_mode,
        }
    atomic_json(status_path, status)

    staged_gaia2 = Path(manifest["gaia2"]["staged_repo"])
    benchmark = Path(manifest["gaia2"]["runtime"]["are_benchmark"])
    env = _base_environment(local_repo, staged_gaia2, directory, judge_key)
    supervisor = Supervisor(logs, env)
    metrics: dict[str, Any]
    try:
        status["state"] = "agent_services_startup"
        atomic_json(status_path, status)
        manager_command, worker_command = decomposer_vllm_commands(experiment)
        manager_process = None
        if experiment.requires_llm_proxy:
            proxy_process = supervisor.start(
                "remote_manager_proxy",
                remote_manager_proxy_command(experiment),
                cwd=local_repo,
            )
            wait_http(
                f"http://127.0.0.1:{experiment.manager_port}/health",
                [proxy_process],
                300,
            )
        if manager_command is not None:
            manager_process = supervisor.start(
                "manager_vllm",
                manager_command,
                cwd=local_repo,
                env={"CUDA_VISIBLE_DEVICES": visible_devices[0]},
            )
        worker_device_index = 1 if experiment.requires_local_manager else 0
        worker_process = supervisor.start(
            "worker_vllm",
            worker_command,
            cwd=local_repo,
            env={"CUDA_VISIBLE_DEVICES": visible_devices[worker_device_index]},
        )
        if manager_process is not None:
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
        service_config, base_plugin = _runtime_configs(
            local_repo, directory, experiment
        )
        subagent_env = subagent_environment(experiment)
        langgraph_argv, langgraph_cwd = langgraph_command(experiment, directory)
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

        are_env = dict(env)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            are_env.pop(name, None)
        completed: list[int] = []
        for logical_rollout_number in logical_rollout_numbers:
            round_directory = directory / f"round_{logical_rollout_number:02d}"
            round_marker = round_directory / ".round_done.json"
            if round_marker.is_file():
                validate_trace_round(
                    round_directory,
                    trace_directory=directory,
                    logical_rollout_number=logical_rollout_number,
                    scenario_ids=scenario_ids,
                )
                completed.append(logical_rollout_number)
                status["completed_logical_rollouts"] = completed
                atomic_json(status_path, status)
                print(f"Skip (completed round): {round_marker}")
                continue

            archived_round = None
            if round_directory.exists():
                try:
                    round_metrics, _ = validate_trace_round(
                        round_directory,
                        trace_directory=directory,
                        logical_rollout_number=logical_rollout_number,
                        scenario_ids=scenario_ids,
                    )
                except (OSError, ValueError) as error:
                    print(f"Archive incomplete round {logical_rollout_number}: {error}")
                    archived_round = archive_attempt(round_directory)
                else:
                    complete_trace_round(
                        round_directory,
                        logical_rollout_number=logical_rollout_number,
                        scenario_count=len(scenario_ids),
                        concurrency=args.concurrency or experiment.concurrency,
                        metrics=round_metrics,
                        recovered_from_unmarked_artifacts=True,
                    )
                    completed.append(logical_rollout_number)
                    status["completed_logical_rollouts"] = completed
                    atomic_json(status_path, status)
                    print(f"Recovered completed round: {round_marker}")
                    continue

            round_directory.mkdir(parents=True, exist_ok=True)
            round_logs = round_directory / "logs"
            round_logs.mkdir(parents=True, exist_ok=True)
            plugin_config = write_round_plugin_config(base_plugin, round_directory)
            status["state"] = "trace_generation"
            status["active_logical_rollout"] = logical_rollout_number
            atomic_json(status_path, status)
            command = are_command(
                experiment,
                benchmark=benchmark,
                dataset_root=partition_dataset_root(args.partition, spec.name),
                output=round_directory,
                judge_endpoint=judge_endpoint,
                num_repeats=1,
                limit=args.limit,
                plugin_config=plugin_config,
                concurrency=args.concurrency,
                domain=spec.name,
            )
            with (round_logs / "are_benchmark.log").open(
                "a", encoding="utf-8"
            ) as stream:
                subprocess.run(
                    command,
                    check=True,
                    cwd=staged_gaia2,
                    env=are_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            supervisor.assert_running()
            round_metrics, _ = validate_trace_round(
                round_directory,
                trace_directory=directory,
                logical_rollout_number=logical_rollout_number,
                scenario_ids=scenario_ids,
            )
            complete_trace_round(
                round_directory,
                logical_rollout_number=logical_rollout_number,
                scenario_count=len(scenario_ids),
                concurrency=args.concurrency or experiment.concurrency,
                metrics=round_metrics,
                archived_round=archived_round,
            )
            completed.append(logical_rollout_number)
            status["completed_logical_rollouts"] = completed
            atomic_json(status_path, status)

        status.pop("active_logical_rollout", None)
        status["state"] = "aggregation"
        atomic_json(status_path, status)
        metrics, _ = aggregate_trace_manifest(
            directory,
            logical_rollout_numbers=logical_rollout_numbers,
            scenario_ids=scenario_ids,
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


def execute(local_repo: Path, args: argparse.Namespace) -> int:
    if args.purpose == "trace-generation":
        return execute_trace_generation(local_repo, args)
    spec = get_domain_spec(getattr(args, "domain", DOMAIN))
    if args.partition not in ("full", "test"):
        raise ValueError("Gaia2 evaluation supports only full or pinned test data")
    if args.rollout_offset:
        raise ValueError("--rollout-offset is only valid for trace generation")
    requested_prompt_profile = getattr(args, "prompt_profile", None)
    experiment = select_prompt_profile(
        get_experiment(args.experiment), requested_prompt_profile
    )
    visible_devices = selected_cuda_devices(experiment, args.cuda_visible_devices)
    directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else output_dir(
            experiment,
            args.num_repeats,
            args.limit,
            partition=args.partition,
            prompt_profile=requested_prompt_profile,
            domain=spec.name,
        )
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
                    purpose=args.purpose,
                    partition=args.partition,
                    concurrency=args.concurrency,
                    rollout_offset=args.rollout_offset,
                    domain=spec.name,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    identity = run_identity(
        experiment,
        domain=spec.name,
        purpose=args.purpose,
        partition=args.partition,
        num_repeats=args.num_repeats,
        concurrency=args.concurrency or experiment.concurrency,
        limit=args.limit,
    )
    marker = directory / ".eval_done.json"
    if marker.is_file() and not args.force:
        validate_run_identity(marker, identity, require_complete=True)
        print(f"Skip (completed): {marker}")
        return 0
    manifest = validate_preparation(
        experiment, partition=args.partition, domain=spec.name
    )
    judge_endpoint = os.environ.get("LLM_PROXY_URL", "").rstrip("/")
    judge_key = os.environ.get("LLM_PROXY_MASTER_KEY", "")
    if not judge_endpoint or not judge_key:
        raise RuntimeError("LLM_PROXY_URL and LLM_PROXY_MASTER_KEY are required")
    if experiment.requires_openrouter:
        if not os.environ.get("OPENROUTER_API_KEY_DECOMPOSER"):
            raise RuntimeError("OPENROUTER_API_KEY_DECOMPOSER is required")
        if not (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")):
            raise RuntimeError("HTTPS_PROXY or https_proxy is required for OpenRouter")
    check_judge(judge_endpoint, judge_key)

    comparison_baselines: list[dict[str, Any]] | None = None
    if (
        spec.name == "execution"
        and args.partition == "test"
        and args.limit is None
        and args.num_repeats == 3
    ):
        from gyms.gaia2.comparison import collect_baseline_summaries

        comparison_baselines = collect_baseline_summaries()

    archived = archive_attempt(directory) if directory.exists() else None
    directory.mkdir(parents=True, exist_ok=True)
    logs = directory / "logs"
    status_path = directory / "run_status.json"
    started = time.monotonic()
    status: dict[str, Any] = {
        "schema_version": 1,
        "state": "starting",
        **identity,
        "manager_parallel_tool_calls": (
            experiment.manager_parallel_tool_calls
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "cuda_visible_devices": list(visible_devices),
        "output_dir": str(directory),
        "preparation_manifest": str(preparation_manifest(experiment, spec.name)),
        "started_at": utc_now(),
    }
    if archived is not None:
        status["archived_attempt"] = str(archived)
    if isinstance(experiment, DecomposerExperiment) and experiment.requires_llm_proxy:
        status["manager"] = {
            "backend": experiment.manager_backend,
            "model": experiment.manager_served_name,
            "endpoint": judge_endpoint,
            "response_tool_parser": experiment.manager_response_tool_parser,
            "reasoning_mode": experiment.manager_reasoning_mode,
        }
    atomic_json(status_path, status)

    staged_gaia2 = Path(manifest["gaia2"]["staged_repo"])
    benchmark = Path(manifest["gaia2"]["runtime"]["are_benchmark"])
    env = _base_environment(local_repo, staged_gaia2, directory, judge_key)
    if isinstance(experiment, SimpleExperiment):
        env["ARE_MAX_ITERATIONS"] = str(experiment.max_model_calls)
    supervisor = Supervisor(logs, env)
    plugin_config: Path | None = None
    try:
        status["state"] = "agent_services_startup"
        atomic_json(status_path, status)
        if isinstance(experiment, SimpleExperiment):
            if experiment.requires_openrouter:
                process = supervisor.start(
                    "openrouter_policy_proxy",
                    openrouter_proxy_command(experiment),
                    cwd=local_repo,
                )
                wait_http(
                    f"http://127.0.0.1:{experiment.port}/health",
                    [process],
                    300,
                )
            else:
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
                simple_sampling_parameters(experiment), separators=(",", ":")
            )
        else:
            manager_command, worker_command = decomposer_vllm_commands(experiment)
            manager_process = None
            if experiment.requires_llm_proxy:
                proxy_process = supervisor.start(
                    "remote_manager_proxy",
                    remote_manager_proxy_command(experiment),
                    cwd=local_repo,
                )
                wait_http(
                    f"http://127.0.0.1:{experiment.manager_port}/health",
                    [proxy_process],
                    300,
                )
            if manager_command is not None:
                manager_process = supervisor.start(
                    "manager_vllm",
                    manager_command,
                    cwd=local_repo,
                    env={"CUDA_VISIBLE_DEVICES": visible_devices[0]},
                )
            worker_device_index = 1 if experiment.requires_local_manager else 0
            worker_process = supervisor.start(
                "worker_vllm",
                worker_command,
                cwd=local_repo,
                env={"CUDA_VISIBLE_DEVICES": visible_devices[worker_device_index]},
            )
            if manager_process is not None:
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
            subagent_env = subagent_environment(experiment)
            langgraph_argv, langgraph_cwd = langgraph_command(experiment, directory)
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
            dataset_root=partition_dataset_root(args.partition, spec.name),
            output=directory,
            judge_endpoint=judge_endpoint,
            num_repeats=args.num_repeats,
            limit=args.limit,
            plugin_config=plugin_config,
            concurrency=args.concurrency,
            domain=spec.name,
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
        all_scenario_ids = partition_scenario_ids(
            args.partition, domain=spec.name
        )
        metrics = validate_result(
            directory,
            num_repeats=args.num_repeats,
            limit=args.limit,
            scenario_count=len(all_scenario_ids),
            scenario_ids=all_scenario_ids,
        )
        atomic_json(directory / "metrics.json", metrics)
        if comparison_baselines is not None:
            from gyms.gaia2.comparison import build_heldout_comparison

            comparison = build_heldout_comparison(
                experiment,
                directory,
                baselines=comparison_baselines,
            )
            comparison_path = directory / "comparison.json"
            atomic_json(comparison_path, comparison)
            status["comparison_report"] = str(comparison_path)
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
    parser.add_argument("--domain", choices=DOMAINS, default=DOMAIN)
    parser.add_argument(
        "--purpose",
        choices=("evaluation", "trace-generation"),
        default="evaluation",
    )
    parser.add_argument("--partition", choices=PARTITIONS, default="full")
    parser.add_argument("--num-repeats", type=positive_int, default=3)
    parser.add_argument("--concurrency", type=positive_int)
    parser.add_argument("--prompt-profile", choices=DECOMPOSER_PROMPT_PROFILES)
    parser.add_argument("--rollout-offset", type=nonnegative_int, default=0)
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
