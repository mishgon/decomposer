"""Run one Workplace Assistant experiment locally, without submitting a job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decomposer.prompts import (  # noqa: E402
    DECOMPOSER_PROMPT_PROFILES,
    resolve_decomposer_system_prompt,
)

from gyms.workplace_assistant.experiments import (  # noqa: E402
    ARTIFACTS_ROOT,
    HF_HOME,
    PROJECT_VENV,
    RUN_PURPOSES,
    SPLIT_ROWS,
    SPLITS,
    UV_BIN,
    UV_CACHE,
    DecomposerExperiment,
    Experiment,
    ModelServer,
    RunPurpose,
    SimpleExperiment,
    component_venv_root,
    decomposer_dataset,
    decomposer_prompt_profile,
    get_experiment,
    gym_venv,
    models_for_experiment,
    output_dir,
    preparation_manifest,
    run_name,
    source_dataset,
    validate_num_repeats,
    validate_purpose_for_experiment,
)


WORKPLACE_ROLLOUT_FAILURE_POLICY = "score_zero"
SIMPLE_VLLM_PORT = 8000
LANGGRAPH_PORT = 2024
GYM_HEAD_PORT = 11000
GYM_COMPONENT_PORT_LOW = 11001
GYM_COMPONENT_PORT_HIGH = 11999
MAX_TCP_PORT = 65535
SUBAGENT_MODEL_URLS_ENV = "WORKPLACE_ASSISTANT_MODEL_BASE_URLS_JSON"


@dataclass(frozen=True)
class WorkplacePortLayout:
    offset: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise TypeError("port offset must be an integer")
        if self.offset < 0:
            raise ValueError("port offset must be non-negative")
        if self.gym_component_high > MAX_TCP_PORT:
            raise ValueError(
                f"port offset {self.offset} exceeds the maximum TCP port "
                f"({self.gym_component_high} > {MAX_TCP_PORT})"
            )

    def shifted(self, base_port: int) -> int:
        port = base_port + self.offset
        if not 1 <= port <= MAX_TCP_PORT:
            raise ValueError(
                f"port offset {self.offset} maps {base_port} outside the valid "
                f"TCP port range: {port}"
            )
        return port

    @property
    def simple_vllm(self) -> int:
        return self.shifted(SIMPLE_VLLM_PORT)

    @property
    def langgraph(self) -> int:
        return self.shifted(LANGGRAPH_PORT)

    @property
    def gym_head(self) -> int:
        return self.shifted(GYM_HEAD_PORT)

    @property
    def gym_component_low(self) -> int:
        return self.shifted(GYM_COMPONENT_PORT_LOW)

    @property
    def gym_component_high(self) -> int:
        return self.shifted(GYM_COMPONENT_PORT_HIGH)

    def model_port(self, model: ModelServer) -> int:
        return self.shifted(model.port)

    def manager_proxy_port(self, experiment: DecomposerExperiment) -> int:
        if experiment.manager_proxy_port is None:
            raise ValueError(f"{experiment.name} has no local manager proxy")
        return self.shifted(experiment.manager_proxy_port)

    def as_dict(self, experiment: Experiment) -> dict[str, Any]:
        value: dict[str, Any] = {
            "offset": self.offset,
            "simple_agent_vllm": self.simple_vllm,
            "langgraph": self.langgraph,
            "gym_head": self.gym_head,
            "gym_component_range": [
                self.gym_component_low,
                self.gym_component_high,
            ],
            "model_servers": {},
            "manager_proxy": None,
        }
        if isinstance(experiment, DecomposerExperiment):
            value["model_servers"] = {
                model.model_id: self.model_port(model)
                for model in models_for_experiment(experiment)
            }
            if experiment.requires_llm_proxy:
                value["manager_proxy"] = self.manager_proxy_port(experiment)
        return value


DEFAULT_PORT_LAYOUT = WorkplacePortLayout()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def loopback_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def local_run_name(
    experiment: Experiment,
    num_repeats: int,
    *,
    prompt_profile: str | None,
    ports: WorkplacePortLayout,
) -> str:
    identity = run_name(
        experiment,
        num_repeats,
        prompt_profile=prompt_profile,
    )
    if ports.offset:
        identity += f"-port-offset-{ports.offset}"
    return identity


def hydra_flow_mapping(values: Mapping[str, Any]) -> str:
    def encode(value: Any) -> str:
        if isinstance(value, Mapping):
            return hydra_flow_mapping(value)
        return json.dumps(value)

    return (
        "{" + ",".join(f"{key}:{encode(value)}" for key, value in values.items()) + "}"
    )


def runtime_configuration(experiment: Experiment) -> dict[str, Any]:
    if isinstance(experiment, SimpleExperiment):
        return {
            "max_model_len": experiment.max_model_len,
            "max_output_tokens": experiment.max_output_tokens,
            "max_model_calls": experiment.max_steps,
            "thinking": experiment.thinking,
            "sampling": {
                "temperature": experiment.temperature,
                "top_p": experiment.top_p,
                "top_k": experiment.top_k,
                "min_p": experiment.min_p,
                "presence_penalty": experiment.presence_penalty,
                "repetition_penalty": experiment.repetition_penalty,
            },
        }
    return {
        "max_model_len": experiment.max_model_len,
        "max_output_tokens": experiment.max_output_tokens,
        "evaluation_prompt_profile": experiment.evaluation_prompt_profile,
        "manager": {
            "backend": experiment.manager_backend,
            "reasoning_mode": experiment.manager_reasoning_mode,
            "max_model_calls": experiment.manager_max_model_calls,
            "sampling": (
                experiment.remote_manager_extra_body
                if experiment.manager_sampling is not None
                else None
            ),
        },
        "subagent": {
            "max_model_calls": experiment.subagent_max_model_calls,
            "recursion_limit": experiment.subagent_recursion_limit,
        },
    }


def _decomposer_config_source(
    local_repo: Path, experiment: DecomposerExperiment
) -> Path:
    return (
        local_repo
        / "gyms"
        / "workplace_assistant"
        / "configs"
        / experiment.gym_config_filename
    )


def _configured_policy_model(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        value = config["policy_model"]["responses_api_models"][name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Workplace config is missing policy model {name!r}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"Workplace policy model {name!r} must be a mapping")
    return value


def _rewrite_decomposer_endpoints(
    config: dict[str, Any],
    experiment: DecomposerExperiment,
    ports: WorkplacePortLayout,
) -> list[dict[str, str]]:
    rewrites: list[dict[str, str]] = []

    def replace_url(
        value: dict[str, Any], key: str, expected: str, replacement: str
    ) -> None:
        observed = value.get(key)
        if observed != expected:
            raise ValueError(
                f"{experiment.gym_config_filename}: expected {key}={expected!r}, "
                f"found {observed!r}"
            )
        value[key] = replacement
        if replacement != expected:
            rewrites.append({"from": expected, "to": replacement})

    if experiment.requires_local_manager:
        policy = _configured_policy_model(config, "vllm_model")
        manager_model_id = policy.get("model")
        matches = [
            model
            for model in models_for_experiment(experiment)
            if model.model_id == manager_model_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{experiment.gym_config_filename}: policy model "
                f"{manager_model_id!r} does not identify one local model server"
            )
        manager = matches[0]
        replace_url(
            policy,
            "base_url",
            loopback_url(manager.port),
            loopback_url(ports.model_port(manager)),
        )
    elif experiment.requires_llm_proxy:
        policy = _configured_policy_model(config, "openai_model")
        if experiment.manager_proxy_port is None:
            raise ValueError(f"{experiment.name} has no manager proxy port")
        replace_url(
            policy,
            "openai_base_url",
            loopback_url(experiment.manager_proxy_port),
            loopback_url(ports.manager_proxy_port(experiment)),
        )

    try:
        subagent_types = config["decomposer"]["responses_api_agents"][
            "decomposer_agent"
        ]["subagent_types"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{experiment.gym_config_filename}: subagent_types is missing"
        ) from error
    if not isinstance(subagent_types, list) or not subagent_types:
        raise ValueError(
            f"{experiment.gym_config_filename}: subagent_types must be a non-empty list"
        )
    for subagent in subagent_types:
        if not isinstance(subagent, dict):
            raise ValueError(
                f"{experiment.gym_config_filename}: subagent entry must be a mapping"
            )
        replace_url(
            subagent,
            "url",
            f"http://127.0.0.1:{LANGGRAPH_PORT}",
            f"http://127.0.0.1:{ports.langgraph}",
        )
    return rewrites


def runtime_decomposer_config(
    local_repo: Path,
    experiment: DecomposerExperiment,
    directory: Path,
    ports: WorkplacePortLayout,
    *,
    materialize: bool,
) -> tuple[Path, dict[str, Any]]:
    source = _decomposer_config_source(local_repo, experiment)
    source_locator = source.relative_to(local_repo).as_posix()
    source_bytes = source.read_bytes()
    config = yaml.safe_load(source_bytes)
    if not isinstance(config, dict):
        raise ValueError(f"Workplace config must contain a mapping: {source}")
    rewrites = _rewrite_decomposer_endpoints(config, experiment, ports)
    if ports.offset == 0:
        return source, {
            "source": source_locator,
            "path": source_locator,
            "sha256": sha256_bytes(source_bytes),
            "rewrites": rewrites,
        }

    destination = directory / "configuration" / "workplace_assistant.runtime.yaml"
    runtime_bytes = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
    if materialize:
        atomic_bytes(destination, runtime_bytes)
    return destination, {
        "source": source_locator,
        "path": str(destination),
        "sha256": sha256_bytes(runtime_bytes),
        "rewrites": rewrites,
    }


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
        env: dict[str, str] | None = None,
    ) -> subprocess.Popen[Any]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stream = (self.log_dir / f"{name}.log").open("a")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env={**self.env, **(env or {})},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.processes.append((name, process, stream))
        return process

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
    url: str, processes: Sequence[subprocess.Popen[Any]], timeout: float
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


def simple_vllm_command(
    experiment: SimpleExperiment,
    ports: WorkplacePortLayout = DEFAULT_PORT_LAYOUT,
) -> list[str]:
    if experiment.requires_openrouter or experiment.checkpoint is None:
        raise ValueError(f"{experiment.name} does not use a local vLLM server")
    command = [
        str(PROJECT_VENV / "bin" / "vllm"),
        "serve",
        str(experiment.checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        str(ports.simple_vllm),
        "--tensor-parallel-size",
        str(experiment.num_gpus),
        "--gpu-memory-utilization",
        str(experiment.gpu_memory_utilization),
        "--max-model-len",
        str(experiment.max_model_len),
        "--trust-remote-code",
        "--language-model-only",
        "--default-chat-template-kwargs",
        json.dumps(experiment.chat_template_kwargs, separators=(",", ":")),
    ]
    if experiment.gdn_prefill_backend is not None:
        command.extend(["--gdn-prefill-backend", experiment.gdn_prefill_backend])
    if experiment.effective_reasoning_parser is not None:
        command.extend(["--reasoning-parser", experiment.effective_reasoning_parser])
    command.extend(
        ["--enable-auto-tool-choice", "--tool-call-parser", experiment.tool_call_parser]
    )
    return command


def decomposer_vllm_command(
    model: ModelServer,
    experiment: DecomposerExperiment,
    ports: WorkplacePortLayout = DEFAULT_PORT_LAYOUT,
) -> list[str]:
    command = [
        str(PROJECT_VENV / "bin" / "vllm"),
        "serve",
        str(model.snapshot),
        "--served-model-name",
        model.model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(ports.model_port(model)),
        "--max-model-len",
        str(experiment.max_model_len),
        "--max-num-seqs",
        str(experiment.max_num_seqs),
        "--gpu-memory-utilization",
        str(model.gpu_memory_utilization),
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        model.tool_call_parser,
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": model.thinking}, separators=(",", ":")),
    ]
    if model.dtype is not None:
        command.extend(["--dtype", model.dtype])
    if model.reasoning_parser is not None:
        command.extend(["--reasoning-parser", model.reasoning_parser])
    if model.gdn_prefill_backend is not None:
        command.extend(["--gdn-prefill-backend", model.gdn_prefill_backend])
    return command


def remote_manager_proxy_command(
    experiment: DecomposerExperiment,
    ports: WorkplacePortLayout = DEFAULT_PORT_LAYOUT,
) -> list[str]:
    if not experiment.requires_llm_proxy:
        raise ValueError(f"{experiment.name} does not use the LLM proxy manager")
    required = {
        "manager_proxy_port": experiment.manager_proxy_port,
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
        str(ports.manager_proxy_port(experiment)),
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


def langgraph_command(
    local_repo: Path,
    experiment: DecomposerExperiment,
    ports: WorkplacePortLayout = DEFAULT_PORT_LAYOUT,
) -> tuple[list[str], Path]:
    if ports.offset or experiment.subagent_graph in {"qwen35", "repository"}:
        directory = local_repo / "gyms" / "workplace_assistant" / "subagents"
    else:
        directory = (
            local_repo
            / "external"
            / "Gym"
            / "responses_api_agents"
            / "decomposer_agent"
            / "subagents"
        )
    return (
        [
            str(PROJECT_VENV / "bin" / "langgraph"),
            "dev",
            "--config",
            str(directory / "langgraph.json"),
            "--host",
            "127.0.0.1",
            "--port",
            str(ports.langgraph),
            "--n-jobs-per-worker",
            str(experiment.langgraph_jobs),
            "--no-browser",
            "--no-reload",
        ],
        directory,
    )


def gym_start_command(
    local_repo: Path,
    experiment: Experiment,
    *,
    purpose: RunPurpose,
    gym_bin: Path,
    component_root: Path,
    logs: Path,
    prompt_profile: str | None = None,
    ports: WorkplacePortLayout = DEFAULT_PORT_LAYOUT,
    config_path: Path | None = None,
) -> list[str]:
    validate_purpose_for_experiment(experiment, purpose)
    common = [
        "+head_server.host=127.0.0.1",
        f"+head_server.port={ports.gym_head}",
        f"+port_range_low={ports.gym_component_low}",
        f"+port_range_high={ports.gym_component_high}",
        "+skip_venv_if_present=true",
        f"+uv_venv_dir={component_root}",
        f"+uv_cache_dir={UV_CACHE}",
        f"+nemo_gym_log_dir={logs / 'gym_components'}",
    ]
    if isinstance(experiment, DecomposerExperiment):
        config = config_path or _decomposer_config_source(local_repo, experiment)
        prompt_profile = decomposer_prompt_profile(
            purpose,
            prompt_profile,
            experiment.evaluation_prompt_profile,
        )
        prompt_override = (
            "++decomposer.responses_api_agents.decomposer_agent."
            f"decomposer_system_prompt_profile={prompt_profile}"
        )
        manager_call_limit_override = (
            "++decomposer.responses_api_agents.decomposer_agent."
            f"manager_max_model_calls={experiment.manager_max_model_calls}"
        )
        subagent_recursion_limit_override = (
            "++decomposer.responses_api_agents.decomposer_agent."
            f"subagent_recursion_limit={experiment.subagent_recursion_limit}"
        )
        command = [
            str(gym_bin),
            "env",
            "start",
            "--config",
            str(config),
            prompt_override,
            manager_call_limit_override,
            subagent_recursion_limit_override,
            *common,
        ]
        if experiment.max_output_tokens is not None:
            command.append(
                "++responses_create_params.max_output_tokens="
                f"{experiment.max_output_tokens}"
            )
        if experiment.requires_openrouter:
            command.append(
                "++policy_api_key=${oc.env:OPENROUTER_API_KEY_DECOMPOSER}"
            )
        return command
    if experiment.requires_openrouter:
        if experiment.model_id is None or experiment.base_url is None:
            raise ValueError(f"{experiment.name}: incomplete OpenRouter configuration")
        command = [
            str(gym_bin),
            "env",
            "start",
            "--environment",
            "workplace_assistant",
            "--model-type",
            "openai_model",
            "--model",
            experiment.model_id,
            "--model-url",
            experiment.base_url,
        ]
        if experiment.remote_extra_body:
            command.append(
                "++policy_model.responses_api_models.openai_model.extra_body="
                + hydra_flow_mapping(experiment.remote_extra_body)
            )
        command.extend(
            [
                "++policy_api_key=${oc.env:OPENROUTER_API_KEY_DECOMPOSER}",
                (
                    "++workplace_assistant_simple_agent.responses_api_agents."
                    f"simple_agent.max_steps={experiment.max_steps}"
                ),
                *common,
            ]
        )
        return command
    if experiment.checkpoint is None:
        raise ValueError(f"{experiment.name}: local policy has no checkpoint")
    return [
        str(gym_bin),
        "env",
        "start",
        "--environment",
        "workplace_assistant",
        "--model-type",
        "vllm_model",
        "--model",
        str(experiment.checkpoint),
        "--model-url",
        loopback_url(ports.simple_vllm),
        "--model-api-key",
        "EMPTY",
        (
            "++policy_model.responses_api_models.vllm_model.extra_body="
            + hydra_flow_mapping(experiment.extra_body)
        ),
        (
            "++workplace_assistant_simple_agent.responses_api_agents."
            f"simple_agent.max_steps={experiment.max_steps}"
        ),
        *common,
    ]


def gym_eval_command(
    experiment: Experiment,
    *,
    gym_bin: Path,
    split: str,
    output: Path,
    num_repeats: int,
    limit: int | None,
    resume: bool,
    concurrency: int | None = None,
    ports: WorkplacePortLayout = DEFAULT_PORT_LAYOUT,
) -> list[str]:
    dataset = (
        decomposer_dataset(split)
        if isinstance(experiment, DecomposerExperiment)
        else source_dataset(split)
    )
    agent = (
        "decomposer"
        if experiment.kind == "decomposer"
        else "workplace_assistant_simple_agent"
    )
    command = [
        str(gym_bin),
        "eval",
        "run",
        "--no-serve",
        "--agent",
        agent,
        "--input",
        str(dataset),
        "--output",
        str(output),
        "--num-repeats",
        str(num_repeats),
        "--concurrency",
        str(concurrency or experiment.concurrency),
        f"+rollout_failure_policy={WORKPLACE_ROLLOUT_FAILURE_POLICY}",
        "+head_server.host=127.0.0.1",
        f"+head_server.port={ports.gym_head}",
    ]
    if isinstance(experiment, SimpleExperiment):
        command.extend(
            [
                "--temperature",
                str(experiment.temperature),
                "--top-p",
                str(experiment.top_p),
            ]
        )
        if experiment.max_output_tokens is not None:
            command.extend(
                ["--max-output-tokens", str(experiment.max_output_tokens)]
            )
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if resume:
        command.append("--resume")
    return command


def _validate_file_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    for filename, item in (manifest.get("files") or {}).items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != item.get("size"):
            raise ValueError(f"Prepared model file changed or is missing: {path}")
        if item.get("sha256") and sha256_file(path) != item["sha256"]:
            raise ValueError(f"Prepared model file checksum changed: {path}")


def validate_preparation(
    repo_root: Path, experiment: Experiment, split: str
) -> dict[str, Any]:
    path = preparation_manifest(split, experiment.name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Run `python -m gyms.workplace_assistant.prepare eval --split {split} "
            f"--experiment {experiment.name}` first: {path}"
        )
    manifest = json.loads(path.read_text())
    if manifest.get("experiment") != {"name": experiment.name, "kind": experiment.kind}:
        raise ValueError(f"Preparation manifest does not match {experiment.name}")
    if manifest.get("split") != split:
        raise ValueError(f"Preparation manifest does not match split {split}")
    dataset_key = "decomposer" if experiment.kind == "decomposer" else "simple"
    dataset = manifest["datasets"][dataset_key]
    expected_path = (
        decomposer_dataset(split)
        if experiment.kind == "decomposer"
        else source_dataset(split)
    )
    if Path(dataset["path"]) != expected_path or dataset["rows"] != SPLIT_ROWS[split]:
        raise ValueError("Preparation manifest contains unexpected dataset metadata")
    if sha256_file(expected_path) != dataset["sha256"]:
        raise ValueError(f"Prepared dataset checksum changed: {expected_path}")
    expected_gym = gym_venv(repo_root) / "bin" / "gym"
    if Path(manifest["gym"]["runtime"]["gym_bin"]) != expected_gym:
        raise ValueError("Preparation manifest points at an unexpected Gym runtime")
    if not expected_gym.is_file():
        raise FileNotFoundError(expected_gym)
    gym_lock = repo_root / "external" / "Gym" / "uv.lock"
    if sha256_file(gym_lock) != manifest["gym"]["runtime"]["uv_lock_sha256"]:
        raise ValueError("Prepared Gym runtime lock checksum changed")
    gym_commit = subprocess.run(
        ["git", "-C", str(repo_root / "external" / "Gym"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if gym_commit != manifest["gym"]["commit"]:
        raise ValueError("Prepared Gym revision does not match the current submodule")
    expected_components = component_venv_root(repo_root)
    if Path(manifest["component_runtime"]["root"]) != expected_components:
        raise ValueError(
            "Preparation manifest points at unexpected component environments"
        )
    for component in manifest["component_runtime"]["components"].values():
        python = Path(component["python"])
        if not python.is_file() or not (python.parent / "activate").is_file():
            raise FileNotFoundError(component["python"])
    if isinstance(experiment, SimpleExperiment):
        policy = manifest["models"]["policy"]
        if experiment.requires_openrouter:
            expected_policy = {
                "backend": experiment.backend,
                "model_id": experiment.model_id,
                "base_url": experiment.base_url,
                "reasoning_effort": experiment.reasoning_effort,
            }
            if policy != expected_policy:
                raise ValueError(
                    "Preparation manifest points at an unexpected remote policy"
                )
        else:
            if experiment.checkpoint is None:
                raise ValueError(f"{experiment.name}: local policy has no checkpoint")
            if Path(policy["path"]) != experiment.checkpoint:
                raise ValueError(
                    "Preparation manifest points at an unexpected policy checkpoint"
                )
            _validate_file_manifest(experiment.checkpoint, policy)
    else:
        selected = {
            model.model_id: model for model in models_for_experiment(experiment)
        }
        prepared = manifest["models"]["subagents"]
        if set(prepared) != set(selected):
            raise ValueError("Preparation manifest contains unexpected subagent models")
        for model_id, model in selected.items():
            if Path(prepared[model_id]["path"]) != model.snapshot:
                raise ValueError(f"Unexpected model path for {model_id}")
            _validate_file_manifest(model.snapshot, prepared[model_id])
        if experiment.requires_llm_proxy:
            expected_manager = {
                "backend": experiment.manager_backend,
                "model_id": experiment.manager_model_id,
                "upstream_url_env": experiment.manager_upstream_url_env,
                "api_key_env": experiment.manager_api_key_env,
                "response_tool_parser": experiment.manager_response_tool_parser,
                "reasoning_mode": experiment.manager_reasoning_mode,
                "verify_tls": experiment.manager_verify_tls,
            }
            if manifest["models"].get("manager") != expected_manager:
                raise ValueError(
                    "Preparation manifest points at an unexpected remote manager"
                )
    return manifest


def validate_result(
    split: str,
    num_repeats: int,
    *,
    rollout_path: Path,
    limit: int | None,
) -> dict[str, Any]:
    aggregate = rollout_path.with_stem(
        rollout_path.stem + "_aggregate_metrics"
    ).with_suffix(".json")
    if not rollout_path.is_file():
        raise FileNotFoundError(rollout_path)
    if not aggregate.is_file():
        raise FileNotFoundError(aggregate)
    json.loads(aggregate.read_text())
    tasks = min(SPLIT_ROWS[split], limit) if limit is not None else SPLIT_ROWS[split]
    expected_rows = tasks * num_repeats
    expected_keys = {
        (task_index, rollout_index)
        for task_index in range(tasks)
        for rollout_index in range(num_repeats)
    }
    observed_keys: set[tuple[int, int]] = set()
    rollout_error_types: Counter[str] = Counter()
    with rollout_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {rollout_path}:{line_number}"
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"Expected an object at {rollout_path}:{line_number}"
                )
            task_index = row.get("_ng_task_index")
            rollout_index = row.get("_ng_rollout_index")
            if (
                isinstance(task_index, bool)
                or not isinstance(task_index, int)
                or isinstance(rollout_index, bool)
                or not isinstance(rollout_index, int)
            ):
                raise ValueError(
                    f"Invalid rollout identity at {rollout_path}:{line_number}: "
                    f"({task_index!r}, {rollout_index!r})"
                )
            key = (task_index, rollout_index)
            if key not in expected_keys:
                raise ValueError(
                    f"Unexpected rollout identity at {rollout_path}:{line_number}: "
                    f"{key}"
                )
            if key in observed_keys:
                raise ValueError(
                    f"Duplicate rollout identity at {rollout_path}:{line_number}: "
                    f"{key}"
                )
            observed_keys.add(key)

            reward = row.get("reward")
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or float(reward) not in {0.0, 1.0}
            ):
                raise ValueError(
                    f"Invalid binary reward at {rollout_path}:{line_number}: "
                    f"{reward!r}"
                )

            rollout_error = row.get("_ng_rollout_error")
            if rollout_error is None:
                continue
            if not isinstance(rollout_error, Mapping):
                raise ValueError(
                    f"Invalid rollout error at {rollout_path}:{line_number}"
                )
            error_type = rollout_error.get("type")
            status_code = rollout_error.get("status_code")
            detail = rollout_error.get("detail")
            if (
                not isinstance(error_type, str)
                or not error_type
                or (
                    status_code is not None
                    and (
                        isinstance(status_code, bool)
                        or not isinstance(status_code, int)
                    )
                )
                or not isinstance(detail, str)
                or float(reward) != 0.0
            ):
                raise ValueError(
                    f"Invalid reward-0 rollout error at "
                    f"{rollout_path}:{line_number}"
                )
            rollout_error_types[error_type] += 1

    missing_keys = expected_keys - observed_keys
    if missing_keys:
        preview = sorted(missing_keys)[:10]
        raise ValueError(
            f"Expected {expected_rows} unique rollouts, found {len(observed_keys)}; "
            f"missing {len(missing_keys)} identities (first: {preview})"
        )
    actual_rows = len(observed_keys)
    return {
        "rollouts": str(rollout_path),
        "aggregate_metrics": str(aggregate),
        "task_rows": tasks,
        "rollout_rows": actual_rows,
        "rollout_error_rows": sum(rollout_error_types.values()),
        "rollout_error_types": dict(sorted(rollout_error_types.items())),
    }


def _attempt_entries(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.iterdir()
        if path.name != "attempts"
        and not (path.is_dir() and re.fullmatch(r"smoke_[1-9][0-9]*", path.name))
    ]


def archive_attempt(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    entries = _attempt_entries(directory)
    if not entries:
        return None
    status_path = directory / "run_status.json"
    if status_path.is_file():
        started_at = str(
            json.loads(status_path.read_text()).get("started_at", utc_now())
        )
    else:
        started_at = utc_now()
    attempt_id = (
        started_at.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace("+0000", "Z")
        .replace(".", "_")
    )
    archive = directory / "attempts" / attempt_id
    if archive.exists():
        archive = directory / "attempts" / f"{attempt_id}_{os.getpid()}"
    archive.mkdir(parents=True)
    for path in entries:
        shutil.move(str(path), str(archive / path.name))
    return archive


def _legacy_purpose(experiment: Experiment) -> RunPurpose:
    if isinstance(experiment, DecomposerExperiment):
        return "trace-generation"
    return "evaluation"


def _attempt_metadata(directory: Path) -> dict[str, Any] | None:
    for filename in (".eval_done.json", "run_status.json"):
        path = directory / filename
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid existing run metadata: {path}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"Existing run metadata must be an object: {path}")
        return value
    return None


def validate_existing_attempt_identity(
    directory: Path,
    experiment: Experiment,
    *,
    purpose: RunPurpose,
    split: str,
    num_repeats: int,
    limit: int | None,
    force: bool,
    prompt_profile: str | None = None,
    ports: WorkplacePortLayout = DEFAULT_PORT_LAYOUT,
    runtime_gym_config_sha256: str | None = None,
) -> None:
    if not directory.is_dir() or force:
        return
    entries = _attempt_entries(directory)
    if not entries:
        return
    metadata = _attempt_metadata(directory)
    if metadata is None:
        raise RuntimeError(
            f"Existing output has no run identity metadata: {directory}. "
            "Use a different --output-dir or pass --force to archive it."
        )

    existing_purpose = metadata.get("purpose", _legacy_purpose(experiment))
    mismatches: list[str] = []
    expected = {
        "experiment": experiment.name,
        "split": split,
        "num_repeats": num_repeats,
        "limit": limit,
        "purpose": purpose,
        "port_offset": ports.offset,
        "port_layout": ports.as_dict(experiment),
        "runtime_configuration": runtime_configuration(experiment),
    }
    if runtime_gym_config_sha256 is not None:
        expected["runtime_gym_config_sha256"] = runtime_gym_config_sha256
    if isinstance(experiment, SimpleExperiment):
        expected["simple_agent_max_steps"] = experiment.max_steps
    else:
        expected.update(
            {
                "decomposer_manager_max_model_calls": (
                    experiment.manager_max_model_calls
                ),
                "decomposer_subagent_max_model_calls": (
                    experiment.subagent_max_model_calls
                ),
                "decomposer_subagent_recursion_limit": (
                    experiment.subagent_recursion_limit
                ),
            }
        )
    existing_offset = metadata.get("port_offset", 0)
    observed = {
        **metadata,
        "purpose": existing_purpose,
        "port_offset": existing_offset,
    }
    if "port_layout" not in observed:
        observed["port_layout"] = (
            ports.as_dict(experiment)
            if existing_offset == 0 and ports.offset == 0
            else None
        )
    if (
        runtime_gym_config_sha256 is not None
        and "runtime_gym_config_sha256" not in observed
    ):
        observed["runtime_gym_config_sha256"] = (
            runtime_gym_config_sha256
            if existing_offset == 0 and ports.offset == 0
            else None
        )
    for field, expected_value in expected.items():
        if field not in observed:
            if field in {
                "simple_agent_max_steps",
                "decomposer_manager_max_model_calls",
                "decomposer_subagent_max_model_calls",
                "decomposer_subagent_recursion_limit",
            }:
                mismatches.append(f"missing required identity field {field}")
        elif observed[field] != expected_value:
            mismatches.append(
                f"{field}={observed[field]!r} (requested {expected_value!r})"
            )

    if isinstance(experiment, DecomposerExperiment):
        existing_profile = metadata.get(
            "decomposer_system_prompt_profile",
            "teacher" if "purpose" not in metadata else None,
        )
        expected_profile = decomposer_prompt_profile(
            purpose,
            prompt_profile,
            experiment.evaluation_prompt_profile,
        )
        if existing_profile != expected_profile:
            mismatches.append(
                "decomposer_system_prompt_profile="
                f"{existing_profile!r} (requested {expected_profile!r})"
            )

    if mismatches:
        raise RuntimeError(
            f"Existing output identity does not match this run: {directory}: "
            + "; ".join(mismatches)
            + ". Use a different --output-dir or pass --force to archive it."
        )


def _base_environment(
    local_repo: Path,
    experiment: Experiment,
    run_identity: str,
    ports: WorkplacePortLayout = DEFAULT_PORT_LAYOUT,
) -> dict[str, str]:
    worker_ip = socket.gethostbyname(socket.gethostname())
    no_proxy = ",".join(
        value
        for value in (
            os.environ.get("NO_PROXY", ""),
            "127.0.0.1",
            "localhost",
            worker_ip,
        )
        if value
    )
    cache_root = ARTIFACTS_ROOT / "cache" / "workplace-assistant" / run_identity
    caches = {
        "VLLM_CACHE_ROOT": cache_root / "vllm",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "FLASHINFER_WORKSPACE_BASE": cache_root / "flashinfer",
    }
    for path in caches.values():
        path.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PATH": os.pathsep.join(
            (
                str(UV_BIN.parent),
                str(PROJECT_VENV / "bin"),
                str(gym_venv(local_repo) / "bin"),
                os.environ.get("PATH", ""),
            )
        ),
        "PYTHONPATH": os.pathsep.join(
            (
                str(local_repo),
                str(local_repo / "src"),
                str(local_repo / "external" / "Gym"),
                os.environ.get("PYTHONPATH", ""),
            )
        ),
        "HF_HOME": str(HF_HOME),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "UV_CACHE_DIR": str(UV_CACHE),
        "UV_LINK_MODE": "copy",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
        "LD_LIBRARY_PATH": os.pathsep.join(
            (str(PROJECT_VENV / "lib"), os.environ.get("LD_LIBRARY_PATH", ""))
        ),
        **{name: str(path) for name, path in caches.items()},
    }
    if isinstance(experiment, DecomposerExperiment):
        env["DECOMPOSER_SUBAGENT_MAX_MODEL_CALLS"] = str(
            experiment.subagent_max_model_calls
        )
        env[SUBAGENT_MODEL_URLS_ENV] = json.dumps(
            {
                model.model_id: loopback_url(ports.model_port(model))
                for model in models_for_experiment(experiment)
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        if experiment.max_output_tokens is not None:
            env["DECOMPOSER_SUBAGENT_MAX_COMPLETION_TOKENS"] = str(
                experiment.max_output_tokens
            )
    if isinstance(experiment, SimpleExperiment) and not experiment.requires_openrouter:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env.pop(name, None)
    return env


def _dry_plan(
    local_repo: Path,
    experiment: Experiment,
    purpose: RunPurpose,
    split: str,
    num_repeats: int,
    limit: int | None,
    directory: Path,
    visible_devices: tuple[str, ...],
    concurrency: int | None = None,
    prompt_profile: str | None = None,
    ports: WorkplacePortLayout = DEFAULT_PORT_LAYOUT,
) -> dict[str, Any]:
    validate_purpose_for_experiment(experiment, purpose)
    logs = directory / "logs"
    gym_bin = gym_venv(local_repo) / "bin" / "gym"
    runtime_config_path: Path | None = None
    runtime_config_metadata: dict[str, Any] | None = None
    if isinstance(experiment, SimpleExperiment):
        if experiment.requires_openrouter:
            services = []
            gpu_assignments = {}
        else:
            services = [simple_vllm_command(experiment, ports)]
            gpu_assignments = {"policy_vllm": ",".join(visible_devices)}
    else:
        models = models_for_experiment(experiment)
        runtime_config_path, runtime_config_metadata = runtime_decomposer_config(
            local_repo,
            experiment,
            directory,
            ports,
            materialize=False,
        )
        services = []
        if experiment.requires_llm_proxy:
            services.append(remote_manager_proxy_command(experiment, ports))
        services.extend(
            decomposer_vllm_command(model, experiment, ports) for model in models
        )
        services.append(langgraph_command(local_repo, experiment, ports)[0])
        gpu_assignments = {
            f"subagent_vllm_{ports.model_port(model)}": visible_devices[model.gpu]
            for model in models
        }
    rollout_path = directory / "rollouts.jsonl"
    return {
        "decomposer_system_prompt_profile": (
            decomposer_prompt_profile(
                purpose,
                prompt_profile,
                experiment.evaluation_prompt_profile,
            )
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_system_prompt_sha256": (
            hashlib.sha256(
                resolve_decomposer_system_prompt(
                    decomposer_prompt_profile(
                        purpose,
                        prompt_profile,
                        experiment.evaluation_prompt_profile,
                    )
                ).encode("utf-8")
            ).hexdigest()
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "experiment": experiment.name,
        "gpu_assignments": gpu_assignments,
        "kind": experiment.kind,
        "simple_agent_max_steps": (
            experiment.max_steps if isinstance(experiment, SimpleExperiment) else None
        ),
        "decomposer_manager_max_model_calls": (
            experiment.manager_max_model_calls
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_subagent_max_model_calls": (
            experiment.subagent_max_model_calls
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_subagent_recursion_limit": (
            experiment.subagent_recursion_limit
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "purpose": purpose,
        "port_offset": ports.offset,
        "port_layout": ports.as_dict(experiment),
        "rollout_failure_policy": WORKPLACE_ROLLOUT_FAILURE_POLICY,
        "runtime_configuration": runtime_configuration(experiment),
        "runtime_gym_config": runtime_config_metadata,
        "split": split,
        "services": [shlex.join(command) for command in services],
        "gym_start": shlex.join(
            gym_start_command(
                local_repo,
                experiment,
                purpose=purpose,
                gym_bin=gym_bin,
                component_root=component_venv_root(local_repo),
                logs=logs,
                prompt_profile=prompt_profile,
                ports=ports,
                config_path=runtime_config_path,
            )
        ),
        "gym_eval": shlex.join(
            gym_eval_command(
                experiment,
                gym_bin=gym_bin,
                split=split,
                output=rollout_path,
                num_repeats=num_repeats,
                limit=limit,
                resume=rollout_path.exists(),
                concurrency=concurrency,
                ports=ports,
            )
        ),
        "output_dir": str(directory),
    }


def parse_cuda_visible_devices(value: str) -> tuple[str, ...]:
    devices = tuple(device.strip() for device in value.split(","))
    if not devices or any(not device for device in devices):
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


def selected_output_dir(experiment: Experiment, args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    ports = WorkplacePortLayout(getattr(args, "port_offset", 0))
    directory = output_dir(
        experiment,
        args.split,
        args.num_repeats,
        None,
        purpose=args.purpose,
        prompt_profile=getattr(args, "prompt_profile", None),
    )
    if ports.offset:
        directory = directory.with_name(
            f"{directory.name}-port-offset-{ports.offset}"
        )
    return directory if args.limit is None else directory / f"smoke_{args.limit}"


def execute(local_repo: Path, args: argparse.Namespace) -> int:
    experiment = get_experiment(args.experiment)
    ports = WorkplacePortLayout(getattr(args, "port_offset", 0))
    validate_num_repeats(args.num_repeats)
    purpose = validate_purpose_for_experiment(experiment, args.purpose)
    requested_prompt_profile = getattr(args, "prompt_profile", None)
    if requested_prompt_profile is not None and not isinstance(
        experiment, DecomposerExperiment
    ):
        raise ValueError("--prompt-profile is only valid for Decomposer experiments")
    resolved_prompt_profile = (
        decomposer_prompt_profile(
            purpose,
            requested_prompt_profile,
            experiment.evaluation_prompt_profile,
        )
        if isinstance(experiment, DecomposerExperiment)
        else None
    )
    directory = selected_output_dir(experiment, args)
    visible_devices = selected_cuda_devices(experiment, args.cuda_visible_devices)
    if args.dry:
        manifest = preparation_manifest(args.split, experiment.name)
        if not manifest.is_file():
            print(
                f"Warning: preparation manifest is missing: {manifest}", file=sys.stderr
            )
        print(
            json.dumps(
                _dry_plan(
                    local_repo,
                    experiment,
                    purpose,
                    args.split,
                    args.num_repeats,
                    args.limit,
                    directory,
                    visible_devices,
                    args.concurrency,
                    requested_prompt_profile,
                    ports,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    runtime_config_path: Path | None = None
    runtime_config_metadata: dict[str, Any] | None = None
    if isinstance(experiment, DecomposerExperiment):
        runtime_config_path, runtime_config_metadata = runtime_decomposer_config(
            local_repo,
            experiment,
            directory,
            ports,
            materialize=False,
        )
    validate_existing_attempt_identity(
        directory,
        experiment,
        purpose=purpose,
        split=args.split,
        num_repeats=args.num_repeats,
        limit=args.limit,
        force=args.force,
        prompt_profile=requested_prompt_profile,
        ports=ports,
        runtime_gym_config_sha256=(
            runtime_config_metadata["sha256"]
            if runtime_config_metadata is not None
            else None
        ),
    )
    marker = directory / ".eval_done.json"
    if marker.is_file() and not args.force:
        print(f"Skip (completed): {marker}")
        return 0

    manifest = validate_preparation(local_repo, experiment, args.split)
    if experiment.requires_openrouter:
        if not os.environ.get("OPENROUTER_API_KEY_DECOMPOSER"):
            raise RuntimeError("OPENROUTER_API_KEY_DECOMPOSER is not set")
        if not (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")):
            raise RuntimeError("HTTPS_PROXY or https_proxy is required for OpenRouter")
    if isinstance(experiment, DecomposerExperiment) and experiment.requires_llm_proxy:
        required = (
            experiment.manager_upstream_url_env,
            experiment.manager_api_key_env,
        )
        missing = [name for name in required if not name or not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "Remote manager environment is not set: " + ", ".join(missing)
            )

    archived_attempt = archive_attempt(directory) if args.force else None
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(experiment, DecomposerExperiment):
        runtime_config_path, runtime_config_metadata = runtime_decomposer_config(
            local_repo,
            experiment,
            directory,
            ports,
            materialize=True,
        )
    logs = directory / "logs"
    rollout_path = directory / "rollouts.jsonl"
    resume = rollout_path.is_file() and not args.force
    status_path = directory / "run_status.json"
    started = time.monotonic()
    status: dict[str, Any] = {
        "schema_version": 5,
        "state": "starting",
        "experiment": experiment.name,
        "kind": experiment.kind,
        "simple_agent_max_steps": (
            experiment.max_steps if isinstance(experiment, SimpleExperiment) else None
        ),
        "decomposer_manager_max_model_calls": (
            experiment.manager_max_model_calls
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_subagent_max_model_calls": (
            experiment.subagent_max_model_calls
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_subagent_recursion_limit": (
            experiment.subagent_recursion_limit
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "purpose": purpose,
        "port_offset": ports.offset,
        "port_layout": ports.as_dict(experiment),
        "rollout_failure_policy": WORKPLACE_ROLLOUT_FAILURE_POLICY,
        "runtime_configuration": runtime_configuration(experiment),
        "runtime_gym_config": runtime_config_metadata,
        "runtime_gym_config_sha256": (
            runtime_config_metadata["sha256"]
            if runtime_config_metadata is not None
            else None
        ),
        "decomposer_system_prompt_profile": (
            resolved_prompt_profile
            if isinstance(experiment, DecomposerExperiment)
            else None
        ),
        "decomposer_system_prompt_sha256": (
            hashlib.sha256(
                resolve_decomposer_system_prompt(resolved_prompt_profile).encode(
                    "utf-8"
                )
            ).hexdigest()
            if resolved_prompt_profile is not None
            else None
        ),
        "run_name": local_run_name(
            experiment,
            args.num_repeats,
            prompt_profile=requested_prompt_profile,
            ports=ports,
        ),
        "split": args.split,
        "num_repeats": args.num_repeats,
        "concurrency": args.concurrency or experiment.concurrency,
        "limit": args.limit,
        "output_dir": str(directory),
        "cuda_visible_devices": list(visible_devices),
        "started_at": utc_now(),
        "timings_seconds": {},
        "preparation_manifest": str(preparation_manifest(args.split, experiment.name)),
    }
    if isinstance(experiment, DecomposerExperiment) and experiment.requires_llm_proxy:
        status["manager"] = {
            "backend": experiment.manager_backend,
            "model": experiment.manager_model_id,
            "endpoint": os.environ.get(experiment.manager_upstream_url_env or "", ""),
            "response_tool_parser": experiment.manager_response_tool_parser,
            "reasoning_mode": experiment.manager_reasoning_mode,
        }
    if archived_attempt is not None:
        status["archived_attempt"] = str(archived_attempt)
    atomic_json(status_path, status)

    @contextmanager
    def phase(name: str) -> Iterator[None]:
        phase_started = time.monotonic()
        status["state"] = name
        status["phase_started_at"] = utc_now()
        atomic_json(status_path, status)
        try:
            yield
        finally:
            status["timings_seconds"][name] = round(time.monotonic() - phase_started, 3)
            atomic_json(status_path, status)

    env = _base_environment(local_repo, experiment, status["run_name"], ports)
    supervisor = Supervisor(logs, env)
    gym_bin = Path(manifest["gym"]["runtime"]["gym_bin"])
    component_root = Path(manifest["component_runtime"]["root"])
    try:
        with phase("agent_services_startup"):
            if isinstance(experiment, SimpleExperiment):
                if not experiment.requires_openrouter:
                    model_process = supervisor.start(
                        "policy_vllm",
                        simple_vllm_command(experiment, ports),
                        cwd=local_repo,
                        env={"CUDA_VISIBLE_DEVICES": ",".join(visible_devices)},
                    )
                    wait_http(
                        f"http://127.0.0.1:{ports.simple_vllm}/v1/models",
                        [model_process],
                        1800,
                    )
            else:
                if experiment.requires_llm_proxy:
                    proxy_process = supervisor.start(
                        "remote_manager_proxy",
                        remote_manager_proxy_command(experiment, ports),
                        cwd=local_repo,
                    )
                    wait_http(
                        (
                            "http://127.0.0.1:"
                            f"{ports.manager_proxy_port(experiment)}/health"
                        ),
                        [proxy_process],
                        300,
                    )
                model_processes: list[subprocess.Popen[Any]] = []
                models = models_for_experiment(experiment)
                for startup_wave in sorted({model.startup_wave for model in models}):
                    wave = [
                        model for model in models if model.startup_wave == startup_wave
                    ]
                    for model in wave:
                        model_processes.append(
                            supervisor.start(
                                f"subagent_vllm_{ports.model_port(model)}",
                                decomposer_vllm_command(model, experiment, ports),
                                cwd=local_repo,
                                env={
                                    "CUDA_VISIBLE_DEVICES": visible_devices[model.gpu]
                                },
                            )
                        )
                    for model in wave:
                        wait_http(
                            f"http://127.0.0.1:{ports.model_port(model)}/health",
                            model_processes,
                            1800,
                        )
                langgraph_argv, langgraph_cwd = langgraph_command(
                    local_repo, experiment, ports
                )
                langgraph = supervisor.start(
                    "langgraph", langgraph_argv, cwd=langgraph_cwd
                )
                wait_http(
                    f"http://127.0.0.1:{ports.langgraph}/docs", [langgraph], 300
                )

        with phase("gym_startup"):
            gym_process = supervisor.start(
                "gym_servers",
                gym_start_command(
                    local_repo,
                    experiment,
                    purpose=purpose,
                    gym_bin=gym_bin,
                    component_root=component_root,
                    logs=logs,
                    prompt_profile=requested_prompt_profile,
                    ports=ports,
                    config_path=runtime_config_path,
                ),
                cwd=local_repo / "external" / "Gym",
            )
            wait_http(
                f"http://127.0.0.1:{ports.gym_head}/server_instances",
                [gym_process],
                300,
            )
            subprocess.run(
                [
                    str(
                        local_repo
                        / "external"
                        / "Gym"
                        / "scripts"
                        / "wait_for_servers.sh"
                    ),
                    str(gym_process.pid),
                    str(ports.gym_head),
                    str(
                        experiment.gym_wait_timeout
                        if isinstance(experiment, SimpleExperiment)
                        else 1200
                    ),
                ],
                check=True,
                cwd=local_repo / "external" / "Gym",
                env=env,
            )

        with phase("rollout"):
            command = gym_eval_command(
                experiment,
                gym_bin=gym_bin,
                split=args.split,
                output=rollout_path,
                num_repeats=args.num_repeats,
                limit=args.limit,
                resume=resume,
                concurrency=args.concurrency,
                ports=ports,
            )
            with (logs / "gym_eval.log").open("a") as stream:
                subprocess.run(
                    command,
                    check=True,
                    cwd=local_repo / "external" / "Gym",
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )

        with phase("validation"):
            result = validate_result(
                args.split,
                args.num_repeats,
                rollout_path=rollout_path,
                limit=args.limit,
            )
            if (
                experiment.name == "qwen36-35b-a3b-teacher-qwen35-4b-non-thinking"
                and purpose == "trace-generation"
                and args.split == "validation"
                and args.num_repeats == 3
                and args.limit is None
            ):
                from gyms.workplace_assistant.comparison import (
                    build_teacher_comparison,
                )

                comparison_path = directory / "comparison.json"
                atomic_json(comparison_path, build_teacher_comparison(directory))
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
            "result": result,
            "gpu_metadata": gpu_metadata,
            "commit": subprocess.run(
                ["git", "-C", str(local_repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        }
    )
    atomic_json(status_path, status)
    atomic_json(marker, status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


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
        raise argparse.ArgumentTypeError("value must be non-negative")
    try:
        WorkplacePortLayout(parsed)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--purpose", choices=RUN_PURPOSES, required=True)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--num-repeats", type=positive_int, default=1)
    parser.add_argument("--concurrency", type=positive_int)
    parser.add_argument(
        "--port-offset",
        type=nonnegative_int,
        default=0,
        help="add this value to every local Workplace service port",
    )
    parser.add_argument("--prompt-profile", choices=DECOMPOSER_PROMPT_PROFILES)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--dry", "--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--cuda-visible-devices",
        type=parse_cuda_visible_devices,
        help=(
            "comma-separated physical GPU IDs mapped to the experiment's logical "
            "GPU slots"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the directory containing rollouts, logs, status, and marker",
    )
    parser.add_argument("--workdir", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workdir is None:
        repo_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return execute(repo_root, args)
    with tempfile.TemporaryDirectory(prefix="decomposer-wa-") as temporary:
        local_repo = Path(temporary) / "repo"
        shutil.copytree(args.workdir, local_repo, symlinks=True)
        return execute(local_repo, args)


if __name__ == "__main__":
    raise SystemExit(main())
