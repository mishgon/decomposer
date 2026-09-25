"""Local runner for Decomposer runs on the tau2 gym.

Starts the manager endpoint (local vLLM, or a credential-isolating proxy for a remote
manager), the Qwen3.5-4B subagent worker, the LangGraph subagent server and the Gym
servers, performs one `gym eval run`, validates the output, and stops every child
process. Structure follows gyms/workplace_assistant/run.py; that code is duplicated
per gym rather than shared, so this is a trimmed copy of the same phases, port
layout, supervisor and status contract.

What runs is an experiment from `experiments.py` on a task pool from `task_pools/`.
The Gym config is generated from the experiment and written into the run directory,
so a run is described completely by its `run_status.json` and `configuration/`.

The tau2 resources server lives at gyms/tau2_gym/gym_components/resources_servers/
and is found through NEMO_GYM_EXTRA_ROOTS, which Gym searches ahead of its own tree
(nemo_gym/__init__.py:52-79). The Gym submodule is untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402

from gyms.tau2_gym.experiments import (  # noqa: E402
    DEFAULT_SUBAGENT_MODEL_ID,
    EXPERIMENTS_BY_NAME,
    LLM_PROXY_API_KEY_ENV,
    LLM_PROXY_URL_ENV,
    LOCAL_SUBAGENT_MODEL_ID,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    SUBAGENT_ASSISTANT_ID,
    SUBAGENT_DESCRIPTION,
    SUBAGENT_MODEL_ENV,
    SUBAGENT_TYPE_ID,
    Tau2Experiment,
    get_experiment,
)
from gyms.qwen_sampling import (  # noqa: E402
    SUBAGENT_MAX_COMPLETION_TOKENS_ENV,
    SUBAGENT_SAMPLING_ENV,
    non_thinking_subagent_sampling_kwargs,
    subagent_sampling_environment,
)
from gyms.remote_model_proxy import require_upstream_models  # noqa: E402
from gyms.tau2_gym.task_pools import load_pool  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_VENV = REPO_ROOT / ".venv"

# Newer of the two artifact roots on Hertz-2; it holds the prepared Gym venv and
# the tau2 venv. Single constant so it is easy to repoint.
ARTIFACTS_ROOT = Path("/home/sukhorukov/decomposer_artifacts_new")
GYM_VENV_ROOT = ARTIFACTS_ROOT / "venvs" / "gym"
COMPONENT_VENV_ROOT = ARTIFACTS_ROOT / "venvs" / "tau2-gym-components"
TAU2_VENV = ARTIFACTS_ROOT / "venvs" / "tau2"
UV_CACHE = ARTIFACTS_ROOT / "cache" / "uv"
RESULTS_ROOT = ARTIFACTS_ROOT / "evaluation" / "results" / "tau2_gym"
DATASETS_ROOT = ARTIFACTS_ROOT / "datasets" / "tau2_gym"

TAU2_CHECKOUT = REPO_ROOT / "external" / "tau2_gym"
# The Gym CLIs run from here, as in the workplace runner, so Hydra's per-invocation
# outputs/ directories land in Gym's ignored tree instead of the repository root.
GYM_CHECKOUT = REPO_ROOT / "external" / "Gym"
TAU2_DATA_DIR = TAU2_CHECKOUT / "data"
GYM_EXTRA_ROOT = REPO_ROOT / "gyms" / "tau2_gym" / "gym_components"
SUBAGENT_DIR = REPO_ROOT / "gyms" / "tau2_gym" / "subagents"

ROLLOUT_FAILURE_POLICY = "score_zero"
VERIFIER_FACTORY = "responses_api_agents.decomposer_agent.app:_subagent_tool_calls_and_final_message"
SUBAGENT_BACKENDS = ("local_vllm", "llm_proxy")
STATUS_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PortLayout:
    offset: int = 0
    # Subagent worker. Hertz-2 is shared and 8025 often serves someone else's
    # Qwen3.5-4B; --subagent-port moves it without touching the other ports.
    qwen35_4b: int = 8025
    manager_vllm: int = 8026
    # 8142 is the workplace gym's manager proxy; keep clear of it.
    subagent_proxy: int = 8143
    manager_proxy: int = 8144
    langgraph: int = 2024
    gym_head: int = 11000
    gym_component_low: int = 11001
    gym_component_high: int = 11999

    def shifted(self) -> "PortLayout":
        if not self.offset:
            return self
        return replace(
            self,
            qwen35_4b=self.qwen35_4b + self.offset,
            manager_vllm=self.manager_vllm + self.offset,
            subagent_proxy=self.subagent_proxy + self.offset,
            manager_proxy=self.manager_proxy + self.offset,
            langgraph=self.langgraph + self.offset,
            gym_head=self.gym_head + self.offset,
            gym_component_low=self.gym_component_low + self.offset,
            gym_component_high=self.gym_component_high + self.offset,
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def loopback_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def gym_lock_hash() -> str:
    lock = REPO_ROOT / "external" / "Gym" / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]


def gym_venv() -> Path:
    return GYM_VENV_ROOT / gym_lock_hash()


def component_venv_root() -> Path:
    digest = hashlib.sha256()
    for path in (REPO_ROOT / "uv.lock", REPO_ROOT / "external" / "Gym" / "uv.lock"):
        digest.update(path.read_bytes())
    return COMPONENT_VENV_ROOT / digest.hexdigest()[:16]


def subset_sha256(tasks_file: Path) -> str:
    """Mirrors tau2_export.subset_sha256, which runs in the tau2 venv."""
    tasks = sorted((str(domain), str(task_id)) for domain, task_id in json.loads(tasks_file.read_text()))
    return hashlib.sha256(json.dumps([list(key) for key in tasks]).encode()).hexdigest()[:12]


def dataset_path(
    pool: str, pool_sha: str, tasks_per_domain: int | None, tasks_file: Path | None = None
) -> Path:
    """Mirrors tau2_export.dataset_filename, which runs in the tau2 venv."""
    suffix = f"-k{tasks_per_domain}" if tasks_per_domain is not None else ""
    if tasks_file is not None:
        suffix += f"-s{subset_sha256(tasks_file)}"
    return DATASETS_ROOT / f"{pool}-{pool_sha}{suffix}.decomposer.jsonl"


def run_name(
    experiment: Tau2Experiment,
    *,
    pool: str,
    tasks_per_domain: int | None,
    num_repeats: int,
    port_offset: int,
) -> str:
    name = f"{experiment.name}-{pool}"
    if tasks_per_domain is not None:
        name += f"-k{tasks_per_domain}"
    name += f"-n{num_repeats}"
    if port_offset:
        name += f"-port-offset-{port_offset}"
    return name


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def checkpoint_fingerprint(checkpoint: Path) -> str:
    """Cheap identity for a served checkpoint: file names, sizes and mtimes.

    Hashing multi-GB weights on every run is too slow; this still changes whenever a
    checkpoint is rewritten, which is what lets an OPD round prove its rollouts came
    from the weights it just exported.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in checkpoint.rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(f"{path.relative_to(checkpoint)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:16]


def gym_config(experiment: Tau2Experiment, ports: PortLayout) -> dict[str, Any]:
    """The complete Gym config for `experiment`."""
    sampling = experiment.manager_sampling
    if experiment.manager_backend == "openrouter":
        create_params: dict[str, Any] = {"temperature": 1.0, "top_p": 1.0}
        policy: dict[str, Any] = {
            "openai_model": {
                "entrypoint": "app.py",
                "openai_base_url": OPENROUTER_BASE_URL,
                # Resolved by OmegaConf inside Gym; the key never touches disk.
                "openai_api_key": "${oc.env:" + OPENROUTER_API_KEY_ENV + "}",
                "openai_model": experiment.manager_model_id,
                **({"extra_body": dict(experiment.manager_extra_body)} if experiment.manager_extra_body else {}),
            }
        }
    elif experiment.manager_backend == "llm_proxy":
        assert sampling is not None
        # The manager proxy's --extra-body-json overrides these server-side; they are
        # set to the same values so the request is honest on its own.
        create_params = {"temperature": sampling.temperature, "top_p": sampling.top_p}
        policy = {
            "openai_model": {
                "entrypoint": "app.py",
                "openai_base_url": loopback_url(ports.manager_proxy),
                "openai_api_key": "EMPTY",
                "openai_model": experiment.manager_model_id,
            }
        }
    else:
        assert sampling is not None
        create_params = {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "presence_penalty": sampling.presence_penalty,
        }
        policy = {
            "vllm_model": {
                "entrypoint": "app.py",
                "base_url": loopback_url(ports.manager_vllm),
                "api_key": "EMPTY",
                "model": experiment.manager_model_id,
                "return_token_id_information": experiment.return_token_ids,
                # Non-thinking Qwen3.5 is served without a reasoning parser.
                "uses_reasoning_parser": False,
                "chat_template_kwargs": experiment.chat_template_kwargs,
                "extra_body": {**sampling.extra_body, "include_reasoning": False},
            }
        }

    return {
        "responses_create_params": create_params,
        "tau2_gym": {
            "resources_servers": {
                "tau2_gym": {
                    "entrypoint": "app.py",
                    "domain": "agent",
                    "verified": False,
                    "description": "tau2 GAIA2-hard domains as a multi-step tool-using environment.",
                    "value": "Improve multi-step tool use and task decomposition.",
                    "language": "en",
                }
            }
        },
        "policy_model": {"responses_api_models": policy},
        "decomposer": {
            "responses_api_agents": {
                "decomposer_agent": {
                    "entrypoint": "app.py",
                    "resources_server": {"type": "resources_servers", "name": "tau2_gym"},
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                    "join_gym_system_and_user_prompts": True,
                    "decomposer_system_prompt_profile": experiment.prompt_profile,
                    "manager_max_model_calls": experiment.manager_max_model_calls,
                    "subagent_recursion_limit": experiment.subagent_recursion_limit,
                    "response_for_verifier_factory": VERIFIER_FACTORY,
                    "subagent_types": [
                        {
                            "subagent_type_id": SUBAGENT_TYPE_ID,
                            "assistant_id": SUBAGENT_ASSISTANT_ID,
                            "url": f"http://127.0.0.1:{ports.langgraph}",
                            "description": SUBAGENT_DESCRIPTION,
                        }
                    ],
                }
            }
        },
    }


class Supervisor:
    """Starts children in their own process group and tears the group down."""

    def __init__(self, log_dir: Path, env: dict[str, str]) -> None:
        self.log_dir = log_dir
        self.env = env
        self.processes: list[tuple[str, subprocess.Popen[Any], Any]] = []

    def start(
        self, name: str, command: list[str], *, cwd: Path, env: dict[str, str] | None = None
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
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGINT)
        deadline = time.monotonic() + 20
        for _, process, _ in reversed(self.processes):
            if process.poll() is None:
                try:
                    process.wait(max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        for _, _, stream in self.processes:
            stream.close()


def wait_http(url: str, processes: Sequence[subprocess.Popen[Any]], timeout: float) -> None:
    """Poll until `url` answers, failing fast if a supervised process dies."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in processes:
            code = process.poll()
            if code is not None:
                raise RuntimeError(f"Server process {process.pid} exited with code {code}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status < 500:
                    return
        except (OSError, TimeoutError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def _uv_bin_dir() -> str:
    """Directory holding `uv`.

    Gym shells out to `uv` when it creates component venvs, and a
    non-interactive shell does not have ~/.local/bin on PATH.
    """
    pinned = ARTIFACTS_ROOT / "tools" / "uv"
    if pinned.is_file() and os.access(pinned, os.X_OK):
        return str(pinned.parent)
    discovered = shutil.which("uv")
    if discovered:
        return str(Path(discovered).parent)
    return str(Path.home() / ".local" / "bin")


def require_free_port(port: int, name: str) -> None:
    """Fail early if `port` is taken.

    Without this a foreign server on the same port satisfies wait_http() while our
    own process dies of EADDRINUSE, and the run fails much later with a confusing
    error. Hertz-2 is shared: 8025 is often already serving someone else's
    Qwen3.5-4B.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(
                f"Port {port} ({name}) is already in use by another process.\n"
                f"Pass --subagent-port <free port> (subagent model server), or "
                f"--port-offset N (everything else)."
            )


def subagent_base_url(ports: PortLayout, backend: str) -> str:
    """Where subagents send their model traffic.

    Both backends look identical to the subagent graph: loopback, `api_key="EMPTY"`.
    For llm_proxy the loopback endpoint is gyms.remote_model_proxy, which injects the
    real credentials and terminates the proxy's self-signed TLS.
    """
    port = ports.subagent_proxy if backend == "llm_proxy" else ports.qwen35_4b
    return loopback_url(port)


def resolve_subagent_model_id(subagent_backend: str, override: str | None) -> str:
    """The model id the subagents request: explicit, else the one the backend serves."""
    if override:
        return override
    return LOCAL_SUBAGENT_MODEL_ID if subagent_backend == "local_vllm" else DEFAULT_SUBAGENT_MODEL_ID


def base_environment(ports: PortLayout, *, subagent_backend: str, subagent_model_id: str) -> dict[str, str]:
    env = dict(os.environ)
    python_path = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "external" / "Gym")]
    existing = env.get("PYTHONPATH")
    if existing:
        python_path.append(existing)
    path_entries = [
        _uv_bin_dir(),
        str(PROJECT_VENV / "bin"),
        str(gym_venv() / "bin"),
        env.get("PATH", ""),
    ]
    env.update(
        {
            "PATH": os.pathsep.join(entry for entry in path_entries if entry),
            "PYTHONPATH": os.pathsep.join(python_path),
            # Lets `gym env start` discover resources_servers/tau2_gym without
            # touching the Gym submodule.
            "NEMO_GYM_EXTRA_ROOTS": str(GYM_EXTRA_ROOT),
            # tau2 ships its data outside the package (tau2/utils/utils.py:17-27).
            "TAU2_DATA_DIR": str(TAU2_DATA_DIR),
            # tau2 logs every tool response at DEBUG through loguru.
            "LOGURU_LEVEL": env.get("LOGURU_LEVEL", "WARNING"),
            "TAU2_GYM_MODEL_BASE_URLS_JSON": json.dumps(
                {subagent_model_id: subagent_base_url(ports, subagent_backend)}
            ),
            SUBAGENT_MODEL_ENV: subagent_model_id,
            "UV_CACHE_DIR": str(UV_CACHE),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.setdefault("DECOMPOSER_SUBAGENT_MAX_MODEL_CALLS", "100")
    # Subagent sampling comes from the experiment (subagent_environment), never the shell.
    env.pop(SUBAGENT_SAMPLING_ENV, None)
    # Nothing but the vLLM servers needs a GPU; each gets its own CUDA_VISIBLE_DEVICES.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def subagent_environment(experiment: Tau2Experiment) -> dict[str, str]:
    """What the LangGraph server needs so subagents sample as the experiment says."""
    return subagent_sampling_environment(experiment.subagent_sampling)


def subagent_sampling_record(env: Mapping[str, str]) -> dict[str, Any]:
    """The sampling the subagent graph will send, as run_status records it."""
    record = non_thinking_subagent_sampling_kwargs(env)
    record["max_completion_tokens"] = (
        int(env[SUBAGENT_MAX_COMPLETION_TOKENS_ENV]) if SUBAGENT_MAX_COMPLETION_TOKENS_ENV in env else None
    )
    return record


def upstream_model_ids(experiment: Tau2Experiment, *, subagent_backend: str, subagent_model_id: str) -> list[str]:
    """Every model this run requests from the shared LLM proxy."""
    models = [experiment.manager_model_id] if experiment.manager_backend == "llm_proxy" else []
    if subagent_backend == "llm_proxy":
        models.append(subagent_model_id)
    return models


def prepare_dataset(
    *,
    pool: str,
    tasks_per_domain: int | None,
    tasks_file: Path | None,
    output: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    """Materialise the Gym dataset by running tau2_export inside the tau2 venv."""
    python = TAU2_VENV / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"tau2 venv missing at {TAU2_VENV}. Create it with:\n"
            f"  uv venv --python 3.12 {TAU2_VENV}\n"
            f"  VIRTUAL_ENV={TAU2_VENV} uv pip install -e {TAU2_CHECKOUT}"
        )
    command = [
        str(python),
        str(REPO_ROOT / "gyms" / "tau2_gym" / "tau2_export.py"),
        "--pool",
        pool,
        "--output",
        str(output),
    ]
    if tasks_per_domain is not None:
        command.extend(["--tasks-per-domain", str(tasks_per_domain)])
    if tasks_file is not None:
        command.extend(["--tasks-file", str(tasks_file)])
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        # capture_output hides the child's traceback; without this the caller sees
        # only CalledProcessError and has to re-run the command by hand.
        raise SystemExit(
            "tau2_export failed (exit "
            f"{result.returncode}).\n--- stderr ---\n{result.stderr.strip()[-4000:]}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def subagent_vllm_command(
    ports: PortLayout, *, model_id: str, max_model_len: int, gpu_memory_utilization: float
) -> list[str]:
    return [
        str(PROJECT_VENV / "bin" / "vllm"),
        "serve",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(ports.qwen35_4b),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        # Every ReAct turn re-sends a growing context; without prefix caching total
        # prefill is quadratic in turn count.
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": False}, separators=(",", ":")),
    ]


def manager_vllm_command(
    experiment: Tau2Experiment,
    checkpoint: Path,
    ports: PortLayout,
    *,
    gpu_memory_utilization: float,
) -> list[str]:
    """Serves a Qwen3.5-4B manager checkpoint as workplace does for its SFT students."""
    return [
        str(PROJECT_VENV / "bin" / "vllm"),
        "serve",
        str(checkpoint),
        "--served-model-name",
        experiment.manager_model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(ports.manager_vllm),
        "--max-model-len",
        str(experiment.manager_max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        "bfloat16",
        "--language-model-only",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        # Non-thinking Qwen3.5 closes the think block inside the prompt, so the
        # completion carries no tags for a reasoning parser.
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": False}, separators=(",", ":")),
        "--gdn-prefill-backend",
        "triton",
    ]


def require_env(names: Sequence[str], hint: str) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} must be set. {hint}")


def remote_proxy_command(
    port: int, *, response_tool_parser: str | None = None, extra_body: dict[str, Any] | None = None
) -> list[str]:
    """Local credential-isolating proxy in front of the shared LLM proxy.

    Subagents (Chat Completions) get neither option: their upstream already returns
    structured tool calls, and --extra-body-json would override the graph's own
    sampling (remote_model_proxy.py:49-54). The manager (Responses API) gets both:
    the qwen3_xml normaliser, and the experiment's sampling as the one source of truth.
    """
    command = [
        str(PROJECT_VENV / "bin" / "python"),
        "-m",
        "gyms.remote_model_proxy",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--upstream-url-env",
        LLM_PROXY_URL_ENV,
        "--api-key-env",
        LLM_PROXY_API_KEY_ENV,
        "--timeout-seconds",
        "3300",
        "--max-retries",
        "2",
        # The shared proxy presents a self-signed certificate.
        "--no-verify-tls",
    ]
    if response_tool_parser is not None:
        command.extend(["--response-tool-parser", response_tool_parser])
    if extra_body:
        command.extend(["--extra-body-json", json.dumps(extra_body, separators=(",", ":"))])
    return command


def langgraph_command(ports: PortLayout, jobs: int) -> list[str]:
    return [
        str(PROJECT_VENV / "bin" / "langgraph"),
        "dev",
        "--config",
        str(SUBAGENT_DIR / "langgraph.json"),
        "--host",
        "127.0.0.1",
        "--port",
        str(ports.langgraph),
        "--n-jobs-per-worker",
        str(jobs),
        "--no-browser",
        "--no-reload",
    ]


def gym_start_command(ports: PortLayout, *, config: Path, logs: Path) -> list[str]:
    return [
        str(gym_venv() / "bin" / "gym"),
        "env",
        "start",
        "--config",
        str(config),
        "+head_server.host=127.0.0.1",
        f"+head_server.port={ports.gym_head}",
        f"+port_range_low={ports.gym_component_low}",
        f"+port_range_high={ports.gym_component_high}",
        "+skip_venv_if_present=true",
        f"+uv_venv_dir={component_venv_root()}",
        f"+uv_cache_dir={UV_CACHE}",
        f"+nemo_gym_log_dir={logs / 'gym_components'}",
    ]


def gym_eval_command(
    ports: PortLayout,
    *,
    dataset: Path,
    output: Path,
    num_repeats: int,
    concurrency: int,
    limit: int | None,
    resume: bool,
) -> list[str]:
    command = [
        str(gym_venv() / "bin" / "gym"),
        "eval",
        "run",
        "--no-serve",
        "--agent",
        "decomposer",
        "--input",
        str(dataset),
        "--output",
        str(output),
        "--num-repeats",
        str(num_repeats),
        "--concurrency",
        str(concurrency),
        f"+rollout_failure_policy={ROLLOUT_FAILURE_POLICY}",
        "+head_server.host=127.0.0.1",
        f"+head_server.port={ports.gym_head}",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if resume:
        command.append("--resume")
    return command


def validate_result(rollouts: Path, *, expected_tasks: int, num_repeats: int) -> dict[str, Any]:
    """Require a complete task x repeat grid and binary rewards."""
    if not rollouts.is_file():
        raise FileNotFoundError(f"Missing rollouts at {rollouts}")

    seen: set[tuple[int, int]] = set()
    rewards: list[float] = []
    error_types: dict[str, int] = {}
    for line in rollouts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seen.add((row["_ng_task_index"], row["_ng_rollout_index"]))
        reward = float(row.get("reward", 0.0))
        rewards.append(reward)
        failure = row.get("_ng_failure_class")
        if failure:
            error_types[failure] = error_types.get(failure, 0) + 1

    expected = {(t, r) for t in range(expected_tasks) for r in range(num_repeats)}
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    non_binary = [value for value in rewards if value not in (0.0, 1.0)]

    summary = {
        "rollouts": len(rewards),
        "expected": len(expected),
        "missing": missing[:20],
        "unexpected": unexpected[:20],
        "non_binary_rewards": non_binary[:20],
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "pass_rate": (sum(1 for value in rewards if value == 1.0) / len(rewards)) if rewards else 0.0,
        "rollout_error_types": error_types,
    }
    if missing or unexpected:
        raise RuntimeError(f"Incomplete rollout grid: {summary}")
    return summary


@contextlib.contextmanager
def phase(name: str, timings: dict[str, float]):
    start = time.monotonic()
    print(f"[tau2-gym] {name}: start", flush=True)
    try:
        yield
    finally:
        timings[name] = round(time.monotonic() - start, 3)
        print(f"[tau2-gym] {name}: {timings[name]}s", flush=True)


def resolve_checkpoint(experiment: Tau2Experiment, override: str | None) -> Path | None:
    if experiment.manager_backend != "local_vllm":
        if override is not None:
            raise SystemExit(f"{experiment.name} has a remote manager; --manager-checkpoint does not apply")
        return None
    checkpoint = Path(override) if override is not None else experiment.manager_checkpoint
    if checkpoint is None:
        raise SystemExit(f"{experiment.name} needs --manager-checkpoint")
    return checkpoint.resolve()


def resolve_output_dir(args: argparse.Namespace) -> tuple[str, Path]:
    """The run name and output directory `execute` uses for these arguments."""
    experiment = get_experiment(args.experiment)
    name = run_name(
        experiment,
        pool=args.pool or experiment.pool,
        tasks_per_domain=args.tasks_per_domain,
        num_repeats=args.num_repeats,
        port_offset=args.port_offset,
    )
    return name, Path(args.output_dir) if args.output_dir else RESULTS_ROOT / name


def execute(args: argparse.Namespace) -> int:
    experiment = get_experiment(args.experiment)
    ports = PortLayout(offset=args.port_offset).shifted()
    if args.subagent_port is not None:
        ports = replace(ports, qwen35_4b=args.subagent_port)
    pool = args.pool or experiment.pool
    _, pool_meta = load_pool(pool)
    checkpoint = resolve_checkpoint(experiment, args.manager_checkpoint)
    concurrency = args.concurrency or experiment.concurrency

    name, output_dir = resolve_output_dir(args)
    logs = output_dir / "logs"
    rollouts = output_dir / "rollouts.jsonl"
    config_path = output_dir / "configuration" / "tau2_gym.yaml"
    tasks_file = Path(args.tasks_file).resolve() if args.tasks_file else None
    dataset = dataset_path(pool, pool_meta["sha256"], args.tasks_per_domain, tasks_file)
    config = gym_config(experiment, ports)

    subagent_model_id = resolve_subagent_model_id(args.subagent_backend, args.subagent_model_id)
    env = base_environment(ports, subagent_backend=args.subagent_backend, subagent_model_id=subagent_model_id)
    env.update(subagent_environment(experiment))
    use_subagent_proxy = args.subagent_backend == "llm_proxy"
    manager_command: list[str] | None = None
    if experiment.manager_backend == "local_vllm":
        assert checkpoint is not None
        manager_command = manager_vllm_command(
            experiment, checkpoint, ports, gpu_memory_utilization=args.manager_gpu_memory_utilization
        )
    elif experiment.manager_backend == "llm_proxy":
        manager_command = remote_proxy_command(
            ports.manager_proxy,
            response_tool_parser="qwen3_xml",
            extra_body=experiment.manager_proxy_extra_body,
        )
    subagent_command = (
        remote_proxy_command(ports.subagent_proxy)
        if use_subagent_proxy
        else subagent_vllm_command(
            ports,
            model_id=subagent_model_id,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    )
    timings: dict[str, float] = {}

    if args.dry:
        print(json.dumps({
            "run_name": name, "output_dir": str(output_dir), "dataset": str(dataset),
            "pool": pool, "pool_sha256": pool_meta["sha256"], "ports": ports.as_dict(),
            "manager_checkpoint": str(checkpoint) if checkpoint else None,
            "manager": manager_command, "manager_gpu": args.manager_gpu,
            "subagent_backend": args.subagent_backend, "subagent_model_id": subagent_model_id,
            "subagent_sampling": subagent_sampling_record(env),
            "subagent": subagent_command,
            "subagent_gpu": args.subagent_gpu,
            "langgraph": langgraph_command(ports, args.langgraph_jobs),
            "gym_start": gym_start_command(ports, config=config_path, logs=logs),
            "gym_eval": gym_eval_command(ports, dataset=dataset, output=rollouts,
                                         num_repeats=args.num_repeats, concurrency=concurrency,
                                         limit=args.limit, resume=args.resume),
            "gym_config": config,
        }, indent=2))
        return 0

    if experiment.manager_backend == "openrouter":
        require_env([OPENROUTER_API_KEY_ENV], "Source ~/.secrets/decomposer.env first.")
    if experiment.manager_backend == "llm_proxy" or use_subagent_proxy:
        require_env([LLM_PROXY_URL_ENV, LLM_PROXY_API_KEY_ENV], "Source ~/.secrets/decomposer.env first.")
        # The local proxies retry upstream failures, so a model the shared proxy does
        # not serve would stall the run instead of failing it: check before starting.
        try:
            require_upstream_models(
                LLM_PROXY_URL_ENV,
                LLM_PROXY_API_KEY_ENV,
                upstream_model_ids(
                    experiment, subagent_backend=args.subagent_backend, subagent_model_id=subagent_model_id
                ),
                verify_tls=False,
            )
        except RuntimeError as error:
            raise SystemExit(f"LLM proxy pre-flight failed: {error}") from None
    if experiment.manager_backend == "local_vllm" and args.manager_gpu is None:
        raise SystemExit("A local manager needs --manager-gpu")
    if not use_subagent_proxy and args.subagent_gpu is None:
        raise SystemExit("--subagent-backend local_vllm needs --subagent-gpu")
    if checkpoint is not None and not checkpoint.is_dir():
        raise SystemExit(f"Manager checkpoint not found: {checkpoint}")

    if output_dir.exists() and not (args.force or args.resume):
        raise SystemExit(f"{output_dir} already exists; pass --force or --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with phase("prepare", timings):
        summary = prepare_dataset(
            pool=pool,
            tasks_per_domain=args.tasks_per_domain,
            tasks_file=tasks_file,
            output=dataset,
            env=env,
        )
        print(f"[tau2-gym] dataset: {json.dumps(summary)}", flush=True)
    expected_tasks = summary["rows"] if args.limit is None else min(args.limit, summary["rows"])

    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_name": name,
        "experiment": asdict(experiment),
        "pool": pool,
        "pool_sha256": pool_meta["sha256"],
        "tasks_per_domain": args.tasks_per_domain,
        "tasks_file": str(tasks_file) if tasks_file else None,
        "num_repeats": args.num_repeats,
        "concurrency": concurrency,
        "limit": args.limit,
        "manager_checkpoint": str(checkpoint) if checkpoint else None,
        "manager_checkpoint_fingerprint": checkpoint_fingerprint(checkpoint) if checkpoint else None,
        "subagent_backend": args.subagent_backend,
        "subagent_model_id": subagent_model_id,
        "subagent_sampling": subagent_sampling_record(env),
        "subagent_endpoint": subagent_base_url(ports, args.subagent_backend),
        "ports": ports.as_dict(),
        "gyms_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()[:16],
        "dataset": str(dataset),
        "dataset_rows": summary["rows"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manager_gpu": args.manager_gpu,
        "subagent_gpu": args.subagent_gpu,
    }
    atomic_json(output_dir / "run_status.json", status)

    supervisor = Supervisor(logs, env)
    try:
        with phase("model_startup", timings):
            require_free_port(ports.langgraph, "LangGraph subagent server")
            require_free_port(ports.gym_head, "Gym head server")
            model_processes: list[subprocess.Popen[Any]] = []
            if experiment.manager_backend == "local_vllm":
                require_free_port(ports.manager_vllm, "manager vLLM")
                model_processes.append(supervisor.start(
                    "vllm_manager", manager_command, cwd=REPO_ROOT,
                    env={"CUDA_VISIBLE_DEVICES": args.manager_gpu},
                ))
            elif experiment.manager_backend == "llm_proxy":
                require_free_port(ports.manager_proxy, "manager LLM proxy")
                model_processes.append(supervisor.start("manager_llm_proxy", manager_command, cwd=REPO_ROOT))
            if use_subagent_proxy:
                # No GPU and no model load: subagents reach the shared replicas
                # through a local credential-isolating proxy.
                require_free_port(ports.subagent_proxy, "subagent LLM proxy")
                model_processes.append(supervisor.start("subagent_llm_proxy", subagent_command, cwd=REPO_ROOT))
            else:
                require_free_port(ports.qwen35_4b, "Qwen3.5-4B subagent vLLM")
                model_processes.append(supervisor.start(
                    "vllm_qwen35_4b", subagent_command, cwd=REPO_ROOT,
                    env={"CUDA_VISIBLE_DEVICES": args.subagent_gpu},
                ))
            if experiment.manager_backend == "local_vllm":
                wait_http(f"{loopback_url(ports.manager_vllm)}/models", model_processes, 1800)
            elif experiment.manager_backend == "llm_proxy":
                wait_http(f"http://127.0.0.1:{ports.manager_proxy}/health", model_processes, 300)
            if use_subagent_proxy:
                wait_http(f"http://127.0.0.1:{ports.subagent_proxy}/health", model_processes, 300)
            else:
                wait_http(f"{loopback_url(ports.qwen35_4b)}/models", model_processes, 1800)

        with phase("langgraph_startup", timings):
            langgraph = supervisor.start(
                "langgraph", langgraph_command(ports, args.langgraph_jobs), cwd=SUBAGENT_DIR
            )
            wait_http(f"http://127.0.0.1:{ports.langgraph}/docs", [*model_processes, langgraph], 300)

        with phase("gym_startup", timings):
            gym = supervisor.start(
                "gym_env", gym_start_command(ports, config=config_path, logs=logs), cwd=GYM_CHECKOUT
            )
            wait_http(
                f"http://127.0.0.1:{ports.gym_head}/server_instances",
                [*model_processes, langgraph, gym],
                900,
            )

        with phase("rollout", timings):
            command = gym_eval_command(
                ports, dataset=dataset, output=rollouts, num_repeats=args.num_repeats,
                concurrency=concurrency, limit=args.limit, resume=args.resume,
            )
            with (logs / "gym_eval.log").open("a") as stream:
                subprocess.run(
                    command, cwd=GYM_CHECKOUT, env=env, check=True,
                    stdout=stream, stderr=subprocess.STDOUT,
                )
    finally:
        supervisor.stop()

    with phase("validation", timings):
        result = validate_result(rollouts, expected_tasks=expected_tasks, num_repeats=args.num_repeats)
        print(f"[tau2-gym] result: {json.dumps(result)}", flush=True)

    status.update(
        {"finished_at": datetime.now(timezone.utc).isoformat(),
         "timings_seconds": timings, "result": result}
    )
    atomic_json(output_dir / "run_status.json", status)
    atomic_json(output_dir / ".eval_done.json", status)
    print(f"[tau2-gym] done: {output_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS_BY_NAME))
    parser.add_argument("--pool", default=None, help="task pool; defaults to the experiment's")
    parser.add_argument("--tasks-per-domain", type=int, default=None,
                        help="deterministic stratified subsample of the pool")
    parser.add_argument("--tasks-file", default=None,
                        help="explicit task subset of the pool: JSON list of [domain, task_id]")
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=None,
                        help="defaults to the experiment's concurrency")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manager-checkpoint", default=None,
                        help="local manager weights; overrides the experiment's checkpoint")
    parser.add_argument("--manager-gpu", default=None, help="CUDA device(s) for a local manager vLLM")
    parser.add_argument("--manager-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--subagent-backend", choices=SUBAGENT_BACKENDS, default="llm_proxy",
                        help="local_vllm starts a dedicated Qwen3.5-4B on --subagent-gpu; "
                             "llm_proxy reaches the shared replicas and needs no GPU.")
    parser.add_argument("--subagent-model-id", default=None,
                        help=f"model the subagents request; default {DEFAULT_SUBAGENT_MODEL_ID} on the proxy, "
                             f"{LOCAL_SUBAGENT_MODEL_ID} on a local vLLM")
    parser.add_argument("--subagent-gpu", default=None, help="CUDA device(s) for a local subagent vLLM")
    parser.add_argument("--subagent-port", type=int, default=None,
                        help="move the subagent vLLM port independently of --port-offset")
    parser.add_argument("--port-offset", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=128000, help="subagent vLLM context")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="subagent vLLM")
    parser.add_argument("--langgraph-jobs", type=int, default=16)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry", "--dry-run", dest="dry", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return execute(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
