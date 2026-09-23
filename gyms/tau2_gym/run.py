"""Local runner for Decomposer evaluations on the tau2 gym.

Starts the Qwen3.5-4B subagent worker, the LangGraph subagent server, the Gym
servers, performs one `gym eval run`, validates the output, and stops every child
process. Structure follows gyms/workplace_assistant/run.py; that code is
duplicated per gym rather than shared, so this is a trimmed copy of the same
phases, port layout, supervisor and status contract.

The tau2 resources server lives at gyms/tau2_gym/gym_components/resources_servers/
and is found through NEMO_GYM_EXTRA_ROOTS, which Gym searches ahead of its own
tree (nemo_gym/__init__.py:52-79). The Gym submodule is untouched.
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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
TAU2_DATA_DIR = TAU2_CHECKOUT / "data"
GYM_EXTRA_ROOT = REPO_ROOT / "gyms" / "tau2_gym" / "gym_components"
SUBAGENT_DIR = REPO_ROOT / "gyms" / "tau2_gym" / "subagents"
GYM_CONFIG = (
    REPO_ROOT
    / "gyms"
    / "tau2_gym"
    / "configs"
    / "tau2_gym_deepseek_v4_flash_0731_qwen35_4b_non_thinking.yaml"
)

QWEN35_4B_MODEL_ID = "Qwen/Qwen3.5-4B"
ROLLOUT_FAILURE_POLICY = "score_zero"

# Shared LLM proxy serving Qwen/Qwen3.5-4B (two replicas). Credentials live in
# ~/.secrets/decomposer.env; gyms.remote_model_proxy injects them so the subagent
# process only ever talks plaintext HTTP to loopback.
LLM_PROXY_URL_ENV = "LLM_PROXY_URL"
LLM_PROXY_API_KEY_ENV = "LLM_PROXY_MASTER_KEY"
SUBAGENT_BACKENDS = ("local_vllm", "llm_proxy")
STATUS_SCHEMA_VERSION = 1

# Defaults for the first tau2 milestone: three GAIA2-hard domains that run with
# ScriptedUser (single compound user turn, no user-simulator LLM) -- see
# _EXPLICIT_AUTO_DOMAINS in external/tau2_gym/training/tau2_env_manager.py:60-68.
DEFAULT_DOMAINS = ("hotel_reservations", "library_lending", "gym_memberships")
DEFAULT_TASKS_PER_DOMAIN = 5


@dataclass(frozen=True)
class PortLayout:
    offset: int = 0
    qwen35_4b: int = 8025
    # 8142 is the workplace gym's manager proxy; keep clear of it.
    subagent_proxy: int = 8143
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
            subagent_proxy=self.subagent_proxy + self.offset,
            langgraph=self.langgraph + self.offset,
            gym_head=self.gym_head + self.offset,
            gym_component_low=self.gym_component_low + self.offset,
            gym_component_high=self.gym_component_high + self.offset,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "offset": self.offset,
            "qwen35_4b": self.qwen35_4b,
            "subagent_proxy": self.subagent_proxy,
            "langgraph": self.langgraph,
            "gym_head": self.gym_head,
            "gym_component_low": self.gym_component_low,
            "gym_component_high": self.gym_component_high,
        }


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


def dataset_filename(domains: Sequence[str], tasks_per_domain: int | None) -> str:
    """Readable for a few domains, hashed once it would blow the 255-byte limit.

    Joining 26 domain names produces a ~450 character filename and ENAMETOOLONG.
    """
    stem = "-".join(domains)
    suffix = f"-{tasks_per_domain}.decomposer.jsonl"
    if len(stem) + len(suffix) <= 120:
        return stem + suffix
    digest = hashlib.sha256("\n".join(domains).encode()).hexdigest()[:12]
    return f"{len(domains)}domains-{digest}{suffix}"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


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
                f"Pass --qwen-port <free port> (model server), or --port-offset N "
                f"(langgraph/gym; note the gym config pins the langgraph URL at 2024)."
            )


def subagent_base_url(ports: PortLayout, backend: str) -> str:
    """Where subagents send their model traffic.

    Both backends look identical to the subagent graph: loopback, `api_key="EMPTY"`.
    For llm_proxy the loopback endpoint is gyms.remote_model_proxy, which injects the
    real credentials and terminates the proxy's self-signed TLS.
    """
    port = ports.subagent_proxy if backend == "llm_proxy" else ports.qwen35_4b
    return f"http://127.0.0.1:{port}/v1"


def base_environment(
    ports: PortLayout, cuda_devices: str | None, *, subagent_backend: str = "local_vllm"
) -> dict[str, str]:
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
                {QWEN35_4B_MODEL_ID: subagent_base_url(ports, subagent_backend)}
            ),
            "UV_CACHE_DIR": str(UV_CACHE),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.setdefault("DECOMPOSER_SUBAGENT_MAX_MODEL_CALLS", "100")
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    return env


def prepare_dataset(
    *, domains: Sequence[str], tasks_per_domain: int | None, output: Path, env: dict[str, str]
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
        "--domains",
        ",".join(domains),
        "--output",
        str(output),
    ]
    if tasks_per_domain is not None:
        command.extend(["--tasks-per-domain", str(tasks_per_domain)])
    result = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        # capture_output hides the child's traceback; without this the caller sees
        # only CalledProcessError and has to re-run the command by hand.
        raise SystemExit(
            "tau2_export failed (exit "
            f"{result.returncode}).\n--- stderr ---\n{result.stderr.strip()[-4000:]}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def vllm_command(ports: PortLayout, *, max_model_len: int, gpu_memory_utilization: float) -> list[str]:
    return [
        str(PROJECT_VENV / "bin" / "vllm"),
        "serve",
        QWEN35_4B_MODEL_ID,
        "--host",
        "127.0.0.1",
        "--port",
        str(ports.qwen35_4b),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        # The manager re-sends a growing context every ReAct turn; without prefix
        # caching total prefill is quadratic in turn count.
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": False}, separators=(",", ":")),
    ]


def require_llm_proxy_credentials() -> None:
    missing = [
        name
        for name in (LLM_PROXY_URL_ENV, LLM_PROXY_API_KEY_ENV)
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(
            f"--subagent-backend llm_proxy needs {' and '.join(missing)} in the "
            f"environment. Source them first:\n  source ~/.secrets/decomposer.env"
        )


def subagent_proxy_command(ports: PortLayout) -> list[str]:
    """Local credential-isolating proxy in front of the shared LLM proxy.

    Deliberately omits two flags:
      --response-tool-parser  its Qwen XML normalisation only runs on the Responses
                              API path (remote_model_proxy.py:317); subagents use Chat
                              Completions, and the upstream already returns structured
                              tool calls, so normalising would be wrong.
      --extra-body-json       it merges server-side and overrides caller keys
                              (remote_model_proxy.py:49-54); leaving it empty keeps the
                              graph's own sampling, so proxy and local runs stay
                              comparable.
    """
    return [
        str(PROJECT_VENV / "bin" / "python"),
        "-m",
        "gyms.remote_model_proxy",
        "--host",
        "127.0.0.1",
        "--port",
        str(ports.subagent_proxy),
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


def gym_start_command(ports: PortLayout, *, prompt_profile: str, logs: Path) -> list[str]:
    prefix = "++decomposer.responses_api_agents.decomposer_agent."
    return [
        str(gym_venv() / "bin" / "gym"),
        "env",
        "start",
        "--config",
        str(GYM_CONFIG),
        f"{prefix}decomposer_system_prompt_profile={prompt_profile}",
        f"{prefix}manager_max_model_calls=100",
        f"{prefix}subagent_recursion_limit=1000",
        "++policy_api_key=${oc.env:OPENROUTER_API_KEY_DECOMPOSER}",
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


def execute(args: argparse.Namespace) -> int:
    ports = PortLayout(offset=args.port_offset).shifted()
    if args.qwen_port is not None:
        ports = replace(ports, qwen35_4b=args.qwen_port)
    domains = tuple(d.strip() for d in args.domains.split(",") if d.strip())

    run_name = f"deepseek-v4-flash-0731-teacher-qwen35-4b-non-thinking-n{args.num_repeats}"
    if args.port_offset:
        run_name += f"-port-offset-{args.port_offset}"
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_ROOT / run_name
    logs = output_dir / "logs"
    rollouts = output_dir / "rollouts.jsonl"
    dataset = DATASETS_ROOT / dataset_filename(domains, args.tasks_per_domain)

    env = base_environment(
        ports, args.cuda_visible_devices, subagent_backend=args.subagent_backend
    )
    use_proxy = args.subagent_backend == "llm_proxy"
    timings: dict[str, float] = {}

    if args.dry:
        print(json.dumps({
            "run_name": run_name, "output_dir": str(output_dir), "dataset": str(dataset),
            "domains": domains, "ports": ports.as_dict(),
            "subagent_backend": args.subagent_backend,
            "subagent_endpoint": subagent_base_url(ports, args.subagent_backend),
            **({"subagent_proxy": subagent_proxy_command(ports)} if use_proxy else
               {"vllm": vllm_command(ports, max_model_len=args.max_model_len,
                                     gpu_memory_utilization=args.gpu_memory_utilization)}),
            "langgraph": langgraph_command(ports, args.langgraph_jobs),
            "gym_start": gym_start_command(ports, prompt_profile=args.prompt_profile, logs=logs),
            "gym_eval": gym_eval_command(ports, dataset=dataset, output=rollouts,
                                         num_repeats=args.num_repeats,
                                         concurrency=args.concurrency, limit=args.limit,
                                         resume=args.resume),
        }, indent=2))
        return 0

    if use_proxy:
        require_llm_proxy_credentials()

    if output_dir.exists() and not (args.force or args.resume):
        raise SystemExit(f"{output_dir} already exists; pass --force or --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    with phase("prepare", timings):
        summary = prepare_dataset(
            domains=domains, tasks_per_domain=args.tasks_per_domain, output=dataset, env=env
        )
        print(f"[tau2-gym] dataset: {summary}", flush=True)
    expected_tasks = summary["rows"] if args.limit is None else min(args.limit, summary["rows"])

    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_name": run_name,
        "domains": list(domains),
        "tasks_per_domain": args.tasks_per_domain,
        "num_repeats": args.num_repeats,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "prompt_profile": args.prompt_profile,
        "subagent_backend": args.subagent_backend,
        "subagent_endpoint": subagent_base_url(ports, args.subagent_backend),
        "ports": ports.as_dict(),
        "dataset": str(dataset),
        "dataset_rows": summary["rows"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": args.cuda_visible_devices,
    }
    atomic_json(output_dir / "run_status.json", status)

    supervisor = Supervisor(logs, env)
    try:
        with phase("model_startup", timings):
            require_free_port(ports.langgraph, "LangGraph subagent server")
            require_free_port(ports.gym_head, "Gym head server")
            if use_proxy:
                # No GPU and no model load: subagents reach the shared replicas
                # through a local credential-isolating proxy.
                require_free_port(ports.subagent_proxy, "subagent LLM proxy")
                proxy = supervisor.start(
                    "subagent_llm_proxy", subagent_proxy_command(ports), cwd=REPO_ROOT
                )
                wait_http(f"http://127.0.0.1:{ports.subagent_proxy}/health", [proxy], 300)
                model_processes = [proxy]
            else:
                require_free_port(ports.qwen35_4b, "Qwen3.5-4B vLLM")
                vllm = supervisor.start(
                    "vllm_qwen35_4b",
                    vllm_command(ports, max_model_len=args.max_model_len,
                                 gpu_memory_utilization=args.gpu_memory_utilization),
                    cwd=REPO_ROOT,
                )
                wait_http(f"http://127.0.0.1:{ports.qwen35_4b}/v1/models", [vllm], 1800)
                model_processes = [vllm]

        with phase("langgraph_startup", timings):
            langgraph = supervisor.start(
                "langgraph", langgraph_command(ports, args.langgraph_jobs), cwd=SUBAGENT_DIR
            )
            wait_http(
                f"http://127.0.0.1:{ports.langgraph}/docs",
                [*model_processes, langgraph],
                300,
            )

        with phase("gym_startup", timings):
            gym = supervisor.start(
                "gym_env", gym_start_command(ports, prompt_profile=args.prompt_profile, logs=logs),
                cwd=REPO_ROOT,
            )
            wait_http(
                f"http://127.0.0.1:{ports.gym_head}/server_instances",
                [*model_processes, langgraph, gym],
                900,
            )

        with phase("rollout", timings):
            command = gym_eval_command(
                ports, dataset=dataset, output=rollouts, num_repeats=args.num_repeats,
                concurrency=args.concurrency, limit=args.limit, resume=args.resume,
            )
            with (logs / "gym_eval.log").open("a") as stream:
                subprocess.run(
                    command, cwd=REPO_ROOT, env=env, check=True,
                    stdout=stream, stderr=subprocess.STDOUT,
                )
    finally:
        supervisor.stop()

    with phase("validation", timings):
        result = validate_result(
            rollouts, expected_tasks=expected_tasks, num_repeats=args.num_repeats
        )
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--tasks-per-domain", type=int, default=DEFAULT_TASKS_PER_DOMAIN)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-profile", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--subagent-backend", choices=SUBAGENT_BACKENDS,
                        default="local_vllm",
                        help="local_vllm starts a dedicated Qwen3.5-4B on a GPU; "
                             "llm_proxy reaches the shared replicas through "
                             "gyms.remote_model_proxy and needs no GPU.")
    parser.add_argument("--port-offset", type=int, default=0)
    parser.add_argument("--qwen-port", type=int, default=None,
                        help="Override the Qwen3.5-4B vLLM port independently of --port-offset. "
                             "The subagent graph reads its endpoint from an env var, so this "
                             "does not require editing the gym config.")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--max-model-len", type=int, default=128000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
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
