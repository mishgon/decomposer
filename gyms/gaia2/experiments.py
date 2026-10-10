"""Experiments-as-code and artifact layout for Gaia2 evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from gyms.gaia2.prompts import Gaia2ManagerPromptAddendumProfile
from gyms.qwen_sampling import (
    QWEN35_UNLOOPED_NON_THINKING,
    qwen35_general_sampling,
    qwen36_non_thinking_sampling,
    qwen36_thinking_sampling,
)

ARTIFACTS_ROOT = Path("/home/sukhorukov/decomposer_artifacts")
PROJECT_ROOT = Path("/home/sukhorukov/decomposer_sft")
PROJECT_VENV = PROJECT_ROOT / ".venv"
DEFAULT_GAIA2_REPO = PROJECT_ROOT / "external" / "gaia2"

DATASET_ID = "meta-agents-research-environments/gaia2"
DATASET_REVISION = "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99"
FILESYSTEM_DATASET_ID = "meta-agents-research-environments/gaia2_filesystem"
FILESYSTEM_DATASET_REVISION = "132e26376f5e963bb59f64bcccdd02188cb08dee"
GAIA2_REVISION = "993389ceb9e789a3965576c4a31a4d74f3cbba5b"
SPLIT = "validation"
SPLIT_MANIFEST_SEED = 42
PARTITIONS = ("train", "test", "full")

Gaia2Domain = Literal["execution", "search", "ambiguity"]


@dataclass(frozen=True)
class Gaia2DomainSpec:
    name: Gaia2Domain
    scenario_count: int
    split_manifest_name: str
    split_manifest_sha256: str
    dataset_aggregate_sha256: str
    train_scenario_count: int
    test_scenario_count: int
    test_universes: tuple[int, ...] = (25, 26, 28)
    supports_trace_generation: bool = False

    @property
    def split_manifest_relpath(self) -> str:
        return f"gyms/gaia2/split_manifests/{self.split_manifest_name}.json"


EXECUTION_DOMAIN = Gaia2DomainSpec(
    name="execution",
    scenario_count=160,
    split_manifest_name="execution-110-50-v1",
    split_manifest_sha256=(
        "79f2725c48afc2c18db9a6266992bb27705ed86d5f8939dbee13964a768c5a20"
    ),
    dataset_aggregate_sha256=(
        "600818deac34268a261cd846ef3019c6a4c87b068dccfea5e006cd72fe112518"
    ),
    train_scenario_count=110,
    test_scenario_count=50,
    supports_trace_generation=True,
)
SEARCH_DOMAIN = Gaia2DomainSpec(
    name="search",
    scenario_count=160,
    split_manifest_name="search-118-42-v1",
    split_manifest_sha256=(
        "2770ae571e37d89f7240232929c4f2bac7ede0dd5c3ce69d9b429edadb8cb105"
    ),
    dataset_aggregate_sha256=(
        "8cc67f8334d72e1dd760685cf4c5e274cd225b5cc63369fdabf1d5e3c84ee6df"
    ),
    train_scenario_count=118,
    test_scenario_count=42,
)
AMBIGUITY_DOMAIN = Gaia2DomainSpec(
    name="ambiguity",
    scenario_count=160,
    split_manifest_name="ambiguity-128-32-v1",
    split_manifest_sha256=(
        "d2020e48d3a375ecd78145ef41a4c455f997681d719bbb7f1e0d9125fc8b7176"
    ),
    dataset_aggregate_sha256=(
        "81a3e9cf8e01a764c61f8cbfdd56e2bf29fa487d32b52280e8a3071e4f5aeb72"
    ),
    train_scenario_count=128,
    test_scenario_count=32,
)
DOMAIN_SPECS: dict[Gaia2Domain, Gaia2DomainSpec] = {
    spec.name: spec
    for spec in (EXECUTION_DOMAIN, SEARCH_DOMAIN, AMBIGUITY_DOMAIN)
}
DOMAINS = tuple(DOMAIN_SPECS)
DOMAIN: Gaia2Domain = "execution"

# Backward-compatible execution aliases. New code should resolve the selected
# domain through ``get_domain_spec`` instead of reading these constants.
SCENARIO_COUNT = EXECUTION_DOMAIN.scenario_count
SPLIT_MANIFEST_NAME = EXECUTION_DOMAIN.split_manifest_name
TRAIN_SCENARIO_COUNT = EXECUTION_DOMAIN.train_scenario_count
TEST_SCENARIO_COUNT = EXECUTION_DOMAIN.test_scenario_count


def get_domain_spec(domain: str) -> Gaia2DomainSpec:
    try:
        return DOMAIN_SPECS[domain]  # type: ignore[index]
    except KeyError as error:
        expected = ", ".join(DOMAINS)
        raise ValueError(
            f"Unknown Gaia2 domain {domain!r}; expected one of: {expected}"
        ) from error

DATA_ROOT = ARTIFACTS_ROOT / "evaluation" / "data" / "gaia2"
RESULTS_ROOT = ARTIFACTS_ROOT / "evaluation" / "gaia2" / "results"
TRACES_ROOT = ARTIFACTS_ROOT / "evaluation" / "gaia2" / "traces"
PARTITION_DATA_ROOT = DATA_ROOT / "partitions" / SPLIT_MANIFEST_NAME
PREPARATION_MANIFEST_ROOT = DATA_ROOT / "manifests" / SPLIT / DOMAIN
DECOMPOSER_STAGING_ROOT = ARTIFACTS_ROOT / "code" / "decomposer"
GAIA2_STAGING_ROOT = ARTIFACTS_ROOT / "code" / "gaia2"
GAIA2_VENV_ROOT = ARTIFACTS_ROOT / "venvs" / "gaia2"
UV_CACHE = ARTIFACTS_ROOT / "cache" / "uv"
UV_BIN = ARTIFACTS_ROOT / "tools" / "uv"
HF_HOME = Path("/home/sukhorukov/.cache/huggingface")

BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.42"
INSTANCE_TYPES_BY_NUM_GPUS = {
    1: "a100plus.1gpu.80vG.12C.182G",
    2: "a100plus.2gpu.80vG.24C.364G",
}

JUDGE_MODEL = "openai/nvidia/Llama-3.3-70B-Instruct-FP8"
SCENARIO_TIMEOUT_SECONDS = 3600

GEMMA4_E4B_BASE = (
    HF_HOME
    / "hub"
    / "models--google--gemma-4-E4B-it"
    / "snapshots"
    / "ee0ef6023621cff504d758262d4e04895a5af4a2"
)
GEMMA4_E2B_BASE = (
    HF_HOME
    / "hub"
    / "models--google--gemma-4-E2B-it"
    / "snapshots"
    / "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
)
GEMMA4_31B_BASE = (
    HF_HOME
    / "hub"
    / "models--google--gemma-4-31B-it"
    / "snapshots"
    / "842da3794eaa0b77d5f08bae87a17459d91ff475"
)
GEMMA4_26B_A4B_BASE = (
    HF_HOME
    / "hub"
    / "models--google--gemma-4-26B-A4B-it"
    / "snapshots"
    / "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
)
GEMMA4_E4B_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-full-4gpu"
    / "final-vllm"
)
QWEN35_4B_BASE = (
    HF_HOME
    / "hub"
    / "models--Qwen--Qwen3.5-4B"
    / "snapshots"
    / "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
QWEN35_2B_BASE = (
    HF_HOME
    / "hub"
    / "models--Qwen--Qwen3.5-2B"
    / "snapshots"
    / "15852e8c16360a2fea060d615a32b45270f8a8fc"
)
QWEN35_9B_BASE = (
    HF_HOME
    / "hub"
    / "models--Qwen--Qwen3.5-9B"
    / "snapshots"
    / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
QWEN35_4B_SFT_SERVED_NAME = "decomposer/qwen35-4b-sft-workplace-v1-3765-32k"
QWEN35_4B_BASE_MANAGER_SERVED_NAME = "decomposer/qwen35-4b-base-manager"
QWEN35_4B_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "qwen35-4b-nonthinking-workplace-v1-3765-32k-full-4gpu"
    / "final"
)
QWEN35_4B_MIXED_SFT_SERVED_NAME = (
    "decomposer/qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k"
)
QWEN35_4B_MIXED_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "qwen35-4b-nonthinking-mixed-v1-partial-3983f605-327-32k-full-4gpu"
    / "final"
)
QWEN35_4B_FINAL_MIXED_SFT_SERVED_NAME = (
    "decomposer/qwen35-4b-sft-mixed-v1-final-493c24c4-404"
)
QWEN35_4B_FINAL_MIXED_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "qwen35-4b-nonthinking-mixed-v1-final-493c24c4-404-32k-full-4gpu"
    / "final"
)
QWEN35_4B_FILTERED_SFT_SERVED_NAME = (
    "decomposer/qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279"
)
QWEN35_4B_FILTERED_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / (
        "qwen35-4b-nonthinking-mixed-v1-final-493c24c4-404-"
        "filtered-pass-qgt90-32k-full-4gpu-patience1-stopped-step558"
    )
    / "final-patience1-best-step279"
)
QWEN35_4B_GAIA2_SFT_SERVED_NAME = (
    "decomposer/qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2"
)
QWEN35_4B_GAIA2_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "qwen35-4b-nonthinking-mixed-v2-493c24c4-gaia2-110-n3-filtered-32k-full-4gpu"
    / "final"
)
QWEN35_4B_TOOLATHLON_ONLY_SFT_SERVED_NAME = (
    "decomposer/qwen35-4b-sft-toolathlon-only-v1-493c24c4-"
    "teacher-prompt-filtered-32k"
)
QWEN35_4B_TOOLATHLON_ONLY_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / (
        "qwen35-4b-nonthinking-toolathlon-only-v1-493c24c4-teacher-prompt-"
        "filtered-32k-hf-fa2-fla-b8-e8-full-4gpu"
    )
    / "final"
)
QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_SERVED_NAME = (
    "decomposer/qwen35-4b-sft-gaia2-execution-only-v1-110-n10-"
    "teacher-prompt-r1-balanced-32k"
)
QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / (
        "qwen35-4b-nonthinking-gaia2-execution-only-v1-110-n10-teacher-"
        "prompt-r1-balanced-32k-hf-fa2-fla-b8-e24-full-4gpu"
    )
    / "final"
)

DecomposerManagerBackend = Literal["local_vllm", "openrouter", "llm_proxy"]
# A worker is either a local vLLM or a model on the shared LLM proxy, reached
# through a loopback proxy (gyms/remote_model_proxy.py) as tau2's subagents are.
DecomposerWorkerBackend = Literal["local_vllm", "llm_proxy"]
# `reasoning.effort` for an llm_proxy manager on the Responses API, where the proxy
# ignores chat_template_kwargs (see gyms/tau2_gym/experiments.py).
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]
DecomposerPromptProfile = Literal["student", "teacher"]
# "are_native": workers get ARE's native agent system prompt for the scenario
# (gyms/gaia2/worker_prompt.py); "legacy": the worker text used before it.
WorkerSystemPrompt = Literal["are_native", "legacy"]
# "frozen_turn": ARE's clock is frozen during each Decomposer turn and moves only
# through tool calls, so generation is free as for ARE's native agent
# (gyms/gaia2/simulated_time.py); "wall_clock": the clock runs in real time, as
# in runs made before the setting existed.
SimulatedTime = Literal["frozen_turn", "wall_clock"]
# Domains whose scenarios schedule nothing during a turn (checked per scenario by
# gyms/gaia2/simulated_time.py:frozen_turn_refusals); Ambiguity has several turns.
FROZEN_TURN_DOMAINS: tuple[Gaia2Domain, ...] = ("execution", "search")
SimpleAgentBackend = Literal["local_vllm", "openrouter", "llm_proxy"]
Purpose = Literal["evaluation", "trace-generation"]
Partition = Literal["train", "test", "full"]


def gaia2_lock_hash(gaia2_root: Path) -> str:
    return hashlib.sha256((gaia2_root / "uv.lock").read_bytes()).hexdigest()[:16]


def gaia2_venv(gaia2_root: Path) -> Path:
    return GAIA2_VENV_ROOT / gaia2_lock_hash(gaia2_root)


def dataset_revision_root(domain: Gaia2Domain = DOMAIN) -> Path:
    """Return the immutable source root for one domain.

    Execution keeps its historical layout. Additional domains use an isolated
    subtree so preparing them cannot rewrite the completed execution source.
    """

    spec = get_domain_spec(domain)
    root = DATA_ROOT / DATASET_REVISION
    return root if spec.name == DOMAIN else root / "domains" / spec.name


def dataset_root(domain: Gaia2Domain = DOMAIN) -> Path:
    """Root passed to ARE with the selected ``--config``."""

    return dataset_revision_root(domain) / SPLIT


def scenario_dir(domain: Gaia2Domain = DOMAIN) -> Path:
    return dataset_root(domain) / domain


def dataset_manifest(domain: Gaia2Domain = DOMAIN) -> Path:
    return dataset_revision_root(domain) / "dataset_manifest.json"


def partition_data_root(domain: Gaia2Domain = DOMAIN) -> Path:
    return DATA_ROOT / "partitions" / get_domain_spec(domain).split_manifest_name


def partition_dataset_root(
    partition: Partition, domain: Gaia2Domain = DOMAIN
) -> Path:
    if partition == "full":
        return dataset_root(domain)
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown Gaia2 partition: {partition!r}")
    return partition_data_root(domain) / partition


def filesystem_revision_root() -> Path:
    return DATA_ROOT / "filesystem" / FILESYSTEM_DATASET_REVISION


def filesystem_dir() -> Path:
    return filesystem_revision_root() / "demo_filesystem"


def filesystem_manifest() -> Path:
    return filesystem_revision_root() / "filesystem_manifest.json"


@dataclass(frozen=True)
class WorkerSampling:
    """Subagent sampling for a worker from a different family than the manager.

    GAIA2 otherwise shares one sampling block between the two actors, which only
    holds while both are the same model family. The penalty fields stay optional
    so an override can leave them unset exactly as the shared fields do.
    """

    temperature: float
    top_p: float
    top_k: int
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    # Caps the worker's completions alone; None inherits max_completion_tokens.
    max_completion_tokens: int | None = None


@dataclass(frozen=True)
class DecomposerExperiment:
    name: str
    # None only for a worker served by the LLM proxy.
    worker_checkpoint: Path | None
    manager_checkpoint: Path | None = None
    manager_backend: DecomposerManagerBackend = "local_vllm"
    manager_upstream_url_env: str | None = None
    manager_api_key_env: str | None = None
    manager_response_tool_parser: str | None = None
    manager_reasoning_mode: Literal[
        "service_default", "non_thinking", "thinking"
    ] | None = None
    manager_verify_tls: bool = True
    manager_reasoning_effort: ReasoningEffort | None = None
    worker_backend: DecomposerWorkerBackend = "local_vllm"
    worker_upstream_url_env: str | None = None
    worker_api_key_env: str | None = None
    worker_verify_tls: bool = True
    prompt_profile: DecomposerPromptProfile = "student"
    manager_prompt_addendum_profile: Gaia2ManagerPromptAddendumProfile | None = None
    worker_system_prompt: WorkerSystemPrompt = "are_native"
    simulated_time: SimulatedTime = "frozen_turn"
    num_gpus: int = 2
    manager_served_name: str = "decomposer/gemma4-e4b-sft-deepseek-e4b-v1-8k"
    worker_served_name: str = "google/gemma-4-E4B-it"
    manager_port: int = 8020
    worker_port: int = 8021
    service_port: int = 8124
    subagent_port: int = 2024
    max_model_len: int = 65536
    max_num_seqs: int = 64
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    # Overrides the sampling block above for the subagent alone; None inherits it.
    worker_sampling: WorkerSampling | None = None
    gpu_memory_utilization: float = 0.90
    concurrency: int = 4
    manager_parallel_tool_calls: bool = False
    manager_thinking: bool = False
    manager_tool_call_parser: str = "gemma4"
    manager_reasoning_parser: str | None = "gemma4"
    manager_language_model_only: bool = True
    manager_trust_remote_code: bool = False
    manager_gdn_prefill_backend: str | None = None
    worker_thinking: bool = True
    worker_tool_call_parser: str = "gemma4"
    worker_reasoning_parser: str | None = "gemma4"
    worker_language_model_only: bool = True
    worker_trust_remote_code: bool = False
    worker_gdn_prefill_backend: str | None = None
    share_local_vllm: bool = False
    manager_max_model_calls: int | None = 80
    subagent_max_model_calls: int | None = 80
    manager_recursion_limit: int = 200
    subagent_recursion_limit: int = 200
    kind: Literal["decomposer"] = field(init=False, default="decomposer")

    def __post_init__(self) -> None:
        if self.max_completion_tokens is not None and self.max_completion_tokens < 1:
            raise ValueError(
                f"{self.name}: max_completion_tokens must be at least 1"
            )
        for field_name in (
            "manager_max_model_calls",
            "subagent_max_model_calls",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{self.name}: {field_name} must be at least 1")
        for field_name in ("manager_recursion_limit", "subagent_recursion_limit"):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{self.name}: {field_name} must be at least 1")
        if self.manager_backend == "local_vllm" and self.manager_checkpoint is None:
            raise ValueError("A local_vllm manager requires manager_checkpoint")
        if (
            self.worker_sampling is not None
            and self.worker_sampling.max_completion_tokens is not None
            and self.worker_sampling.max_completion_tokens < 1
        ):
            raise ValueError(
                f"{self.name}: worker max_completion_tokens must be at least 1"
            )
        remote_worker_fields = (self.worker_upstream_url_env, self.worker_api_key_env)
        if self.worker_backend == "local_vllm":
            if self.worker_checkpoint is None:
                raise ValueError(f"{self.name}: a local_vllm worker requires worker_checkpoint")
            if any(value is not None for value in remote_worker_fields):
                raise ValueError(
                    f"{self.name}: remote worker fields require worker_backend=llm_proxy"
                )
        else:
            if self.worker_checkpoint is not None:
                raise ValueError(f"{self.name}: an llm_proxy worker has no checkpoint")
            if any(value is None for value in remote_worker_fields):
                raise ValueError(
                    f"{self.name}: llm_proxy requires complete remote worker fields"
                )
        if self.share_local_vllm:
            if self.worker_backend != "local_vllm":
                raise ValueError(
                    f"{self.name}: share_local_vllm requires a local_vllm worker"
                )
            if self.manager_backend != "local_vllm":
                raise ValueError(
                    f"{self.name}: share_local_vllm requires a local_vllm manager"
                )
            if self.manager_checkpoint != self.worker_checkpoint:
                raise ValueError(
                    f"{self.name}: a shared manager/worker server requires one checkpoint"
                )
            if self.manager_served_name != self.worker_served_name:
                raise ValueError(
                    f"{self.name}: a shared manager/worker server requires one served name"
                )
            if self.manager_port != self.worker_port:
                raise ValueError(
                    f"{self.name}: a shared manager/worker server requires one port"
                )
            server_pairs = (
                (
                    "tool_call_parser",
                    self.manager_tool_call_parser,
                    self.worker_tool_call_parser,
                ),
                (
                    "reasoning_parser",
                    self.manager_reasoning_parser,
                    self.worker_reasoning_parser,
                ),
                (
                    "language_model_only",
                    self.manager_language_model_only,
                    self.worker_language_model_only,
                ),
                (
                    "trust_remote_code",
                    self.manager_trust_remote_code,
                    self.worker_trust_remote_code,
                ),
                (
                    "gdn_prefill_backend",
                    self.manager_gdn_prefill_backend,
                    self.worker_gdn_prefill_backend,
                ),
            )
            mismatched = [
                name
                for name, manager, worker in server_pairs
                if manager != worker
            ]
            if mismatched:
                raise ValueError(
                    f"{self.name}: shared manager/worker server options differ: "
                    + ", ".join(mismatched)
                )
        # One GPU per local vLLM server: a dedicated manager, a worker, or both
        # sharing one server.
        expected_gpus = (
            int(self.requires_local_manager)
            + int(self.requires_local_worker)
            - int(self.share_local_vllm)
        )
        if self.num_gpus != expected_gpus:
            raise ValueError(
                f"{self.manager_backend} Decomposer with a {self.worker_backend} "
                f"worker requires {expected_gpus} GPU(s)"
            )
        remote_fields = (
            self.manager_upstream_url_env,
            self.manager_api_key_env,
            self.manager_response_tool_parser,
            self.manager_reasoning_mode,
        )
        if self.manager_backend == "llm_proxy":
            if any(value is None for value in remote_fields):
                raise ValueError(
                    f"{self.name}: llm_proxy requires complete remote manager fields"
                )
        elif any(value is not None for value in remote_fields):
            raise ValueError(
                f"{self.name}: remote manager fields require manager_backend=llm_proxy"
            )
        if self.manager_reasoning_effort is not None:
            # "none" is how a non-thinking manager stops thinking on the Responses
            # API, which ignores chat_template_kwargs; the other efforts need thinking.
            mode = (
                "non_thinking" if self.manager_reasoning_effort == "none" else "thinking"
            )
            if (
                self.manager_backend != "llm_proxy"
                or self.manager_reasoning_mode != mode
            ):
                raise ValueError(
                    f"{self.name}: manager_reasoning_effort="
                    f"{self.manager_reasoning_effort!r} requires a {mode} "
                    "llm_proxy manager"
                )

    @property
    def requires_local_manager(self) -> bool:
        return self.manager_backend == "local_vllm"

    @property
    def requires_local_worker(self) -> bool:
        return self.worker_backend == "local_vllm"

    @property
    def uses_llm_proxy_models(self) -> bool:
        """Whether the manager or the worker is served by the LLM proxy."""

        return self.requires_llm_proxy or self.worker_backend == "llm_proxy"

    @property
    def remote_manager_record(self) -> dict[str, object]:
        """How a remote manager is identified in preparation manifests."""

        record: dict[str, object] = {
            "backend": self.manager_backend,
            "model": self.manager_served_name,
        }
        if self.requires_llm_proxy:
            record.update(
                {
                    "upstream_url_env": self.manager_upstream_url_env,
                    "api_key_env": self.manager_api_key_env,
                    "response_tool_parser": self.manager_response_tool_parser,
                    "reasoning_mode": self.manager_reasoning_mode,
                    "verify_tls": self.manager_verify_tls,
                }
            )
            if self.manager_reasoning_effort is not None:
                record["reasoning_effort"] = self.manager_reasoning_effort
        return record

    @property
    def remote_worker_record(self) -> dict[str, object]:
        """How an LLM-proxy worker is identified in preparation manifests."""

        return {
            "backend": self.worker_backend,
            "model": self.worker_served_name,
            "upstream_url_env": self.worker_upstream_url_env,
            "api_key_env": self.worker_api_key_env,
            "verify_tls": self.worker_verify_tls,
        }

    @property
    def worker_max_completion_tokens(self) -> int | None:
        if (
            self.worker_sampling is not None
            and self.worker_sampling.max_completion_tokens is not None
        ):
            return self.worker_sampling.max_completion_tokens
        return self.max_completion_tokens

    @property
    def requires_openrouter(self) -> bool:
        return self.manager_backend == "openrouter"

    @property
    def requires_llm_proxy(self) -> bool:
        return self.manager_backend == "llm_proxy"

    @property
    def requires_remote_manager(self) -> bool:
        return self.manager_backend != "local_vllm"

    @property
    def remote_manager_extra_body(self) -> dict[str, object]:
        if self.manager_reasoning_mode == "service_default":
            return {}
        if self.manager_reasoning_mode not in ("non_thinking", "thinking"):
            return {}
        body: dict[str, object] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "include_reasoning": self.manager_reasoning_mode == "thinking",
            "chat_template_kwargs": {
                "enable_thinking": self.manager_reasoning_mode == "thinking",
                **(
                    {"preserve_thinking": True}
                    if self.manager_reasoning_mode == "thinking"
                    else {}
                ),
            },
        }
        if self.manager_reasoning_effort is not None:
            body["reasoning"] = {"effort": self.manager_reasoning_effort}
        if self.max_completion_tokens is not None:
            body["max_output_tokens"] = self.max_completion_tokens
        return body


@dataclass(frozen=True)
class SimpleExperiment:
    name: str
    checkpoint: Path | None
    backend: SimpleAgentBackend = "local_vllm"
    num_gpus: int = 1
    served_name: str = "google/gemma-4-E4B-it"
    port: int = 8100
    base_url: str | None = None
    # For llm_proxy: the environment variable that holds the proxy URL.
    upstream_url_env: str | None = None
    api_key_env: str | None = None
    verify_tls: bool = True
    reasoning_effort: str | None = None
    max_model_len: int = 65536
    max_num_seqs: int = 64
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = 64
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    gpu_memory_utilization: float = 0.90
    concurrency: int = 4
    thinking: bool = True
    tool_call_parser: str = "gemma4"
    reasoning_parser: str | None = "gemma4"
    language_model_only: bool = True
    trust_remote_code: bool = False
    gdn_prefill_backend: str | None = None
    max_model_calls: int = 80
    kind: Literal["simple"] = field(init=False, default="simple")

    def __post_init__(self) -> None:
        if self.max_completion_tokens is not None and self.max_completion_tokens < 1:
            raise ValueError(
                f"{self.name}: max_completion_tokens must be at least 1"
            )
        if self.max_model_calls < 1:
            raise ValueError(f"{self.name}: max_model_calls must be at least 1")
        if self.backend == "local_vllm":
            if self.checkpoint is None:
                raise ValueError(f"{self.name}: local_vllm requires checkpoint")
            if self.num_gpus != 1:
                raise ValueError(f"{self.name}: local_vllm requires one GPU")
            if any(
                value is not None
                for value in (self.base_url, self.upstream_url_env, self.api_key_env)
            ):
                raise ValueError(
                    f"{self.name}: local_vllm cannot configure remote model fields"
                )
        else:
            if self.checkpoint is not None:
                raise ValueError(f"{self.name}: {self.backend} cannot use checkpoint")
            if self.num_gpus != 0:
                raise ValueError(
                    f"{self.name}: {self.backend} simple agent uses no GPU"
                )
            if self.backend == "openrouter" and (
                not self.base_url or not self.api_key_env or self.upstream_url_env
            ):
                raise ValueError(
                    f"{self.name}: openrouter requires base_url and api_key_env"
                )
            if self.backend == "llm_proxy" and (
                not self.upstream_url_env or not self.api_key_env or self.base_url
            ):
                raise ValueError(
                    f"{self.name}: llm_proxy requires upstream_url_env and api_key_env"
                )
            if self.backend == "llm_proxy" and self.reasoning_effort is not None:
                raise ValueError(
                    f"{self.name}: llm_proxy sampling comes from ARE_SAMPLING_PARAMS"
                )

    @property
    def requires_openrouter(self) -> bool:
        return self.backend == "openrouter"

    @property
    def requires_llm_proxy(self) -> bool:
        return self.backend == "llm_proxy"

    @property
    def requires_local_model(self) -> bool:
        return self.backend == "local_vllm"

    @property
    def remote_policy_record(self) -> dict[str, object]:
        """How a remote policy is identified in preparation manifests."""

        record: dict[str, object] = {
            "backend": self.backend,
            "model": self.served_name,
        }
        if self.requires_llm_proxy:
            record.update(
                {
                    "upstream_url_env": self.upstream_url_env,
                    "api_key_env": self.api_key_env,
                    "verify_tls": self.verify_tls,
                }
            )
        return record

    @property
    def remote_extra_body(self) -> dict[str, object]:
        if self.reasoning_effort is None:
            return {}
        return {"reasoning": {"effort": self.reasoning_effort}}


Experiment = DecomposerExperiment | SimpleExperiment

DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name=("gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking"),
    worker_checkpoint=GEMMA4_E4B_BASE,
    manager_checkpoint=GEMMA4_E4B_SFT,
)
DEEPSEEK_GEMMA_EXPERIMENT = DecomposerExperiment(
    name="deepseek-v4-flash-0731-teacher-gemma4-e4b-thinking",
    worker_checkpoint=GEMMA4_E4B_BASE,
    manager_backend="openrouter",
    prompt_profile="teacher",
    num_gpus=1,
    manager_served_name="deepseek/deepseek-v4-flash-0731",
    manager_thinking=True,
)
_QWEN35_NON_THINKING_SAMPLING = qwen35_general_sampling(thinking=False)
QWEN35_SFT_EXPERIMENT = DecomposerExperiment(
    name=("qwen35-4b-sft-workplace-v1-3765-32k-non-thinking-" "qwen35-4b-non-thinking"),
    worker_checkpoint=QWEN35_4B_BASE,
    manager_checkpoint=QWEN35_4B_SFT,
    manager_served_name=QWEN35_4B_SFT_SERVED_NAME,
    worker_served_name="Qwen/Qwen3.5-4B",
    manager_port=8026,
    worker_port=8025,
    service_port=8126,
    subagent_port=2026,
    max_model_len=131072,
    temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_gdn_prefill_backend="triton",
    worker_thinking=False,
    worker_tool_call_parser="qwen3_xml",
    worker_reasoning_parser=None,
    worker_language_model_only=False,
    worker_trust_remote_code=True,
    worker_gdn_prefill_backend="triton",
)
QWEN35_MIXED_SFT_EXPERIMENT = DecomposerExperiment(
    name=(
        "qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-"
        "qwen35-4b-non-thinking"
    ),
    worker_checkpoint=QWEN35_4B_BASE,
    manager_checkpoint=QWEN35_4B_MIXED_SFT,
    manager_served_name=QWEN35_4B_MIXED_SFT_SERVED_NAME,
    worker_served_name="Qwen/Qwen3.5-4B",
    manager_port=8026,
    worker_port=8025,
    service_port=8126,
    subagent_port=2026,
    max_model_len=131072,
    temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_gdn_prefill_backend="triton",
    worker_thinking=False,
    worker_tool_call_parser="qwen3_xml",
    worker_reasoning_parser=None,
    worker_language_model_only=False,
    worker_trust_remote_code=True,
    worker_gdn_prefill_backend="triton",
)
QWEN35_FILTERED_SFT_EXPERIMENT = DecomposerExperiment(
    name=(
        "qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-"
        "non-thinking-qwen35-4b-non-thinking"
    ),
    worker_checkpoint=QWEN35_4B_BASE,
    manager_checkpoint=QWEN35_4B_FILTERED_SFT,
    manager_served_name=QWEN35_4B_FILTERED_SFT_SERVED_NAME,
    worker_served_name="Qwen/Qwen3.5-4B",
    manager_port=8026,
    worker_port=8025,
    service_port=8126,
    subagent_port=2026,
    max_model_len=131072,
    temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_gdn_prefill_backend="triton",
    worker_thinking=False,
    worker_tool_call_parser="qwen3_xml",
    worker_reasoning_parser=None,
    worker_language_model_only=False,
    worker_trust_remote_code=True,
    worker_gdn_prefill_backend="triton",
)
QWEN35_FINAL_MIXED_SFT_EXPERIMENT = DecomposerExperiment(
    name=(
        "qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-"
        "qwen35-4b-non-thinking"
    ),
    worker_checkpoint=QWEN35_4B_BASE,
    manager_checkpoint=QWEN35_4B_FINAL_MIXED_SFT,
    manager_served_name=QWEN35_4B_FINAL_MIXED_SFT_SERVED_NAME,
    worker_served_name="Qwen/Qwen3.5-4B",
    manager_port=8026,
    worker_port=8025,
    service_port=8126,
    subagent_port=2026,
    max_model_len=131072,
    temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_gdn_prefill_backend="triton",
    worker_thinking=False,
    worker_tool_call_parser="qwen3_xml",
    worker_reasoning_parser=None,
    worker_language_model_only=False,
    worker_trust_remote_code=True,
    worker_gdn_prefill_backend="triton",
)
QWEN35_GAIA2_SFT_EXPERIMENT = replace(
    QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
    name=(
        "qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-"
        "non-thinking-qwen35-4b-non-thinking"
    ),
    manager_checkpoint=QWEN35_4B_GAIA2_SFT,
    manager_served_name=QWEN35_4B_GAIA2_SFT_SERVED_NAME,
)
QWEN35_TOOLATHLON_ONLY_SFT_EXPERIMENT = replace(
    QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
    name=(
        "qwen35-4b-sft-toolathlon-only-v1-493c24c4-teacher-prompt-"
        "filtered-32k-non-thinking-qwen35-4b-non-thinking"
    ),
    manager_checkpoint=QWEN35_4B_TOOLATHLON_ONLY_SFT,
    manager_served_name=QWEN35_4B_TOOLATHLON_ONLY_SFT_SERVED_NAME,
)
QWEN35_GAIA2_EXECUTION_ONLY_SFT_EXPERIMENT = replace(
    QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
    name=(
        "qwen35-4b-sft-gaia2-execution-only-v1-110-n10-teacher-prompt-"
        "r1-balanced-32k-non-thinking-qwen35-4b-non-thinking"
    ),
    manager_checkpoint=QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT,
    manager_served_name=QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_SERVED_NAME,
)
QWEN35_BASE_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name="qwen35-4b-base-non-thinking-qwen35-4b-non-thinking",
    worker_checkpoint=QWEN35_4B_BASE,
    manager_checkpoint=QWEN35_4B_BASE,
    manager_served_name=QWEN35_4B_BASE_MANAGER_SERVED_NAME,
    worker_served_name="Qwen/Qwen3.5-4B",
    manager_port=8028,
    worker_port=8027,
    service_port=8128,
    subagent_port=2028,
    max_model_len=131072,
    temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_language_model_only=False,
    manager_trust_remote_code=True,
    manager_gdn_prefill_backend="triton",
    worker_thinking=False,
    worker_tool_call_parser="qwen3_xml",
    worker_reasoning_parser=None,
    worker_language_model_only=False,
    worker_trust_remote_code=True,
    worker_gdn_prefill_backend="triton",
)
QWEN35_BASE_TEACHER_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_BASE_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-base-non-thinking-teacher-qwen35-4b-non-thinking",
    prompt_profile="teacher",
)
DEEPSEEK_QWEN_EXPERIMENT = DecomposerExperiment(
    name="deepseek-v4-flash-0731-teacher-qwen35-4b-non-thinking",
    worker_checkpoint=QWEN35_4B_BASE,
    manager_backend="openrouter",
    prompt_profile="teacher",
    num_gpus=1,
    manager_served_name="deepseek/deepseek-v4-flash-0731",
    manager_thinking=True,
    worker_served_name="Qwen/Qwen3.5-4B",
    worker_port=8031,
    service_port=8134,
    subagent_port=2034,
    max_model_len=131072,
    temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
    worker_thinking=False,
    worker_tool_call_parser="qwen3_xml",
    worker_reasoning_parser=None,
    worker_language_model_only=False,
    worker_trust_remote_code=True,
    worker_gdn_prefill_backend="triton",
)
DEEPSEEK_QWEN_AMBIGUITY_POLICY_EXPERIMENT = replace(
    DEEPSEEK_QWEN_EXPERIMENT,
    name=(
        "deepseek-v4-flash-0731-teacher-gaia2-ambiguity-policy-"
        "qwen35-4b-non-thinking"
    ),
    manager_prompt_addendum_profile="gaia2-ambiguity",
    # Ambiguity scenarios have several turns, which frozen_turn does not support.
    simulated_time="wall_clock",
)
# Known upstream fault for every Qwen3.6 llm_proxy profile below: the Responses
# deployment returns reasoning, but discards reasoning items that the harness
# sends back in later inputs. We save output reasoning, yet these experiments
# remain capture-only and are not clean comparisons with replayed DeepSeek/Gemma.
QWEN36_QWEN_EXPERIMENT = replace(
    DEEPSEEK_QWEN_EXPERIMENT,
    name="qwen36-35b-a3b-teacher-qwen35-4b-non-thinking",
    manager_backend="llm_proxy",
    manager_served_name="Qwen/Qwen3.6-35B-A3B-FP8",
    manager_port=8142,
    manager_upstream_url_env="LLM_PROXY_URL",
    manager_api_key_env="LLM_PROXY_MASTER_KEY",
    manager_response_tool_parser="qwen3_xml",
    manager_reasoning_mode="service_default",
    manager_verify_tls=False,
    concurrency=16,
)
GEMMA4_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name="gemma4-26b-a4b-thinking-gemma4-e4b-thinking-text-defaults",
    worker_checkpoint=GEMMA4_E4B_BASE,
    manager_checkpoint=GEMMA4_26B_A4B_BASE,
    manager_served_name="google/gemma-4-26B-A4B-it",
    worker_served_name="google/gemma-4-E4B-it",
    manager_port=8023,
    worker_port=8021,
    service_port=8127,
    subagent_port=2027,
    max_model_len=131072,
    manager_thinking=True,
    worker_thinking=True,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
_QWEN36_NON_THINKING_SAMPLING = qwen36_non_thinking_sampling()
_QWEN36_THINKING_SAMPLING = qwen36_thinking_sampling()
# Capture-only upstream fault; see the Qwen3.6 llm_proxy warning above.
QWEN36_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT = replace(
    DEEPSEEK_QWEN_EXPERIMENT,
    name=(
        "qwen36-35b-a3b-non-thinking-teacher-"
        "qwen35-4b-non-thinking-text-defaults"
    ),
    manager_backend="llm_proxy",
    manager_served_name="Qwen/Qwen3.6-35B-A3B-FP8",
    manager_port=8142,
    manager_upstream_url_env="LLM_PROXY_URL",
    manager_api_key_env="LLM_PROXY_MASTER_KEY",
    manager_response_tool_parser="qwen3_xml",
    manager_reasoning_mode="non_thinking",
    manager_verify_tls=False,
    prompt_profile="teacher",
    concurrency=16,
    max_model_len=131072,
    temperature=_QWEN36_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN36_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN36_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN36_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN36_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN36_NON_THINKING_SAMPLING.repetition_penalty,
    manager_thinking=False,
    worker_thinking=False,
    worker_language_model_only=True,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
# Capture-only upstream fault; replayed reasoning is discarded by the proxy
# deployment even though this thinking profile captures and sends it.
QWEN36_THINKING_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT = replace(
    DEEPSEEK_QWEN_EXPERIMENT,
    name=(
        "qwen36-35b-a3b-thinking-teacher-"
        "qwen35-4b-non-thinking-text-defaults"
    ),
    manager_backend="llm_proxy",
    manager_served_name="Qwen/Qwen3.6-35B-A3B-FP8",
    manager_port=8142,
    manager_upstream_url_env="LLM_PROXY_URL",
    manager_api_key_env="LLM_PROXY_MASTER_KEY",
    manager_response_tool_parser="qwen3_xml",
    manager_reasoning_mode="thinking",
    manager_verify_tls=False,
    prompt_profile="teacher",
    concurrency=16,
    max_model_len=131072,
    temperature=_QWEN36_THINKING_SAMPLING.temperature,
    top_p=_QWEN36_THINKING_SAMPLING.top_p,
    top_k=_QWEN36_THINKING_SAMPLING.top_k,
    min_p=_QWEN36_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN36_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN36_THINKING_SAMPLING.repetition_penalty,
    manager_thinking=True,
    worker_thinking=False,
    worker_language_model_only=True,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
# Both actors on the shared LLM proxy, so the run needs no GPU. The manager is
# tau2's `qwen38_flash_teacher_thinking_low` teacher; the worker is the proxy's
# unlooped Qwen3.5-4B, non-thinking. Sampling values are the ones chosen for this
# pair (2026-09-25); unset penalties keep the serving defaults.
QWEN38_FLASH_MODEL_ID = "Qwen/Qwen3.8-Flash-Next-NVFP4"
QWEN35_4B_UNLOOPED_MODEL_ID = "Qwen/Qwen3.5-4B-unlooped"
QWEN35_4B_UNLOOPED_SAMPLING = WorkerSampling(
    temperature=0.6, top_p=0.95, top_k=20, max_completion_tokens=8192
)
QWEN38_LOW_QWEN35_UNLOOPED_EXPERIMENT = DecomposerExperiment(
    name="qwen38-flash-thinking-low-teacher-qwen35-4b-unlooped-non-thinking",
    worker_checkpoint=None,
    manager_backend="llm_proxy",
    manager_upstream_url_env="LLM_PROXY_URL",
    manager_api_key_env="LLM_PROXY_MASTER_KEY",
    manager_response_tool_parser="qwen3_xml",
    manager_reasoning_mode="thinking",
    manager_reasoning_effort="low",
    manager_verify_tls=False,
    worker_backend="llm_proxy",
    worker_upstream_url_env="LLM_PROXY_URL",
    worker_api_key_env="LLM_PROXY_MASTER_KEY",
    worker_verify_tls=False,
    prompt_profile="teacher",
    num_gpus=0,
    manager_served_name=QWEN38_FLASH_MODEL_ID,
    worker_served_name=QWEN35_4B_UNLOOPED_MODEL_ID,
    manager_port=8076,
    worker_port=8077,
    service_port=8155,
    subagent_port=2053,
    max_model_len=131072,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    worker_sampling=QWEN35_4B_UNLOOPED_SAMPLING,
    concurrency=16,
    manager_thinking=True,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_language_model_only=False,
    worker_thinking=False,
    worker_tool_call_parser="qwen3_xml",
    worker_reasoning_parser=None,
    worker_language_model_only=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
DEEPSEEK_GEMMA4_26B_NON_THINKING_EXPERIMENT = DecomposerExperiment(
    name="deepseek-v4-flash-0731-teacher-gemma4-26b-a4b-non-thinking",
    worker_checkpoint=GEMMA4_26B_A4B_BASE,
    manager_backend="openrouter",
    prompt_profile="teacher",
    num_gpus=1,
    manager_served_name="deepseek/deepseek-v4-flash-0731",
    manager_thinking=True,
    worker_served_name="google/gemma-4-26B-A4B-it",
    worker_port=8032,
    service_port=8135,
    subagent_port=2035,
    max_model_len=131072,
    worker_thinking=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
GEMMA4_26B_SHARED_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name=(
        "gemma4-26b-a4b-thinking-teacher-"
        "gemma4-26b-a4b-non-thinking-text-defaults"
    ),
    worker_checkpoint=GEMMA4_26B_A4B_BASE,
    manager_checkpoint=GEMMA4_26B_A4B_BASE,
    prompt_profile="teacher",
    num_gpus=1,
    manager_served_name="google/gemma-4-26B-A4B-it",
    worker_served_name="google/gemma-4-26B-A4B-it",
    manager_port=8033,
    worker_port=8033,
    service_port=8136,
    subagent_port=2036,
    max_model_len=131072,
    manager_thinking=True,
    worker_thinking=False,
    share_local_vllm=True,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
GEMMA4_31B_SHARED_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name=(
        "gemma4-31b-thinking-teacher-"
        "gemma4-31b-non-thinking-text-defaults"
    ),
    worker_checkpoint=GEMMA4_31B_BASE,
    manager_checkpoint=GEMMA4_31B_BASE,
    prompt_profile="teacher",
    num_gpus=1,
    manager_served_name="google/gemma-4-31B-it",
    worker_served_name="google/gemma-4-31B-it",
    manager_port=8035,
    worker_port=8035,
    service_port=8138,
    subagent_port=2038,
    max_model_len=131072,
    concurrency=32,
    manager_thinking=True,
    worker_thinking=False,
    share_local_vllm=True,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
# The 14T volume holds the SFT checkpoints; ARTIFACTS_ROOT is a separate filesystem.
GEMMA4_E2B_SFT_MIXED_V3_VLLM = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints"
    "/gemma4-e2b-nonthinking-4gpu-mixed-v3/final-vllm"
)
GEMMA4_E4B_SFT_MIXED_V3_VLLM = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints/gemma4-e4b-nonthinking-4gpu-mixed-v3/final-vllm"
)

# Qwen exports need no vllm_compat pass: that rebuilds Gemma-4's KV-shared
# k_norm tensors and rejects a config without num_kv_shared_layers.
QWEN35_4B_SFT_MIXED_V3 = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints"
    "/qwen35-4b-nonthinking-mixed-v3-8gpu/final"
)
# Same distillation data as the release above, stamped with the 111-character
# student prompt instead of the 7,842-character teacher prompt, to test whether
# the policy is carried by the demonstrations rather than by the prompt.
QWEN35_4B_SFT_MIXED_V3_STUDENT = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints"
    "/qwen35-4b-nonthinking-mixed-v3-student-4gpu/final"
)
GEMMA4_E4B_SFT_MIXED_V3_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name=("gemma4-e4b-sft-mixed-v3-non-thinking-gemma4-26b-a4b-non-thinking"),
    worker_checkpoint=GEMMA4_26B_A4B_BASE,
    manager_checkpoint=GEMMA4_E4B_SFT_MIXED_V3_VLLM,
    prompt_profile="teacher",
    num_gpus=2,
    manager_served_name="decomposer/gemma4-e4b-sft-mixed-v3",
    worker_served_name="google/gemma-4-26B-A4B-it",
    manager_port=8038,
    worker_port=8039,
    service_port=8140,
    subagent_port=2040,
    max_model_len=131072,
    manager_thinking=False,
    worker_thinking=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
GEMMA4_E2B_SFT_MIXED_V3_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name=("gemma4-e2b-sft-mixed-v3-non-thinking-gemma4-26b-a4b-non-thinking"),
    worker_checkpoint=GEMMA4_26B_A4B_BASE,
    manager_checkpoint=GEMMA4_E2B_SFT_MIXED_V3_VLLM,
    # The SFT release was built with the teacher prompt, so evaluate under it.
    prompt_profile="teacher",
    num_gpus=2,
    manager_served_name="decomposer/gemma4-e2b-sft-mixed-v3",
    worker_served_name="google/gemma-4-26B-A4B-it",
    manager_port=8036,
    worker_port=8037,
    service_port=8139,
    subagent_port=2039,
    max_model_len=131072,
    manager_thinking=False,
    worker_thinking=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)

_QWEN35_SFT_MIXED_V3_SAMPLING = qwen35_general_sampling(thinking=False)
QWEN35_SFT_MIXED_V3_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name="qwen35-4b-sft-mixed-v3-non-thinking-gemma4-26b-a4b-non-thinking",
    worker_checkpoint=GEMMA4_26B_A4B_BASE,
    manager_checkpoint=QWEN35_4B_SFT_MIXED_V3,
    # The SFT release was built with the teacher prompt, so evaluate under it.
    prompt_profile="teacher",
    num_gpus=2,
    manager_served_name="decomposer/qwen35-4b-sft-mixed-v3",
    worker_served_name="google/gemma-4-26B-A4B-it",
    manager_port=8040,
    worker_port=8041,
    service_port=8141,
    subagent_port=2041,
    max_model_len=131072,
    # The manager runs Qwen3.5's official non-thinking preset; the Gemma worker
    # keeps the preset every other 26B-A4B profile uses.
    temperature=_QWEN35_SFT_MIXED_V3_SAMPLING.temperature,
    top_p=_QWEN35_SFT_MIXED_V3_SAMPLING.top_p,
    top_k=_QWEN35_SFT_MIXED_V3_SAMPLING.top_k,
    min_p=_QWEN35_SFT_MIXED_V3_SAMPLING.min_p,
    presence_penalty=_QWEN35_SFT_MIXED_V3_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_SFT_MIXED_V3_SAMPLING.repetition_penalty,
    worker_sampling=WorkerSampling(temperature=1.0, top_p=0.95, top_k=64),
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_gdn_prefill_backend="triton",
    worker_thinking=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)

QWEN35_SFT_MIXED_V3_E4B_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_SFT_MIXED_V3_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-sft-mixed-v3-non-thinking-gemma4-e4b-non-thinking",
    worker_checkpoint=GEMMA4_E4B_BASE,
    worker_served_name="google/gemma-4-E4B-it",
    manager_port=8042,
    worker_port=8043,
    service_port=8143,
    subagent_port=2042,
)
QWEN35_SFT_MIXED_V3_E2B_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_SFT_MIXED_V3_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-sft-mixed-v3-non-thinking-gemma4-e2b-non-thinking",
    worker_checkpoint=GEMMA4_E2B_BASE,
    worker_served_name="google/gemma-4-E2B-it",
    manager_port=8044,
    worker_port=8045,
    service_port=8144,
    subagent_port=2043,
)

# The student-prompt release. Identical to the trio above except for the
# checkpoint and prompt_profile: same worker sampling, same budgets, so a
# difference in results is attributable to what the manager was trained under.
QWEN35_SFT_STUDENT_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_SFT_MIXED_V3_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-sft-student-non-thinking-gemma4-26b-a4b-non-thinking",
    manager_checkpoint=QWEN35_4B_SFT_MIXED_V3_STUDENT,
    manager_served_name="decomposer/qwen35-4b-sft-student",
    prompt_profile="student",
    manager_port=8070,
    worker_port=8071,
    service_port=8152,
    subagent_port=2050,
)
QWEN35_SFT_STUDENT_E4B_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_SFT_STUDENT_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-sft-student-non-thinking-gemma4-e4b-non-thinking",
    worker_checkpoint=GEMMA4_E4B_BASE,
    worker_served_name="google/gemma-4-E4B-it",
    manager_port=8072,
    worker_port=8073,
    service_port=8153,
    subagent_port=2051,
)
QWEN35_SFT_STUDENT_E2B_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_SFT_STUDENT_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-sft-student-non-thinking-gemma4-e2b-non-thinking",
    worker_checkpoint=GEMMA4_E2B_BASE,
    worker_served_name="google/gemma-4-E2B-it",
    manager_port=8074,
    worker_port=8075,
    service_port=8154,
    subagent_port=2052,
)

# Untuned-manager baselines. The manager is the raw Qwen3.5-4B snapshot rather
# than an SFT export, so unlike the tuned entries it needs the multimodal loader
# and remote code; the Gemma workers keep their own parsers and sampling. Each
# pairing is registered twice, once per prompt profile, with its own ports so the
# two can run concurrently.

QWEN35_BASE_E2B_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name="qwen35-4b-base-non-thinking-gemma4-e2b-non-thinking",
    worker_checkpoint=GEMMA4_E2B_BASE,
    manager_checkpoint=QWEN35_4B_BASE,
    manager_served_name=QWEN35_4B_BASE_MANAGER_SERVED_NAME,
    worker_served_name="google/gemma-4-E2B-it",
    num_gpus=2,
    manager_port=8046,
    worker_port=8047,
    service_port=8146,
    subagent_port=2044,
    max_model_len=131072,
    temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
    worker_sampling=WorkerSampling(temperature=1.0, top_p=0.95, top_k=64),
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_language_model_only=False,
    manager_trust_remote_code=True,
    manager_gdn_prefill_backend="triton",
    worker_thinking=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
QWEN35_BASE_TEACHER_E2B_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_BASE_E2B_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-base-non-thinking-teacher-gemma4-e2b-non-thinking",
    prompt_profile="teacher",
    manager_port=8048,
    worker_port=8049,
    service_port=8147,
    subagent_port=2045,
)

QWEN35_BASE_E4B_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name="qwen35-4b-base-non-thinking-gemma4-e4b-non-thinking",
    worker_checkpoint=GEMMA4_E4B_BASE,
    manager_checkpoint=QWEN35_4B_BASE,
    manager_served_name=QWEN35_4B_BASE_MANAGER_SERVED_NAME,
    worker_served_name="google/gemma-4-E4B-it",
    num_gpus=2,
    manager_port=8050,
    worker_port=8051,
    service_port=8148,
    subagent_port=2046,
    max_model_len=131072,
    temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
    worker_sampling=WorkerSampling(temperature=1.0, top_p=0.95, top_k=64),
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_language_model_only=False,
    manager_trust_remote_code=True,
    manager_gdn_prefill_backend="triton",
    worker_thinking=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
QWEN35_BASE_TEACHER_E4B_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_BASE_E4B_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-base-non-thinking-teacher-gemma4-e4b-non-thinking",
    prompt_profile="teacher",
    manager_port=8052,
    worker_port=8053,
    service_port=8149,
    subagent_port=2047,
)

QWEN35_BASE_26B_A4B_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name="qwen35-4b-base-non-thinking-gemma4-26b-a4b-non-thinking",
    worker_checkpoint=GEMMA4_26B_A4B_BASE,
    manager_checkpoint=QWEN35_4B_BASE,
    manager_served_name=QWEN35_4B_BASE_MANAGER_SERVED_NAME,
    worker_served_name="google/gemma-4-26B-A4B-it",
    num_gpus=2,
    manager_port=8054,
    worker_port=8055,
    service_port=8150,
    subagent_port=2048,
    max_model_len=131072,
    temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
    top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
    top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
    min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
    presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
    repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
    worker_sampling=WorkerSampling(temperature=1.0, top_p=0.95, top_k=64),
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_language_model_only=False,
    manager_trust_remote_code=True,
    manager_gdn_prefill_backend="triton",
    worker_thinking=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
QWEN35_BASE_TEACHER_26B_A4B_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_BASE_26B_A4B_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-base-non-thinking-teacher-gemma4-26b-a4b-non-thinking",
    prompt_profile="teacher",
    manager_port=8056,
    worker_port=8057,
    service_port=8151,
    subagent_port=2049,
)
DEEPSEEK_PRO_GEMMA4_26B_NON_THINKING_EXPERIMENT = DecomposerExperiment(
    name="deepseek-v4-pro-0813-teacher-gemma4-26b-a4b-non-thinking",
    worker_checkpoint=GEMMA4_26B_A4B_BASE,
    manager_backend="openrouter",
    prompt_profile="teacher",
    num_gpus=1,
    manager_served_name="deepseek/deepseek-v4-pro-0813",
    manager_thinking=True,
    worker_served_name="google/gemma-4-26B-A4B-it",
    worker_port=8034,
    service_port=8137,
    subagent_port=2037,
    max_model_len=131072,
    worker_thinking=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)


def _gemma4_simple_experiment(
    name: str, checkpoint: Path, served_name: str, *, thinking: bool
) -> SimpleExperiment:
    return SimpleExperiment(
        name=name,
        checkpoint=checkpoint,
        served_name=served_name,
        max_model_len=131072,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        thinking=thinking,
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        language_model_only=True,
        max_model_calls=80,
    )


SIMPLE_GEMMA4_E2B_NON_THINKING_EXPERIMENT = _gemma4_simple_experiment(
    "gemma4-e2b-it-non-thinking",
    GEMMA4_E2B_BASE,
    "google/gemma-4-E2B-it",
    thinking=False,
)
SIMPLE_GEMMA4_E2B_THINKING_EXPERIMENT = _gemma4_simple_experiment(
    "gemma4-e2b-it-thinking",
    GEMMA4_E2B_BASE,
    "google/gemma-4-E2B-it",
    thinking=True,
)
SIMPLE_GEMMA4_E4B_NON_THINKING_EXPERIMENT = _gemma4_simple_experiment(
    "gemma4-e4b-it-non-thinking",
    GEMMA4_E4B_BASE,
    "google/gemma-4-E4B-it",
    thinking=False,
)
SIMPLE_EXPERIMENT = _gemma4_simple_experiment(
    "gemma4-e4b-it-thinking",
    GEMMA4_E4B_BASE,
    "google/gemma-4-E4B-it",
    thinking=True,
)
SIMPLE_GEMMA4_31B_NON_THINKING_EXPERIMENT = _gemma4_simple_experiment(
    "gemma4-31b-it-non-thinking",
    GEMMA4_31B_BASE,
    "google/gemma-4-31B-it",
    thinking=False,
)
SIMPLE_GEMMA4_31B_THINKING_EXPERIMENT = _gemma4_simple_experiment(
    "gemma4-31b-it-thinking",
    GEMMA4_31B_BASE,
    "google/gemma-4-31B-it",
    thinking=True,
)
SIMPLE_GEMMA4_26B_NON_THINKING_EXPERIMENT = _gemma4_simple_experiment(
    "gemma4-26b-a4b-it-non-thinking",
    GEMMA4_26B_A4B_BASE,
    "google/gemma-4-26B-A4B-it",
    thinking=False,
)
SIMPLE_GEMMA4_26B_THINKING_EXPERIMENT = _gemma4_simple_experiment(
    "gemma4-26b-a4b-it-thinking",
    GEMMA4_26B_A4B_BASE,
    "google/gemma-4-26B-A4B-it",
    thinking=True,
)


def _qwen35_simple_experiment(
    name: str, checkpoint: Path, served_name: str
) -> SimpleExperiment:
    return SimpleExperiment(
        name=name,
        checkpoint=checkpoint,
        served_name=served_name,
        max_model_len=131072,
        temperature=_QWEN35_NON_THINKING_SAMPLING.temperature,
        top_p=_QWEN35_NON_THINKING_SAMPLING.top_p,
        top_k=_QWEN35_NON_THINKING_SAMPLING.top_k,
        min_p=_QWEN35_NON_THINKING_SAMPLING.min_p,
        presence_penalty=_QWEN35_NON_THINKING_SAMPLING.presence_penalty,
        repetition_penalty=_QWEN35_NON_THINKING_SAMPLING.repetition_penalty,
        thinking=False,
        tool_call_parser="qwen3_xml",
        reasoning_parser=None,
        language_model_only=False,
        trust_remote_code=True,
        gdn_prefill_backend="triton",
    )


SIMPLE_QWEN_EXPERIMENT = _qwen35_simple_experiment(
    "qwen35-4b-non-thinking", QWEN35_4B_BASE, "Qwen/Qwen3.5-4B"
)
SIMPLE_QWEN_2B_EXPERIMENT = _qwen35_simple_experiment(
    "qwen35-2b-base-non-thinking", QWEN35_2B_BASE, "Qwen/Qwen3.5-2B"
)
SIMPLE_QWEN_9B_EXPERIMENT = _qwen35_simple_experiment(
    "qwen35-9b-base-non-thinking", QWEN35_9B_BASE, "Qwen/Qwen3.5-9B"
)
GEMMA4_E4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT = replace(
    SIMPLE_EXPERIMENT,
    name="gemma4-e4b-thinking-simple-text-defaults",
    max_model_len=131072,
    max_model_calls=80,
)
QWEN35_4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT = replace(
    SIMPLE_QWEN_EXPERIMENT,
    name="qwen35-4b-non-thinking-simple-general-text-defaults",
    max_model_len=131072,
    language_model_only=True,
    max_model_calls=80,
)
SIMPLE_DEEPSEEK_EXPERIMENT = SimpleExperiment(
    name="deepseek-v4-flash-0731",
    checkpoint=None,
    backend="openrouter",
    num_gpus=0,
    served_name="deepseek/deepseek-v4-flash-0731",
    port=8140,
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY_DECOMPOSER",
    reasoning_effort="max",
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    max_num_seqs=0,
    concurrency=4,
    thinking=True,
    tool_call_parser="",
    reasoning_parser=None,
    language_model_only=False,
    gdn_prefill_backend=None,
)
# ARE's native agent on the Decomposer worker's model and sampling above.
SIMPLE_QWEN35_UNLOOPED_EXPERIMENT = SimpleExperiment(
    name="qwen35-4b-unlooped-non-thinking-simple",
    checkpoint=None,
    backend="llm_proxy",
    num_gpus=0,
    served_name=QWEN35_4B_UNLOOPED_MODEL_ID,
    port=8078,
    upstream_url_env="LLM_PROXY_URL",
    api_key_env="LLM_PROXY_MASTER_KEY",
    verify_tls=False,
    max_model_len=131072,
    max_completion_tokens=QWEN35_4B_UNLOOPED_SAMPLING.max_completion_tokens,
    temperature=QWEN35_4B_UNLOOPED_SAMPLING.temperature,
    top_p=QWEN35_4B_UNLOOPED_SAMPLING.top_p,
    top_k=QWEN35_4B_UNLOOPED_SAMPLING.top_k,
    concurrency=16,
    thinking=False,
    tool_call_parser="",
    reasoning_parser=None,
    language_model_only=False,
    max_model_calls=80,
)
# Release 1.0.0 of decomposer-manager-sft (sft/specs/decomposer_manager_sft_1.0.0.yaml):
# its teacher, the unloop student trained on it and that student before training,
# all with the release's subagents and the teacher prompt the release was built
# with. Managers may emit parallel tool calls, as the teacher did; a local vLLM with
# parallel_tool_calls=false keeps only the first call.
QWEN35_4B_UNLOOP_BASE = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/models/qwen35_4b_original_unloop"
    "/checkpoint-0"
)
QWEN35_4B_UNLOOP_SFT_1_0_0_FULL = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints"
    "/qwen35-4b-unloop-nonthinking-manager-sft-1.0.0-full-32k/final"
)
QWEN35_4B_UNLOOP_SFT_1_0_0_LORA = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints"
    "/qwen35-4b-unloop-nonthinking-manager-sft-1.0.0-lora-32k/final"
)
# tau2's lmrouter/qwen_3_5_4b_unlooped_thinking preset, the release's subagents.
QWEN35_4B_UNLOOPED_THINKING_SAMPLING = WorkerSampling(
    temperature=0.6, top_p=0.95, top_k=20
)
# tau2's lmrouter/qwen_3_8_flash_next_non_thinking preset, the release's teacher.
QWEN38_NON_THINKING_QWEN35_UNLOOPED_THINKING_EXPERIMENT = replace(
    QWEN38_LOW_QWEN35_UNLOOPED_EXPERIMENT,
    name="qwen38-flash-non-thinking-teacher-qwen35-4b-unlooped-thinking",
    manager_reasoning_mode="non_thinking",
    manager_reasoning_effort="none",
    manager_port=8079,
    worker_port=8080,
    service_port=8156,
    subagent_port=2054,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    worker_sampling=QWEN35_4B_UNLOOPED_THINKING_SAMPLING,
    manager_parallel_tool_calls=True,
    manager_thinking=False,
    worker_thinking=True,
)
# The unloop student samples with the unlooped model's non-thinking values, which
# send no penalty (gyms/qwen_sampling.py), and keeps the manager uncapped.
QWEN35_UNLOOP_SFT_1_0_0_FULL_DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name="qwen35-4b-unloop-sft-1.0.0-full-32k-non-thinking-qwen35-4b-unlooped-thinking",
    worker_checkpoint=None,
    manager_checkpoint=QWEN35_4B_UNLOOP_SFT_1_0_0_FULL,
    worker_backend="llm_proxy",
    worker_upstream_url_env="LLM_PROXY_URL",
    worker_api_key_env="LLM_PROXY_MASTER_KEY",
    worker_verify_tls=False,
    prompt_profile="teacher",
    num_gpus=1,
    manager_served_name="decomposer/qwen35-4b-unloop-sft-1.0.0-full-32k",
    worker_served_name=QWEN35_4B_UNLOOPED_MODEL_ID,
    manager_port=8081,
    worker_port=8082,
    service_port=8157,
    subagent_port=2055,
    max_model_len=131072,
    temperature=QWEN35_UNLOOPED_NON_THINKING.temperature,
    top_p=QWEN35_UNLOOPED_NON_THINKING.top_p,
    top_k=QWEN35_UNLOOPED_NON_THINKING.top_k,
    worker_sampling=QWEN35_4B_UNLOOPED_THINKING_SAMPLING,
    concurrency=16,
    manager_parallel_tool_calls=True,
    manager_thinking=False,
    manager_tool_call_parser="qwen3_xml",
    manager_reasoning_parser=None,
    manager_gdn_prefill_backend="triton",
    worker_thinking=True,
    worker_tool_call_parser="qwen3_xml",
    worker_reasoning_parser=None,
    worker_language_model_only=False,
    manager_max_model_calls=80,
    subagent_max_model_calls=80,
    manager_recursion_limit=1000,
    subagent_recursion_limit=1000,
)
QWEN35_UNLOOP_SFT_1_0_0_LORA_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_UNLOOP_SFT_1_0_0_FULL_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-unloop-sft-1.0.0-lora-32k-non-thinking-qwen35-4b-unlooped-thinking",
    manager_checkpoint=QWEN35_4B_UNLOOP_SFT_1_0_0_LORA,
    manager_served_name="decomposer/qwen35-4b-unloop-sft-1.0.0-lora-32k",
    manager_port=8083,
    worker_port=8084,
    service_port=8158,
    subagent_port=2056,
)
# checkpoint-0 has the raw Qwen3.5-4B snapshot's layout, so it is served the same way.
QWEN35_UNLOOP_BASE_TEACHER_DECOMPOSER_EXPERIMENT = replace(
    QWEN35_UNLOOP_SFT_1_0_0_FULL_DECOMPOSER_EXPERIMENT,
    name="qwen35-4b-unloop-base-non-thinking-teacher-qwen35-4b-unlooped-thinking",
    manager_checkpoint=QWEN35_4B_UNLOOP_BASE,
    manager_served_name="decomposer/qwen35-4b-unloop-base-manager",
    manager_language_model_only=False,
    manager_trust_remote_code=True,
    manager_port=8085,
    worker_port=8086,
    service_port=8159,
    subagent_port=2057,
)
ALL_EXPERIMENTS: tuple[Experiment, ...] = (
    DECOMPOSER_EXPERIMENT,
    DEEPSEEK_GEMMA_EXPERIMENT,
    QWEN35_SFT_EXPERIMENT,
    QWEN35_MIXED_SFT_EXPERIMENT,
    QWEN35_FINAL_MIXED_SFT_EXPERIMENT,
    QWEN35_FILTERED_SFT_EXPERIMENT,
    QWEN35_GAIA2_SFT_EXPERIMENT,
    QWEN35_TOOLATHLON_ONLY_SFT_EXPERIMENT,
    QWEN35_GAIA2_EXECUTION_ONLY_SFT_EXPERIMENT,
    QWEN35_BASE_DECOMPOSER_EXPERIMENT,
    QWEN35_BASE_TEACHER_DECOMPOSER_EXPERIMENT,
    DEEPSEEK_QWEN_EXPERIMENT,
    DEEPSEEK_QWEN_AMBIGUITY_POLICY_EXPERIMENT,
    QWEN36_QWEN_EXPERIMENT,
    GEMMA4_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT,
    QWEN36_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT,
    QWEN36_THINKING_TEXT_DEFAULTS_DECOMPOSER_EXPERIMENT,
    DEEPSEEK_GEMMA4_26B_NON_THINKING_EXPERIMENT,
    GEMMA4_26B_SHARED_DECOMPOSER_EXPERIMENT,
    GEMMA4_31B_SHARED_DECOMPOSER_EXPERIMENT,
    SIMPLE_GEMMA4_E2B_NON_THINKING_EXPERIMENT,
    SIMPLE_GEMMA4_E2B_THINKING_EXPERIMENT,
    SIMPLE_GEMMA4_E4B_NON_THINKING_EXPERIMENT,
    SIMPLE_EXPERIMENT,
    SIMPLE_GEMMA4_31B_NON_THINKING_EXPERIMENT,
    SIMPLE_GEMMA4_31B_THINKING_EXPERIMENT,
    SIMPLE_GEMMA4_26B_NON_THINKING_EXPERIMENT,
    SIMPLE_GEMMA4_26B_THINKING_EXPERIMENT,
    SIMPLE_QWEN_EXPERIMENT,
    SIMPLE_QWEN_2B_EXPERIMENT,
    SIMPLE_QWEN_9B_EXPERIMENT,
    GEMMA4_E4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT,
    QWEN35_4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT,
    SIMPLE_DEEPSEEK_EXPERIMENT,
    DEEPSEEK_PRO_GEMMA4_26B_NON_THINKING_EXPERIMENT,
    GEMMA4_E2B_SFT_MIXED_V3_DECOMPOSER_EXPERIMENT,
    GEMMA4_E4B_SFT_MIXED_V3_DECOMPOSER_EXPERIMENT,
    QWEN35_SFT_MIXED_V3_DECOMPOSER_EXPERIMENT,
    QWEN35_SFT_MIXED_V3_E4B_DECOMPOSER_EXPERIMENT,
    QWEN35_SFT_MIXED_V3_E2B_DECOMPOSER_EXPERIMENT,
    QWEN35_SFT_STUDENT_DECOMPOSER_EXPERIMENT,
    QWEN35_SFT_STUDENT_E4B_DECOMPOSER_EXPERIMENT,
    QWEN35_SFT_STUDENT_E2B_DECOMPOSER_EXPERIMENT,
    QWEN35_BASE_E2B_DECOMPOSER_EXPERIMENT,
    QWEN35_BASE_TEACHER_E2B_DECOMPOSER_EXPERIMENT,
    QWEN35_BASE_E4B_DECOMPOSER_EXPERIMENT,
    QWEN35_BASE_TEACHER_E4B_DECOMPOSER_EXPERIMENT,
    QWEN35_BASE_26B_A4B_DECOMPOSER_EXPERIMENT,
    QWEN35_BASE_TEACHER_26B_A4B_DECOMPOSER_EXPERIMENT,
    QWEN38_LOW_QWEN35_UNLOOPED_EXPERIMENT,
    SIMPLE_QWEN35_UNLOOPED_EXPERIMENT,
    QWEN38_NON_THINKING_QWEN35_UNLOOPED_THINKING_EXPERIMENT,
    QWEN35_UNLOOP_SFT_1_0_0_FULL_DECOMPOSER_EXPERIMENT,
    QWEN35_UNLOOP_SFT_1_0_0_LORA_DECOMPOSER_EXPERIMENT,
    QWEN35_UNLOOP_BASE_TEACHER_DECOMPOSER_EXPERIMENT,
)
EXPERIMENTS = {experiment.name: experiment for experiment in ALL_EXPERIMENTS}
if len(EXPERIMENTS) != len(ALL_EXPERIMENTS):
    raise ValueError("Gaia2 experiment names must be unique")


def get_experiment(name: str) -> Experiment:
    try:
        return EXPERIMENTS[name]
    except KeyError as error:
        expected = ", ".join(EXPERIMENTS)
        raise ValueError(
            f"Unknown Gaia2 experiment {name!r}; expected: {expected}"
        ) from error


def collect_experiments(
    names: tuple[str, ...] = (),
    filters: tuple[str, ...] = (),
    *,
    default_all: bool = True,
) -> list[Experiment]:
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown Gaia2 experiments: {unknown}")
    if not names and not filters:
        return list(ALL_EXPERIMENTS) if default_all else []
    selected = set(names)
    selected.update(
        experiment.name
        for experiment in ALL_EXPERIMENTS
        if any(pattern in experiment.name for pattern in filters)
    )
    return [experiment for experiment in ALL_EXPERIMENTS if experiment.name in selected]


def preparation_manifest(
    experiment: Experiment, domain: Gaia2Domain = DOMAIN
) -> Path:
    return DATA_ROOT / "manifests" / SPLIT / domain / f"{experiment.name}.json"


def run_name(
    experiment: Experiment,
    num_repeats: int,
    *,
    prompt_profile: DecomposerPromptProfile | None = None,
) -> str:
    if num_repeats < 1:
        raise ValueError("num_repeats must be at least 1")
    name = experiment.name if num_repeats == 1 else f"{experiment.name}-n{num_repeats}"
    return name if prompt_profile is None else f"{name}-prompt-{prompt_profile}"


def output_dir(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None = None,
    *,
    partition: Partition = "full",
    prompt_profile: DecomposerPromptProfile | None = None,
    domain: Gaia2Domain = DOMAIN,
) -> Path:
    spec = get_domain_spec(domain)
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown Gaia2 partition: {partition!r}")
    root = RESULTS_ROOT / SPLIT / spec.name
    if partition != "full":
        root = root / "partitions" / spec.split_manifest_name / partition
    base = root / run_name(experiment, num_repeats, prompt_profile=prompt_profile)
    return base if limit is None else base / f"smoke_{limit}"


def completion_marker(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None = None,
    *,
    partition: Partition = "full",
    prompt_profile: DecomposerPromptProfile | None = None,
    domain: Gaia2Domain = DOMAIN,
) -> Path:
    return (
        output_dir(
            experiment,
            num_repeats,
            limit,
            partition=partition,
            prompt_profile=prompt_profile,
            domain=domain,
        )
        / ".eval_done.json"
    )


def trace_run_name(
    experiment: Experiment,
    num_repeats: int,
    rollout_offset: int,
    *,
    prompt_profile: DecomposerPromptProfile | None = None,
) -> str:
    if num_repeats < 1:
        raise ValueError("num_repeats must be at least 1")
    if rollout_offset < 0:
        raise ValueError("rollout_offset cannot be negative")
    first = rollout_offset + 1
    last = rollout_offset + num_repeats
    name = f"{experiment.name}-r{first:02d}-r{last:02d}"
    return name if prompt_profile is None else f"{name}-prompt-{prompt_profile}"


def trace_output_dir(
    experiment: Experiment,
    num_repeats: int,
    rollout_offset: int,
    partition: Partition,
    limit: int | None = None,
    *,
    prompt_profile: DecomposerPromptProfile | None = None,
    domain: Gaia2Domain = DOMAIN,
) -> Path:
    spec = get_domain_spec(domain)
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown Gaia2 partition: {partition!r}")
    base = (
        TRACES_ROOT
        / spec.split_manifest_name
        / partition
        / trace_run_name(
            experiment,
            num_repeats,
            rollout_offset,
            prompt_profile=prompt_profile,
        )
    )
    return base if limit is None else base / f"smoke_{limit}"


def trace_completion_marker(
    experiment: Experiment,
    num_repeats: int,
    rollout_offset: int,
    partition: Partition,
    limit: int | None = None,
    *,
    prompt_profile: DecomposerPromptProfile | None = None,
    domain: Gaia2Domain = DOMAIN,
) -> Path:
    return (
        trace_output_dir(
            experiment,
            num_repeats,
            rollout_offset,
            partition,
            limit,
            prompt_profile=prompt_profile,
            domain=domain,
        )
        / ".trace_done.json"
    )


def job_description(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None = None,
    *,
    partition: Partition = "full",
    domain: Gaia2Domain = DOMAIN,
) -> str:
    spec = get_domain_spec(domain)
    identity = run_name(experiment, num_repeats)
    if limit is not None:
        identity += f"-smoke-{limit}"
    if partition == "full":
        return f"gaia2-{SPLIT}-{spec.name} {experiment.kind}-agent {identity}"
    return (
        f"gaia2-{SPLIT}-{spec.name} {spec.split_manifest_name}-{partition} "
        f"{experiment.kind}-agent {identity}"
    )
