"""Experiments-as-code and artifact layout for Gaia2 evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from gyms.gaia2.prompts import Gaia2ManagerPromptAddendumProfile
from gyms.qwen_sampling import (
    qwen35_general_sampling,
    qwen36_non_thinking_sampling,
    qwen36_thinking_sampling,
)

ARTIFACTS_ROOT = Path("/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts")
PROJECT_ROOT = Path("/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_sft")
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
HF_HOME = Path("/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface")

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
DecomposerPromptProfile = Literal["student", "teacher"]
SimpleAgentBackend = Literal["local_vllm", "openrouter"]
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
class DecomposerExperiment:
    name: str
    worker_checkpoint: Path
    manager_checkpoint: Path | None = None
    manager_backend: DecomposerManagerBackend = "local_vllm"
    manager_upstream_url_env: str | None = None
    manager_api_key_env: str | None = None
    manager_response_tool_parser: str | None = None
    manager_reasoning_mode: Literal[
        "service_default", "non_thinking", "thinking"
    ] | None = None
    manager_verify_tls: bool = True
    prompt_profile: DecomposerPromptProfile = "student"
    manager_prompt_addendum_profile: Gaia2ManagerPromptAddendumProfile | None = None
    num_gpus: int = 2
    manager_served_name: str = "decomposer/gemma4-e4b-sft-deepseek-e4b-v1-8k"
    worker_served_name: str = "google/gemma-4-E4B-it"
    manager_port: int = 8020
    worker_port: int = 8021
    service_port: int = 8124
    subagent_port: int = 2024
    max_model_len: int = 65536
    max_num_seqs: int = 16
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
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
        expected_gpus = 2 if self.manager_backend == "local_vllm" else 1
        if self.num_gpus != expected_gpus:
            raise ValueError(
                f"{self.manager_backend} Decomposer requires {expected_gpus} GPU(s)"
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

    @property
    def requires_local_manager(self) -> bool:
        return self.manager_backend == "local_vllm"

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
    api_key_env: str | None = None
    reasoning_effort: str | None = None
    max_model_len: int = 65536
    max_num_seqs: int = 16
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
            if any(value is not None for value in (self.base_url, self.api_key_env)):
                raise ValueError(
                    f"{self.name}: local_vllm cannot configure remote model fields"
                )
        elif self.backend == "openrouter":
            if self.checkpoint is not None:
                raise ValueError(f"{self.name}: openrouter cannot use checkpoint")
            if self.num_gpus != 0:
                raise ValueError(f"{self.name}: openrouter simple agent uses no GPU")
            if not self.base_url or not self.api_key_env:
                raise ValueError(
                    f"{self.name}: openrouter requires base_url and api_key_env"
                )

    @property
    def requires_openrouter(self) -> bool:
        return self.backend == "openrouter"

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
)
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
SIMPLE_EXPERIMENT = SimpleExperiment(
    name="gemma4-e4b-it-thinking",
    checkpoint=GEMMA4_E4B_BASE,
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
    reasoning_effort="high",
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
    SIMPLE_EXPERIMENT,
    SIMPLE_QWEN_EXPERIMENT,
    SIMPLE_QWEN_2B_EXPERIMENT,
    SIMPLE_QWEN_9B_EXPERIMENT,
    GEMMA4_E4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT,
    QWEN35_4B_TEXT_DEFAULTS_SIMPLE_EXPERIMENT,
    SIMPLE_DEEPSEEK_EXPERIMENT,
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
