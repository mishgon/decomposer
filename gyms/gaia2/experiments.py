"""Experiments-as-code and artifact layout for Gaia2 execution evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from gyms.qwen_sampling import qwen35_general_sampling

ARTIFACTS_ROOT = Path("/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts")
PROJECT_ROOT = Path("/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_sft")
PROJECT_VENV = PROJECT_ROOT / ".venv"
DEFAULT_GAIA2_REPO = PROJECT_ROOT / "external" / "gaia2"

DATASET_ID = "meta-agents-research-environments/gaia2"
DATASET_REVISION = "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99"
FILESYSTEM_DATASET_ID = "meta-agents-research-environments/gaia2_filesystem"
FILESYSTEM_DATASET_REVISION = "132e26376f5e963bb59f64bcccdd02188cb08dee"
GAIA2_REVISION = "3bee736488864e028231755ce2ee32a7065e8648"
SPLIT = "validation"
DOMAIN = "execution"
SCENARIO_COUNT = 160

DATA_ROOT = ARTIFACTS_ROOT / "evaluation" / "data" / "gaia2"
RESULTS_ROOT = ARTIFACTS_ROOT / "evaluation" / "gaia2" / "results"
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
QWEN35_4B_SFT_SERVED_NAME = (
    "decomposer/qwen35-4b-sft-workplace-v1-3765-32k"
)
QWEN35_4B_BASE_MANAGER_SERVED_NAME = "decomposer/qwen35-4b-base-manager"
QWEN35_4B_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "qwen35-4b-nonthinking-workplace-v1-3765-32k-full-4gpu"
    / "final"
)

DecomposerManagerBackend = Literal["local_vllm", "openrouter"]
DecomposerPromptProfile = Literal["student", "teacher"]


def gaia2_lock_hash(gaia2_root: Path) -> str:
    return hashlib.sha256((gaia2_root / "uv.lock").read_bytes()).hexdigest()[:16]


def gaia2_venv(gaia2_root: Path) -> Path:
    return GAIA2_VENV_ROOT / gaia2_lock_hash(gaia2_root)


def dataset_revision_root() -> Path:
    return DATA_ROOT / DATASET_REVISION


def dataset_root() -> Path:
    """Root passed to ARE with ``--config execution``."""

    return dataset_revision_root() / SPLIT


def scenario_dir() -> Path:
    return dataset_root() / DOMAIN


def dataset_manifest() -> Path:
    return dataset_revision_root() / "dataset_manifest.json"


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
    prompt_profile: DecomposerPromptProfile = "student"
    num_gpus: int = 2
    manager_served_name: str = "decomposer/gemma4-e4b-sft-deepseek-e4b-v1-8k"
    worker_served_name: str = "google/gemma-4-E4B-it"
    manager_port: int = 8020
    worker_port: int = 8021
    service_port: int = 8124
    subagent_port: int = 2024
    max_model_len: int = 65536
    max_num_seqs: int = 16
    max_completion_tokens: int = 4096
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
    kind: Literal["decomposer"] = field(init=False, default="decomposer")

    def __post_init__(self) -> None:
        if self.manager_backend == "local_vllm" and self.manager_checkpoint is None:
            raise ValueError("A local_vllm manager requires manager_checkpoint")
        expected_gpus = 2 if self.manager_backend == "local_vllm" else 1
        if self.num_gpus != expected_gpus:
            raise ValueError(
                f"{self.manager_backend} Decomposer requires {expected_gpus} GPU(s)"
            )

    @property
    def requires_local_manager(self) -> bool:
        return self.manager_backend == "local_vllm"

    @property
    def requires_openrouter(self) -> bool:
        return self.manager_backend == "openrouter"


@dataclass(frozen=True)
class SimpleExperiment:
    name: str
    checkpoint: Path
    num_gpus: int = 1
    served_name: str = "google/gemma-4-E4B-it"
    port: int = 8100
    max_model_len: int = 65536
    max_num_seqs: int = 16
    max_completion_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
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
    kind: Literal["simple"] = field(init=False, default="simple")


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
    name=(
        "qwen35-4b-sft-workplace-v1-3765-32k-non-thinking-"
        "qwen35-4b-non-thinking"
    ),
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
SIMPLE_EXPERIMENT = SimpleExperiment(
    name="gemma4-e4b-it-thinking",
    checkpoint=GEMMA4_E4B_BASE,
)
SIMPLE_QWEN_EXPERIMENT = SimpleExperiment(
    name="qwen35-4b-non-thinking",
    checkpoint=QWEN35_4B_BASE,
    served_name="Qwen/Qwen3.5-4B",
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
ALL_EXPERIMENTS: tuple[Experiment, ...] = (
    DECOMPOSER_EXPERIMENT,
    DEEPSEEK_GEMMA_EXPERIMENT,
    QWEN35_SFT_EXPERIMENT,
    QWEN35_BASE_DECOMPOSER_EXPERIMENT,
    DEEPSEEK_QWEN_EXPERIMENT,
    SIMPLE_EXPERIMENT,
    SIMPLE_QWEN_EXPERIMENT,
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


def preparation_manifest(experiment: Experiment) -> Path:
    return PREPARATION_MANIFEST_ROOT / f"{experiment.name}.json"


def run_name(experiment: Experiment, num_repeats: int) -> str:
    if num_repeats < 1:
        raise ValueError("num_repeats must be at least 1")
    return experiment.name if num_repeats == 1 else f"{experiment.name}-n{num_repeats}"


def output_dir(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None = None,
) -> Path:
    base = RESULTS_ROOT / SPLIT / DOMAIN / run_name(experiment, num_repeats)
    return base if limit is None else base / f"smoke_{limit}"


def completion_marker(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None = None,
) -> Path:
    return output_dir(experiment, num_repeats, limit) / ".eval_done.json"


def job_description(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None = None,
) -> str:
    identity = run_name(experiment, num_repeats)
    if limit is not None:
        identity += f"-smoke-{limit}"
    return f"gaia2-{SPLIT}-{DOMAIN} {experiment.kind}-agent {identity}"
