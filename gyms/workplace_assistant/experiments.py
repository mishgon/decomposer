"""Experiments-as-code registry for Workplace Assistant runs."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

from gyms.qwen_sampling import (
    QwenSamplingParams,
    qwen35_general_sampling,
    qwen36_non_thinking_sampling,
    qwen36_thinking_sampling,
)

ARTIFACTS_ROOT = Path("/home/sukhorukov/decomposer_artifacts")
PROJECT_VENV = Path("/home/sukhorukov/decomposer_sft/.venv")
HF_HUB_ROOT = Path("/home/sukhorukov/.cache/huggingface/hub")
CHECKPOINTS_ROOT = Path("/home/sukhorukov/checkpoints")

DATA_DIR = ARTIFACTS_ROOT / "evaluation" / "data" / "workplace_assistant"
RESULTS_ROOT = ARTIFACTS_ROOT / "evaluation" / "results"
STAGING_ROOT = ARTIFACTS_ROOT / "code"
GYM_VENV_ROOT = ARTIFACTS_ROOT / "venvs" / "gym"
COMPONENT_VENV_ROOT = ARTIFACTS_ROOT / "venvs" / "workplace-assistant"
UV_CACHE = ARTIFACTS_ROOT / "cache" / "uv"
UV_BIN = ARTIFACTS_ROOT / "tools" / "uv"
HF_HOME = ARTIFACTS_ROOT / "cache" / "huggingface"
SFT_OUTPUT_ROOT = ARTIFACTS_ROOT / "datasets" / "sft"

BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.42"
INSTANCE_TYPES_BY_NUM_GPUS = {
    1: "a100plus.1gpu.80vG.12C.244G",
    2: "a100plus.2gpu.80vG.24C.488G",
    3: "a100plus.3gpu.80vG.36C.546G",
}
SPLIT_ROWS = {"train": 1255, "validation": 545}
SPLITS = tuple(SPLIT_ROWS)
RUN_PURPOSES = ("trace-generation", "evaluation")
WORKPLACE_MODEL_CALL_LIMIT = 100
WORKPLACE_SUBAGENT_RECURSION_LIMIT = 1000
RunPurpose = Literal["trace-generation", "evaluation"]
DecomposerPromptProfile = Literal["teacher", "student"]
DecomposerManagerBackend = Literal["openrouter", "llm_proxy", "local_vllm"]
SimpleAgentBackend = Literal["openrouter", "local_vllm"]


def source_dataset(split: str) -> Path:
    validate_split(split)
    return DATA_DIR / f"{split}.jsonl"


def decomposer_dataset(split: str) -> Path:
    validate_split(split)
    return DATA_DIR / f"{split}.decomposer.jsonl"


def preparation_manifest(split: str, experiment_name: str) -> Path:
    validate_split(split)
    return DATA_DIR / "manifests" / split / f"{experiment_name}.json"


def validate_split(split: str) -> str:
    if split not in SPLIT_ROWS:
        expected = ", ".join(SPLITS)
        raise ValueError(f"Unknown split {split!r}; expected one of: {expected}")
    return split


def gym_lock_hash(repo_root: Path) -> str:
    lock = repo_root / "external" / "Gym" / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]


def gym_venv(repo_root: Path) -> Path:
    return GYM_VENV_ROOT / gym_lock_hash(repo_root)


def component_runtime_key(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for path in (repo_root / "uv.lock", repo_root / "external" / "Gym" / "uv.lock"):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def component_venv_root(repo_root: Path) -> Path:
    return COMPONENT_VENV_ROOT / component_runtime_key(repo_root)


@dataclass(frozen=True)
class ModelServer:
    model_id: str
    snapshot: Path
    port: int
    gpu: int
    gpu_memory_utilization: float
    startup_wave: int
    thinking: bool = True
    tool_call_parser: str = "gemma4"
    reasoning_parser: str | None = "gemma4"
    gdn_prefill_backend: str | None = None
    dtype: str | None = None


@dataclass(frozen=True)
class DecomposerExperiment:
    name: str
    gym_config_filename: str
    manager_backend: DecomposerManagerBackend = "openrouter"
    manager_model_id: str | None = None
    manager_proxy_port: int | None = None
    manager_upstream_url_env: str | None = None
    manager_api_key_env: str | None = None
    manager_response_tool_parser: str | None = None
    manager_reasoning_mode: Literal[
        "service_default", "non_thinking", "thinking"
    ] | None = None
    manager_verify_tls: bool = True
    manager_sampling: QwenSamplingParams | None = None
    evaluation_prompt_profile: DecomposerPromptProfile = "student"
    concurrency: int = 8
    max_model_len: int = 32768
    max_num_seqs: int = 64
    langgraph_jobs: int = 16
    num_gpus: int = 3
    model_ids: tuple[str, ...] | None = None
    model_servers: tuple[ModelServer, ...] | None = None
    subagent_graph: Literal["gym_gemma4", "qwen35", "repository"] = "gym_gemma4"
    manager_max_model_calls: int = WORKPLACE_MODEL_CALL_LIMIT
    subagent_max_model_calls: int = WORKPLACE_MODEL_CALL_LIMIT
    subagent_recursion_limit: int = WORKPLACE_SUBAGENT_RECURSION_LIMIT
    max_output_tokens: int | None = None
    kind: Literal["decomposer"] = field(init=False, default="decomposer")

    def __post_init__(self) -> None:
        for field_name in (
            "manager_max_model_calls",
            "subagent_max_model_calls",
            "subagent_recursion_limit",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{self.name}: {field_name} must be at least 1")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError(f"{self.name}: max_output_tokens must be at least 1")
        remote_fields = (
            self.manager_model_id,
            self.manager_proxy_port,
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
        if self.manager_sampling is not None and not self.requires_llm_proxy:
            raise ValueError(
                f"{self.name}: manager_sampling requires manager_backend=llm_proxy"
            )

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
    def requires_local_manager(self) -> bool:
        return self.manager_backend == "local_vllm"

    @property
    def remote_manager_extra_body(self) -> dict[str, object]:
        if self.manager_sampling is None:
            return {}
        sampling = self.manager_sampling
        value: dict[str, object] = {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "presence_penalty": sampling.presence_penalty,
            "repetition_penalty": sampling.repetition_penalty,
        }
        if self.max_output_tokens is not None:
            value["max_output_tokens"] = self.max_output_tokens
        if self.manager_reasoning_mode == "non_thinking":
            value.update(
                {
                    "include_reasoning": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
        elif self.manager_reasoning_mode == "thinking":
            value.update(
                {
                    "include_reasoning": True,
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "preserve_thinking": True,
                    },
                }
            )
        return value


@dataclass(frozen=True)
class SimpleExperiment:
    name: str
    checkpoint: Path | None
    backend: SimpleAgentBackend = "local_vllm"
    model_id: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    reasoning_effort: str | None = None
    thinking: bool = False
    num_gpus: int = 1
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    concurrency: int = 32
    max_model_len: int = 131072
    max_output_tokens: int | None = None
    gpu_memory_utilization: float = 0.90
    max_steps: int = WORKPLACE_MODEL_CALL_LIMIT
    gym_wait_timeout: int = 360
    tool_call_parser: str = "qwen3_xml"
    reasoning_parser: str | None = None
    gdn_prefill_backend: str | None = "triton"
    kind: Literal["simple"] = field(init=False, default="simple")

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError(f"{self.name}: max_output_tokens must be at least 1")
        if self.backend == "local_vllm":
            if self.checkpoint is None:
                raise ValueError(f"{self.name}: local_vllm requires checkpoint")
            if any(
                value is not None
                for value in (self.model_id, self.base_url, self.api_key_env)
            ):
                raise ValueError(
                    f"{self.name}: local_vllm cannot configure remote model fields"
                )
        elif self.backend == "openrouter":
            if self.checkpoint is not None:
                raise ValueError(f"{self.name}: openrouter cannot use checkpoint")
            if not self.model_id or not self.base_url or not self.api_key_env:
                raise ValueError(
                    f"{self.name}: openrouter requires model_id, base_url, and "
                    "api_key_env"
                )

    @property
    def requires_openrouter(self) -> bool:
        return self.backend == "openrouter"

    @property
    def remote_extra_body(self) -> dict[str, object]:
        if self.reasoning_effort is None:
            return {}
        return {"reasoning": {"effort": self.reasoning_effort}}

    @property
    def extra_body(self) -> dict[str, int | float]:
        value: dict[str, int | float] = {"top_k": self.top_k}
        if self.tool_call_parser != "gemma4":
            value.update(
                {
                    "min_p": self.min_p,
                    "presence_penalty": self.presence_penalty,
                    "repetition_penalty": self.repetition_penalty,
                }
            )
        return value

    @property
    def chat_template_kwargs(self) -> dict[str, bool]:
        return {
            "enable_thinking": self.thinking,
            **({"preserve_thinking": True} if self.thinking else {}),
        }

    @property
    def chat_template_kwargs_b64(self) -> str:
        payload = json.dumps(self.chat_template_kwargs, separators=(",", ":"))
        return base64.b64encode(payload.encode()).decode()

    @property
    def effective_reasoning_parser(self) -> str | None:
        if self.reasoning_parser is not None:
            return self.reasoning_parser
        return "qwen3" if self.thinking else None


Experiment = DecomposerExperiment | SimpleExperiment


def validate_run_purpose(purpose: str) -> RunPurpose:
    if purpose not in RUN_PURPOSES:
        expected = ", ".join(RUN_PURPOSES)
        raise ValueError(
            f"Unknown run purpose {purpose!r}; expected one of: {expected}"
        )
    return cast(RunPurpose, purpose)


def validate_purpose_for_experiment(experiment: Experiment, purpose: str) -> RunPurpose:
    validated = validate_run_purpose(purpose)
    if validated == "trace-generation" and isinstance(experiment, SimpleExperiment):
        raise ValueError(
            "trace-generation is only supported for Decomposer experiments; "
            "simple-agent traces are not accepted by the SFT adapter"
        )
    return validated


def decomposer_prompt_profile(
    purpose: str,
    requested: DecomposerPromptProfile | None = None,
    evaluation_default: DecomposerPromptProfile = "student",
) -> DecomposerPromptProfile:
    validated = validate_run_purpose(purpose)
    if requested is not None:
        return requested
    return "teacher" if validated == "trace-generation" else evaluation_default


MODELS = (
    ModelServer(
        "google/gemma-4-E2B-it",
        HF_HUB_ROOT
        / "models--google--gemma-4-E2B-it"
        / "snapshots"
        / "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        8020,
        0,
        0.30,
        0,
    ),
    ModelServer(
        "google/gemma-4-E4B-it",
        HF_HUB_ROOT
        / "models--google--gemma-4-E4B-it"
        / "snapshots"
        / "ee0ef6023621cff504d758262d4e04895a5af4a2",
        8021,
        0,
        0.60,
        1,
    ),
    ModelServer(
        "google/gemma-4-12B-it",
        HF_HUB_ROOT
        / "models--google--gemma-4-12B-it"
        / "snapshots"
        / "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        8022,
        1,
        0.88,
        0,
    ),
    ModelServer(
        "google/gemma-4-26B-A4B-it",
        HF_HUB_ROOT
        / "models--google--gemma-4-26B-A4B-it"
        / "snapshots"
        / "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        8023,
        2,
        0.88,
        0,
    ),
    ModelServer(
        "google/gemma-4-31B-it",
        HF_HUB_ROOT
        / "models--google--gemma-4-31B-it"
        / "snapshots"
        / "842da3794eaa0b77d5f08bae87a17459d91ff475",
        8027,
        3,
        0.88,
        0,
    ),
)

QWEN35_08B_BASE = (
    HF_HUB_ROOT
    / "models--Qwen--Qwen3.5-0.8B"
    / "snapshots"
    / "2fc06364715b967f1860aea9cf38778875588b17"
)
QWEN35_2B_BASE = (
    HF_HUB_ROOT
    / "models--Qwen--Qwen3.5-2B"
    / "snapshots"
    / "15852e8c16360a2fea060d615a32b45270f8a8fc"
)
QWEN35_4B_BASE = (
    HF_HUB_ROOT
    / "models--Qwen--Qwen3.5-4B"
    / "snapshots"
    / "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
QWEN35_9B_BASE = (
    HF_HUB_ROOT
    / "models--Qwen--Qwen3.5-9B"
    / "snapshots"
    / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)

WORKPLACE_E4B_SFT_MODEL_ID = "decomposer/gemma4-e4b-sft-deepseek-e4b-v1-8k"
WORKPLACE_E4B_SFT_FINAL = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-full-4gpu"
    / "final"
)
WORKPLACE_E4B_SFT_VLLM = WORKPLACE_E4B_SFT_FINAL.with_name("final-vllm")

WORKPLACE_E4B_SFT_MIXED_V3_MODEL_ID = "decomposer/gemma4-e4b-sft-mixed-v3"
WORKPLACE_E4B_SFT_MIXED_V3_VLLM = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints/gemma4-e4b-nonthinking-4gpu-mixed-v3/final-vllm"
)

WORKPLACE_E2B_SFT_MIXED_V3_MODEL_ID = "decomposer/gemma4-e2b-sft-mixed-v3"
# The 14T volume holds the checkpoints; ARTIFACTS_ROOT is a separate filesystem.
WORKPLACE_E2B_SFT_MIXED_V3_VLLM = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints"
    "/gemma4-e2b-nonthinking-4gpu-mixed-v3/final-vllm"
)

WORKPLACE_QWEN35_SFT_MIXED_V3_MODEL_ID = "decomposer/qwen35-4b-sft-mixed-v3"
# Qwen needs no final-vllm re-export; that pass rebuilds Gemma-4's k_norm tensors.
WORKPLACE_QWEN35_SFT_MIXED_V3_FINAL = Path(
    "/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints"
    "/qwen35-4b-nonthinking-mixed-v3-8gpu/final"
)

WORKPLACE_QWEN35_4B_SFT_MODEL_ID = "decomposer/qwen35-4b-sft-workplace-v1-3765-32k"
WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID = "decomposer/qwen35-4b-base-manager"
WORKPLACE_QWEN35_4B_SFT_FINAL = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "qwen35-4b-nonthinking-workplace-v1-3765-32k-full-4gpu"
    / "final"
)
WORKPLACE_QWEN35_4B_MIXED_SFT_MODEL_ID = (
    "decomposer/qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k"
)
WORKPLACE_QWEN35_4B_MIXED_SFT_FINAL = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "qwen35-4b-nonthinking-mixed-v1-partial-3983f605-327-32k-full-4gpu"
    / "final"
)
WORKPLACE_QWEN35_4B_FINAL_MIXED_SFT_MODEL_ID = (
    "decomposer/qwen35-4b-sft-mixed-v1-final-493c24c4-404"
)
WORKPLACE_QWEN35_4B_FINAL_MIXED_SFT_FINAL = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "qwen35-4b-nonthinking-mixed-v1-final-493c24c4-404-32k-full-4gpu"
    / "final"
)
WORKPLACE_QWEN35_4B_FILTERED_SFT_MODEL_ID = (
    "decomposer/qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279"
)
WORKPLACE_QWEN35_4B_FILTERED_SFT_FINAL = (
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
WORKPLACE_QWEN35_4B_GAIA2_SFT_MODEL_ID = (
    "decomposer/qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2"
)
WORKPLACE_QWEN35_4B_GAIA2_SFT_FINAL = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / ("qwen35-4b-nonthinking-mixed-v2-493c24c4-gaia2-110-n3-" "filtered-32k-full-4gpu")
    / "final"
)
WORKPLACE_QWEN35_4B_TOOLATHLON_ONLY_SFT_MODEL_ID = (
    "decomposer/qwen35-4b-sft-toolathlon-only-v1-493c24c4-"
    "teacher-prompt-filtered-32k"
)
WORKPLACE_QWEN35_4B_TOOLATHLON_ONLY_SFT_FINAL = (
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
WORKPLACE_QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_MODEL_ID = (
    "decomposer/qwen35-4b-sft-gaia2-execution-only-v1-110-n10-"
    "teacher-prompt-r1-balanced-32k"
)
WORKPLACE_QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_FINAL = (
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

_QWEN36_NON_THINKING_SAMPLING = qwen36_non_thinking_sampling()
_QWEN36_THINKING_SAMPLING = qwen36_thinking_sampling()

# Known upstream fault for every Qwen3.6 llm_proxy profile below: the Responses
# deployment returns reasoning, but discards reasoning items that the harness
# sends back in later inputs. We save output reasoning, yet these experiments
# remain capture-only and are not clean comparisons with replayed DeepSeek/Gemma.
DECOMPOSER_EXPERIMENTS = (
    DecomposerExperiment(
        name="gemma4-26b-a4b-thinking-gemma4-e4b-thinking-text-defaults",
        gym_config_filename=(
            "workplace_assistant_gemma4_26b_a4b_thinking_"
            "gemma4_e4b_thinking_text_defaults.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        max_model_len=131072,
        # max_num_seqs=16,
        subagent_graph="repository",
        model_servers=(
            replace(
                MODELS[3],
                gpu=0,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=True,
            ),
            replace(
                MODELS[1],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=True,
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "gemma4-31b-thinking-teacher-"
            "gemma4-31b-non-thinking-text-defaults"
        ),
        gym_config_filename=(
            "workplace_assistant_gemma4_31b_thinking_teacher_"
            "gemma4_31b_non_thinking_text_defaults.yaml"
        ),
        manager_backend="local_vllm",
        evaluation_prompt_profile="teacher",
        concurrency=16,
        num_gpus=1,
        max_model_len=131072,
        max_num_seqs=64,
        subagent_graph="repository",
        model_ids=("google/gemma-4-31B-it",),
    ),
    DecomposerExperiment(
        name=(
            "gemma4-26b-a4b-thinking-teacher-"
            "gemma4-26b-a4b-non-thinking-text-defaults"
        ),
        gym_config_filename=(
            "workplace_assistant_gemma4_26b_a4b_thinking_teacher_"
            "gemma4_26b_a4b_non_thinking_text_defaults.yaml"
        ),
        manager_backend="local_vllm",
        evaluation_prompt_profile="teacher",
        num_gpus=1,
        max_model_len=131072,
        # max_num_seqs=16,
        subagent_graph="repository",
        model_ids=("google/gemma-4-26B-A4B-it",),
    ),
    # Capture-only upstream fault: this Qwen3.6 proxy deployment discards
    # reasoning items replayed in later Responses inputs.
    DecomposerExperiment(
        name=(
            "qwen36-35b-a3b-non-thinking-teacher-"
            "qwen35-4b-non-thinking-text-defaults"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen36_35b_a3b_non_thinking_teacher_"
            "qwen35_4b_non_thinking_text_defaults.yaml"
        ),
        manager_backend="llm_proxy",
        manager_model_id="Qwen/Qwen3.6-35B-A3B-FP8",
        manager_proxy_port=8142,
        manager_upstream_url_env="LLM_PROXY_URL",
        manager_api_key_env="LLM_PROXY_MASTER_KEY",
        manager_response_tool_parser="qwen3_xml",
        manager_reasoning_mode="non_thinking",
        manager_verify_tls=False,
        manager_sampling=_QWEN36_NON_THINKING_SAMPLING,
        evaluation_prompt_profile="teacher",
        concurrency=16,
        num_gpus=1,
        max_model_len=131072,
        # max_num_seqs=16,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
            ),
        ),
    ),
    # Capture-only upstream fault: output reasoning is saved and replay is sent,
    # but this Qwen3.6 proxy deployment discards the replayed input items.
    DecomposerExperiment(
        name=(
            "qwen36-35b-a3b-thinking-teacher-"
            "qwen35-4b-non-thinking-text-defaults"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen36_35b_a3b_thinking_teacher_"
            "qwen35_4b_non_thinking_text_defaults.yaml"
        ),
        manager_backend="llm_proxy",
        manager_model_id="Qwen/Qwen3.6-35B-A3B-FP8",
        manager_proxy_port=8142,
        manager_upstream_url_env="LLM_PROXY_URL",
        manager_api_key_env="LLM_PROXY_MASTER_KEY",
        manager_response_tool_parser="qwen3_xml",
        manager_reasoning_mode="thinking",
        manager_verify_tls=False,
        manager_sampling=_QWEN36_THINKING_SAMPLING,
        evaluation_prompt_profile="teacher",
        concurrency=16,
        num_gpus=1,
        max_model_len=131072,
        # max_num_seqs=16,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
            ),
        ),
    ),
    DecomposerExperiment(
        name=("qwen35-4b-base-non-thinking-" "qwen35-4b-non-thinking"),
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_base_non_thinking_"
            "qwen35_4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID,
                QWEN35_4B_BASE,
                8026,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                1,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "qwen35-4b-sft-workplace-v1-3765-32k-non-thinking-" "qwen35-4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_sft_workplace_v1_3765_32k_"
            "non_thinking_qwen35_4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_SFT_MODEL_ID,
                WORKPLACE_QWEN35_4B_SFT_FINAL,
                8026,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                1,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-"
            "qwen35-4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_sft_mixed_v1_partial_3983f605_"
            "327_32k_non_thinking_qwen35_4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_MIXED_SFT_MODEL_ID,
                WORKPLACE_QWEN35_4B_MIXED_SFT_FINAL,
                8026,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                1,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-"
            "qwen35-4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_sft_mixed_v1_final_493c24c4_404_"
            "non_thinking_qwen35_4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_FINAL_MIXED_SFT_MODEL_ID,
                WORKPLACE_QWEN35_4B_FINAL_MIXED_SFT_FINAL,
                8026,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                1,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-"
            "non-thinking-qwen35-4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_sft_mixed_v1_final_493c24c4_404_"
            "filtered_p1_s279_non_thinking_qwen35_4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_FILTERED_SFT_MODEL_ID,
                WORKPLACE_QWEN35_4B_FILTERED_SFT_FINAL,
                8026,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                1,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-"
            "non-thinking-qwen35-4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_sft_mixed_v2_493c24c4_gaia2_"
            "110_n3_filtered_p2_non_thinking_qwen35_4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_GAIA2_SFT_MODEL_ID,
                WORKPLACE_QWEN35_4B_GAIA2_SFT_FINAL,
                8026,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                1,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "qwen35-4b-sft-toolathlon-only-v1-493c24c4-teacher-prompt-"
            "filtered-32k-non-thinking-qwen35-4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_sft_toolathlon_only_v1_493c24c4_"
            "teacher_prompt_filtered_32k_non_thinking_qwen35_4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_TOOLATHLON_ONLY_SFT_MODEL_ID,
                WORKPLACE_QWEN35_4B_TOOLATHLON_ONLY_SFT_FINAL,
                8026,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                1,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "qwen35-4b-sft-gaia2-execution-only-v1-110-n10-teacher-prompt-"
            "r1-balanced-32k-non-thinking-qwen35-4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_sft_gaia2_execution_only_v1_110_"
            "n10_teacher_prompt_r1_balanced_32k_non_thinking_"
            "qwen35_4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_MODEL_ID,
                WORKPLACE_QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_FINAL,
                8026,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                1,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
        ),
    ),
    DecomposerExperiment(
        name="deepseek-v4-flash-0731-qwen35-4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_deepseek_v4_flash_0731_qwen35_4b_non_thinking.yaml"
        ),
        num_gpus=1,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
            ),
        ),
    ),
    # Capture-only upstream fault: the service-default Qwen3.6 proxy path also
    # discards input reasoning items, so it is not a full-replay comparison.
    DecomposerExperiment(
        name="qwen36-35b-a3b-teacher-qwen35-4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_qwen36_35b_a3b_teacher_" "qwen35_4b_non_thinking.yaml"
        ),
        manager_backend="llm_proxy",
        manager_model_id="Qwen/Qwen3.6-35B-A3B-FP8",
        manager_proxy_port=8142,
        manager_upstream_url_env="LLM_PROXY_URL",
        manager_api_key_env="LLM_PROXY_MASTER_KEY",
        manager_response_tool_parser="qwen3_xml",
        manager_reasoning_mode="service_default",
        manager_verify_tls=False,
        concurrency=16,
        num_gpus=1,
        max_model_len=131072,
        subagent_graph="qwen35",
        model_servers=(
            ModelServer(
                "Qwen/Qwen3.5-4B",
                QWEN35_4B_BASE,
                8025,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
            ),
        ),
    ),
    DecomposerExperiment(
        name=("gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking"),
        gym_config_filename=(
            "workplace_assistant_gemma4_e4b_sft_deepseek_e4b_v1_8k_"
            "non_thinking_gemma4_e4b_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=2,
        model_servers=(
            ModelServer(
                WORKPLACE_E4B_SFT_MODEL_ID,
                WORKPLACE_E4B_SFT_VLLM,
                8024,
                0,
                0.90,
                0,
                thinking=False,
            ),
            replace(
                MODELS[1],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "gemma4-e2b-sft-mixed-v3-non-thinking-gemma4-26b-a4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_gemma4_e2b_sft_mixed_v3_"
            "non_thinking_gemma4_26b_a4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        # The checkpoint was trained on the teacher prompt, so it must be
        # evaluated under the same one.
        evaluation_prompt_profile="teacher",
        num_gpus=2,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_E2B_SFT_MIXED_V3_MODEL_ID,
                WORKPLACE_E2B_SFT_MIXED_V3_VLLM,
                8028,
                0,
                0.90,
                0,
                thinking=False,
            ),
            # MODELS[3] ships gpu=2; a two-GPU experiment must remap it.
            replace(
                MODELS[3],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "gemma4-e4b-sft-mixed-v3-non-thinking-gemma4-26b-a4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_gemma4_e4b_sft_mixed_v3_"
            "non_thinking_gemma4_26b_a4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        evaluation_prompt_profile="teacher",
        num_gpus=2,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_E4B_SFT_MIXED_V3_MODEL_ID,
                WORKPLACE_E4B_SFT_MIXED_V3_VLLM,
                8029,
                0,
                0.90,
                0,
                thinking=False,
            ),
            replace(
                MODELS[3],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    DecomposerExperiment(
        name=(
            "qwen35-4b-sft-mixed-v3-non-thinking-gemma4-26b-a4b-non-thinking"
        ),
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_sft_mixed_v3_"
            "non_thinking_gemma4_26b_a4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        # The checkpoint was trained on the teacher prompt, so evaluate under it.
        evaluation_prompt_profile="teacher",
        num_gpus=2,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_SFT_MIXED_V3_MODEL_ID,
                WORKPLACE_QWEN35_SFT_MIXED_V3_FINAL,
                8030,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                # Non-thinking Qwen3.5 closes the think block inside the prompt,
                # so the completion carries no tags for a reasoning parser.
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            replace(
                MODELS[3],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    DecomposerExperiment(
        name="qwen35-4b-sft-mixed-v3-non-thinking-gemma4-e4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_sft_mixed_v3_"
            "non_thinking_gemma4_e4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        # The checkpoint was trained on the teacher prompt, so evaluate under it.
        evaluation_prompt_profile="teacher",
        # Sixty-four saturated the single-worker langgraph event loop: every
        # waiting manager polls threads.get_history every five seconds, and the
        # polls started exceeding the httpx read timeout. Sixteen matches the
        # 26B-A4B baselines.
        concurrency=16,
        num_gpus=2,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_SFT_MIXED_V3_MODEL_ID,
                WORKPLACE_QWEN35_SFT_MIXED_V3_FINAL,
                8031,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                # Non-thinking Qwen3.5 closes the think block inside the prompt,
                # so the completion carries no tags for a reasoning parser.
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            # MODELS[1] ships gpu 0, utilization 0.60, wave 1 and thinking on;
            # a two-GPU non-thinking pairing overrides all four.
            replace(
                MODELS[1],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    # Untuned-manager baselines: the raw Qwen3.5-4B snapshot orchestrating each
    # frozen Gemma worker, under both prompt profiles.
    DecomposerExperiment(
        name="qwen35-4b-base-non-thinking-gemma4-e2b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_base_non_thinking_gemma4_e2b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        evaluation_prompt_profile="student",
        concurrency=16,
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID,
                QWEN35_4B_BASE,
                8060,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            replace(
                MODELS[0],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    DecomposerExperiment(
        name="qwen35-4b-base-non-thinking-teacher-gemma4-e2b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_base_non_thinking_teacher_gemma4_e2b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        evaluation_prompt_profile="teacher",
        concurrency=16,
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID,
                QWEN35_4B_BASE,
                8061,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            replace(
                MODELS[0],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    DecomposerExperiment(
        name="qwen35-4b-base-non-thinking-gemma4-e4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_base_non_thinking_gemma4_e4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        evaluation_prompt_profile="student",
        concurrency=16,
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID,
                QWEN35_4B_BASE,
                8062,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            replace(
                MODELS[1],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    DecomposerExperiment(
        name="qwen35-4b-base-non-thinking-teacher-gemma4-e4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_base_non_thinking_teacher_gemma4_e4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        evaluation_prompt_profile="teacher",
        concurrency=16,
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID,
                QWEN35_4B_BASE,
                8063,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            replace(
                MODELS[1],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    DecomposerExperiment(
        name="qwen35-4b-base-non-thinking-gemma4-26b-a4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_base_non_thinking_gemma4_26b_a4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        evaluation_prompt_profile="student",
        concurrency=16,
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID,
                QWEN35_4B_BASE,
                8064,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            replace(
                MODELS[3],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    DecomposerExperiment(
        name="qwen35-4b-base-non-thinking-teacher-gemma4-26b-a4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_qwen35_4b_base_non_thinking_teacher_gemma4_26b_a4b_non_thinking.yaml"
        ),
        manager_backend="local_vllm",
        evaluation_prompt_profile="teacher",
        concurrency=16,
        num_gpus=2,
        max_model_len=131072,
        subagent_graph="repository",
        model_servers=(
            ModelServer(
                WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID,
                QWEN35_4B_BASE,
                8065,
                0,
                0.90,
                0,
                thinking=False,
                tool_call_parser="qwen3_xml",
                reasoning_parser=None,
                gdn_prefill_backend="triton",
                dtype="bfloat16",
            ),
            replace(
                MODELS[3],
                gpu=1,
                gpu_memory_utilization=0.90,
                startup_wave=0,
                thinking=False,
            ),
        ),
    ),
    DecomposerExperiment(
        name="gemma4-e4b-it-non-thinking-gemma4-e4b-thinking",
        gym_config_filename=(
            "workplace_assistant_gemma4_e4b_non_thinking_gemma4_e4b_thinking.yaml"
        ),
        manager_backend="local_vllm",
        num_gpus=1,
        model_ids=("google/gemma-4-E4B-it",),
    ),
    DecomposerExperiment(
        name="deepseek-v4-flash-0731-gemma4-all",
        gym_config_filename="workplace_assistant_deepseek_v4_flash_0731.yaml",
    ),
    DecomposerExperiment(
        name="glm-5-2-gemma4-all",
        gym_config_filename="workplace_assistant_glm_5_2.yaml",
    ),
    DecomposerExperiment(
        name="deepseek-v4-flash-0731-gemma4-26b-a4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_deepseek_v4_flash_0731_"
            "gemma4_26b_a4b_non_thinking.yaml"
        ),
        num_gpus=1,
        model_ids=("google/gemma-4-26B-A4B-it",),
    ),
    DecomposerExperiment(
        name="deepseek-v4-flash-0731-teacher-gemma4-e2b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_deepseek_v4_flash_0731_teacher_"
            "gemma4_e2b_non_thinking.yaml"
        ),
        evaluation_prompt_profile="teacher",
        num_gpus=1,
        max_model_len=131072,
        max_num_seqs=64,
        subagent_graph="repository",
        model_ids=("google/gemma-4-E2B-it",),
    ),
    DecomposerExperiment(
        name="deepseek-v4-flash-0731-teacher-gemma4-e4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_deepseek_v4_flash_0731_teacher_"
            "gemma4_e4b_non_thinking.yaml"
        ),
        evaluation_prompt_profile="teacher",
        num_gpus=1,
        max_model_len=131072,
        max_num_seqs=64,
        subagent_graph="repository",
        model_ids=("google/gemma-4-E4B-it",),
    ),
    DecomposerExperiment(
        name="deepseek-v4-flash-0731-teacher-gemma4-26b-a4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_deepseek_v4_flash_0731_teacher_"
            "gemma4_26b_a4b_non_thinking.yaml"
        ),
        evaluation_prompt_profile="teacher",
        num_gpus=1,
        max_model_len=131072,
        # max_num_seqs=16,
        subagent_graph="repository",
        model_ids=("google/gemma-4-26B-A4B-it",),
    ),
    DecomposerExperiment(
        name="deepseek-v4-flash-0731-gemma4-e4b-thinking",
        gym_config_filename=(
            "workplace_assistant_deepseek_v4_flash_0731_gemma4_e4b_thinking.yaml"
        ),
        num_gpus=1,
        model_ids=("google/gemma-4-E4B-it",),
    ),
    DecomposerExperiment(
        name="glm-5-2-gemma4-26b-a4b-non-thinking",
        gym_config_filename=(
            "workplace_assistant_glm_5_2_gemma4_26b_a4b_non_thinking.yaml"
        ),
        num_gpus=1,
        model_ids=("google/gemma-4-26B-A4B-it",),
    ),
)

GEMMA4_E2B_BASE = MODELS[0].snapshot
GEMMA4_E4B_BASE = MODELS[1].snapshot
GEMMA4_12B_BASE = MODELS[2].snapshot
GEMMA4_26B_A4B_BASE = MODELS[3].snapshot
GEMMA4_31B_BASE = (
    HF_HUB_ROOT
    / "models--google--gemma-4-31B-it"
    / "snapshots"
    / "842da3794eaa0b77d5f08bae87a17459d91ff475"
)


def _gemma4_simple_experiments() -> list[SimpleExperiment]:
    checkpoints = (
        ("e2b", GEMMA4_E2B_BASE),
        ("e4b", GEMMA4_E4B_BASE),
        ("12b", GEMMA4_12B_BASE),
        ("31b", GEMMA4_31B_BASE),
        ("26b-a4b", GEMMA4_26B_A4B_BASE),
    )
    return [
        SimpleExperiment(
            name=f"gemma4-{size}-it-{mode}",
            checkpoint=checkpoint,
            thinking=thinking,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            tool_call_parser="gemma4",
            reasoning_parser="gemma4",
            gdn_prefill_backend=None,
        )
        for size, checkpoint in checkpoints
        for mode, thinking in (("non-thinking", False), ("thinking", True))
    ]


def _qwen35_simple_experiment(
    name: str,
    checkpoint: Path,
    *,
    thinking: bool = False,
) -> SimpleExperiment:
    sampling = qwen35_general_sampling(thinking=thinking)
    return SimpleExperiment(
        name=name,
        checkpoint=checkpoint,
        thinking=thinking,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        presence_penalty=sampling.presence_penalty,
        repetition_penalty=sampling.repetition_penalty,
    )


def _simple_experiments() -> tuple[SimpleExperiment, ...]:
    checkpoint_profiles = (
        "q35-2b-grpo-nightly-step55",
        "q35-4b-gaia2-grpo-base-inband-t09-s21-0895",
        "q35-4b-gaia2-grpo-v23-binres-s15",
        "q35-4b-gaia2-grpo-v23-binres-s27",
        "q35-4b-gaia2-grpo-v23-binres-s45",
    )
    experiments = [
        _qwen35_simple_experiment(
            f"{name}{suffix}",
            CHECKPOINTS_ROOT / name,
            thinking=thinking,
        )
        for name in checkpoint_profiles
        for suffix, thinking in (("", False), ("-thinking", True))
    ]
    experiments.extend(
        _qwen35_simple_experiment(name, CHECKPOINTS_ROOT / name)
        for name in (
            "q35-4b-gaia2-grpo-v32-kl0temp1-s30",
            "q35-4b-gaia2-grpo-v32-kl0temp1-s45",
        )
    )
    experiments.extend(
        (
            _qwen35_simple_experiment("qwen35-0.8b-base-non-thinking", QWEN35_08B_BASE),
            _qwen35_simple_experiment(
                "qwen35-0.8b-base-thinking", QWEN35_08B_BASE, thinking=True
            ),
            _qwen35_simple_experiment("qwen35-2b-base-non-thinking", QWEN35_2B_BASE),
            _qwen35_simple_experiment(
                "qwen35-2b-base-thinking", QWEN35_2B_BASE, thinking=True
            ),
            _qwen35_simple_experiment("qwen35-4b-base-non-thinking", QWEN35_4B_BASE),
            _qwen35_simple_experiment(
                "qwen35-4b-base-thinking", QWEN35_4B_BASE, thinking=True
            ),
            _qwen35_simple_experiment("qwen35-9b-base-non-thinking", QWEN35_9B_BASE),
            SimpleExperiment(
                name="deepseek-v4-flash-0731",
                checkpoint=None,
                backend="openrouter",
                model_id="deepseek/deepseek-v4-flash-0731",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY_DECOMPOSER",
                reasoning_effort="max",
                temperature=1.0,
                top_p=1.0,
                concurrency=8,
            ),
        )
    )
    experiments.extend(
        (
            SimpleExperiment(
                name="gemma4-e4b-thinking-simple-text-defaults",
                checkpoint=GEMMA4_E4B_BASE,
                thinking=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
                max_model_len=131072,
                max_steps=100,
                tool_call_parser="gemma4",
                reasoning_parser="gemma4",
                gdn_prefill_backend=None,
            ),
            _qwen35_simple_experiment(
                "qwen35-4b-non-thinking-simple-general-text-defaults",
                QWEN35_4B_BASE,
            ),
        )
    )
    return tuple([*experiments, *_gemma4_simple_experiments()])


SIMPLE_EXPERIMENTS = _simple_experiments()
ALL_EXPERIMENTS: tuple[Experiment, ...] = (
    *DECOMPOSER_EXPERIMENTS,
    *SIMPLE_EXPERIMENTS,
)
EXPERIMENTS = {experiment.name: experiment for experiment in ALL_EXPERIMENTS}
if len(EXPERIMENTS) != len(ALL_EXPERIMENTS):
    raise ValueError("Workplace Assistant experiment names must be globally unique")


def get_experiment(name: str) -> Experiment:
    try:
        return EXPERIMENTS[name]
    except KeyError as error:
        expected = ", ".join(EXPERIMENTS)
        raise ValueError(
            f"Unknown Workplace Assistant experiment {name!r}; expected: {expected}"
        ) from error


def collect_experiments(
    names: tuple[str, ...] = (),
    filters: tuple[str, ...] = (),
    *,
    default_all: bool = True,
) -> list[Experiment]:
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown Workplace Assistant experiments: {unknown}")
    if not names and not filters:
        return list(ALL_EXPERIMENTS) if default_all else []
    selected_names = set(names)
    selected_names.update(
        experiment.name
        for experiment in ALL_EXPERIMENTS
        if any(pattern in experiment.name for pattern in filters)
    )
    return [
        experiment
        for experiment in ALL_EXPERIMENTS
        if experiment.name in selected_names
    ]


def models_for_experiment(
    experiment: DecomposerExperiment,
) -> tuple[ModelServer, ...]:
    if experiment.model_ids is not None and experiment.model_servers is not None:
        raise ValueError(
            f"{experiment.name} cannot set both model_ids and model_servers"
        )
    if experiment.model_servers is not None:
        selected = experiment.model_servers
    elif experiment.model_ids is None:
        selected = MODELS
    else:
        by_id = {model.model_id: model for model in MODELS}
        missing = [
            model_id for model_id in experiment.model_ids if model_id not in by_id
        ]
        if missing:
            raise ValueError(f"Unknown model IDs for {experiment.name}: {missing}")
        selected = tuple(by_id[model_id] for model_id in experiment.model_ids)
    model_ids = [model.model_id for model in selected]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError(f"Duplicate model IDs for {experiment.name}: {model_ids}")
    ports = [model.port for model in selected]
    if len(ports) != len(set(ports)):
        raise ValueError(f"Duplicate model ports for {experiment.name}: {ports}")
    if len(selected) == 1 and experiment.num_gpus == 1:
        selected = (replace(selected[0], gpu=0, startup_wave=0),)
    invalid = [model.model_id for model in selected if model.gpu >= experiment.num_gpus]
    if invalid:
        raise ValueError(
            f"Models assigned outside {experiment.num_gpus} GPUs for "
            f"{experiment.name}: {invalid}"
        )
    return selected


def validate_num_repeats(value: int) -> int:
    if value < 1:
        raise ValueError("num_repeats must be at least 1")
    return value


def run_name(
    experiment: Experiment,
    num_repeats: int = 1,
    *,
    prompt_profile: DecomposerPromptProfile | None = None,
) -> str:
    validate_num_repeats(num_repeats)
    if isinstance(experiment, SimpleExperiment):
        call_identity = f"calls{experiment.max_steps}"
    else:
        call_identity = (
            f"managercalls{experiment.manager_max_model_calls}-"
            f"subagentcalls{experiment.subagent_max_model_calls}"
        )
    name = f"{experiment.name}-{call_identity}"
    if num_repeats != 1:
        name = f"{name}-n{num_repeats}"
    return name if prompt_profile is None else f"{name}-prompt-{prompt_profile}"


def output_dir(
    experiment: Experiment,
    split: str,
    num_repeats: int = 1,
    limit: int | None = None,
    *,
    purpose: RunPurpose,
    prompt_profile: DecomposerPromptProfile | None = None,
) -> Path:
    validate_split(split)
    validate_purpose_for_experiment(experiment, purpose)
    base = RESULTS_ROOT / split
    if purpose == "evaluation" and isinstance(experiment, DecomposerExperiment):
        base /= "evaluation"
    base /= run_name(experiment, num_repeats, prompt_profile=prompt_profile)
    return base if limit is None else base / f"smoke_{limit}"


def completion_marker(
    experiment: Experiment,
    split: str,
    num_repeats: int = 1,
    limit: int | None = None,
    *,
    purpose: RunPurpose,
    prompt_profile: DecomposerPromptProfile | None = None,
) -> Path:
    return (
        output_dir(
            experiment,
            split,
            num_repeats,
            limit,
            purpose=purpose,
            prompt_profile=prompt_profile,
        )
        / ".eval_done.json"
    )


def job_description(
    experiment: Experiment,
    split: str,
    num_repeats: int = 1,
    limit: int | None = None,
    *,
    purpose: RunPurpose,
    prompt_profile: DecomposerPromptProfile | None = None,
) -> str:
    validate_purpose_for_experiment(experiment, purpose)
    identity = run_name(experiment, num_repeats, prompt_profile=prompt_profile)
    if limit is not None:
        identity = f"{identity}-smoke-{limit}"
    prefix = f"workplace-assistant-{split}"
    if purpose == "evaluation" and isinstance(experiment, DecomposerExperiment):
        prefix += "-evaluation"
    return f"{prefix} {experiment.kind}-agent {identity}"
