"""Experiment registry for the tau2 gym.

An experiment fixes the manager (backend, model, sampling) and its
default task pool. ``run.py`` builds the Gym config from it at run time and writes
the materialized config into the run directory, so this registry is the only place
an experiment is described.

Subagents are Qwen3.5-4B, served by a local vLLM or reached through the shared LLM
proxy (``run.py --subagent-backend``). By default they are non-thinking; an experiment
may set their sampling (``subagent_sampling``), otherwise they use Qwen3.5's general
non-thinking preset.

Newer experiments name presets from ``src/decomposer/models.py`` instead
(``manager_preset``, ``subagent_preset``; bridged by ``gyms/tau2_gym/model_presets.py``): the
preset is then the only source of that role's sampling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from gyms.tau2_gym.model_presets import (
    QWEN35_UNLOOPED_THINKING_PRESET,
    QWEN38_FLASH_NON_THINKING_PRESET,
    SUBAGENT_PRESET_GRAPHS,
    is_thinking,
    manager_responses_body,
    upstream_model_id,
)
from gyms.qwen_sampling import (
    QWEN35_GENERAL_NON_THINKING,
    QWEN35_UNLOOPED_NON_THINKING,
    QWEN38_NON_THINKING,
    QWEN38_TEACHER_THINKING,
    QWEN38_THINKING,
    QwenSamplingParams,
    SubagentSampling,
)

ManagerBackend = Literal["openrouter", "llm_proxy", "local_vllm"]
ReasoningMode = Literal["service_default", "non_thinking", "thinking"]
# Qwen3.8-Flash-Next on the proxy also accepts "none" (= non-thinking) and defaults to "xhigh".
ReasoningEffort = Literal["low", "medium", "xhigh"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY_DECOMPOSER"
LLM_PROXY_URL_ENV = "LLM_PROXY_URL"
LLM_PROXY_API_KEY_ENV = "LLM_PROXY_MASTER_KEY"

# The id and description the SFT releases stamp into the `new` tool's type table
# (policy.subagent_types in sft/specs). Using them here too means the teacher,
# the SFT data and student evaluation all see one tool schema.
SUBAGENT_TYPE_ID = "subagent_non_thinking"
SUBAGENT_DESCRIPTION = "General-purpose tool-calling agent with access to the environment tools."
SUBAGENT_ASSISTANT_ID = "qwen35_4b_non_thinking"

# The model the subagents request, passed from run.py to the LangGraph server through
# SUBAGENT_MODEL_ENV. The shared proxy serves the unlooped Qwen3.5-4B; a local subagent
# vLLM can only serve the original weights, under their own name.
SUBAGENT_MODEL_ENV = "TAU2_GYM_SUBAGENT_MODEL_ID"
DEFAULT_SUBAGENT_MODEL_ID = "Qwen/Qwen3.5-4B-unlooped"
LOCAL_SUBAGENT_MODEL_ID = "Qwen/Qwen3.5-4B"

QWEN38_FLASH_MODEL_ID = "Qwen/Qwen3.8-Flash-Next-NVFP4"
DEEPSEEK_V4_FLASH_MODEL_ID = "deepseek/deepseek-v4-flash-0731"

HF_HUB_ROOT = Path("/home/sukhorukov/.cache/huggingface/hub")
QWEN35_4B_BASE = (
    HF_HUB_ROOT
    / "models--Qwen--Qwen3.5-4B"
    / "snapshots"
    / "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
# The 14T volume holds the SFT checkpoints; the artifacts root is a separate filesystem.
SFT_CHECKPOINTS_ROOT = Path("/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/checkpoints")

TRAIN_POOL = "decomposer_train_v2"
EVAL_POOL = "decomposer_eval_v1"
# Every runnable tasks_hard.json task outside the held-out sets (task_pools/build_pool.py).
BROAD_POOL = "decomposer_broad_v1"


@dataclass(frozen=True)
class Tau2Experiment:
    name: str
    description: str
    manager_backend: ManagerBackend
    # openrouter/llm_proxy: the upstream model id. local_vllm: the served model name.
    manager_model_id: str
    # Default pool; `run.py --pool` overrides it, and the run name records the pool.
    pool: str
    manager_reasoning_mode: ReasoningMode = "service_default"
    # llm_proxy thinking managers only: `reasoning.effort` on the Responses API; None is "xhigh".
    manager_reasoning_effort: ReasoningEffort | None = None
    manager_sampling: QwenSamplingParams | None = None
    # openrouter only: provider request fields, e.g. reasoning effort.
    manager_extra_body: Mapping[str, Any] = field(default_factory=dict)
    # local_vllm only. None means the checkpoint is supplied at run time
    # (`run.py --manager-checkpoint`).
    manager_checkpoint: Path | None = None
    # local_vllm only: Gym records prompt/generation token ids and logprobs on every
    # manager turn (vllm_model.return_token_id_information).
    return_token_ids: bool = False
    # None keeps Qwen3.5's general non-thinking preset (presence penalty 1.5, no cap).
    subagent_sampling: SubagentSampling | None = None
    # The upstream renders replayed reasoning items into the manager's prompt. Verified
    # for Qwen3.8-Flash-Next on the shared proxy's Responses API (2026-09-25: a replayed
    # reasoning item raised the follow-up input from 83 to 158 tokens); recorded in
    # run_status so traces say whether the teacher saw its earlier reasoning.
    upstream_replays_reasoning: bool = False
    concurrency: int = 16
    manager_max_model_calls: int = 100
    subagent_recursion_limit: int = 1000
    manager_max_model_len: int = 131072
    # models.py presets. A manager preset replaces manager_sampling and the reasoning
    # effort. A subagent preset makes the subagents the preset's own create_model()
    # graph, which reaches the proxy itself (no local subagent proxy).
    manager_preset: str | None = None
    subagent_preset: str | None = None
    # The entry the manager's `new` tool lists.
    subagent_type_id: str = SUBAGENT_TYPE_ID
    subagent_assistant_id: str = SUBAGENT_ASSISTANT_ID
    subagent_description: str = SUBAGENT_DESCRIPTION

    def __post_init__(self) -> None:
        local = self.manager_backend == "local_vllm"
        if not local and (self.manager_checkpoint is not None or self.return_token_ids):
            raise ValueError(f"{self.name}: checkpoints and token ids need manager_backend=local_vllm")
        if self.manager_extra_body and self.manager_backend != "openrouter":
            raise ValueError(f"{self.name}: manager_extra_body is for openrouter only")
        if self.manager_backend in ("llm_proxy", "local_vllm"):
            if self.manager_sampling is None and self.manager_preset is None:
                raise ValueError(f"{self.name}: {self.manager_backend} needs manager_sampling")
            if self.manager_reasoning_mode == "service_default":
                raise ValueError(f"{self.name}: {self.manager_backend} needs an explicit reasoning mode")
        if self.manager_reasoning_effort is not None and (
            self.manager_backend != "llm_proxy" or self.manager_reasoning_mode != "thinking"
        ):
            raise ValueError(f"{self.name}: manager_reasoning_effort needs an llm_proxy thinking manager")
        if self.upstream_replays_reasoning and (
            self.manager_backend != "llm_proxy" or self.manager_reasoning_mode != "thinking"
        ):
            raise ValueError(f"{self.name}: upstream_replays_reasoning needs an llm_proxy thinking manager")
        if local and self.manager_reasoning_mode != "non_thinking":
            # The SFT student is trained non-thinking only (sft/model_support.py).
            raise ValueError(f"{self.name}: a local manager must be non_thinking")
        for name in ("concurrency", "manager_max_model_calls", "subagent_recursion_limit"):
            if getattr(self, name) < 1:
                raise ValueError(f"{self.name}: {name} must be at least 1")
        if self.manager_preset is not None:
            if self.manager_backend != "llm_proxy":
                raise ValueError(f"{self.name}: manager_preset needs manager_backend=llm_proxy")
            if self.manager_sampling is not None or self.manager_reasoning_effort is not None:
                raise ValueError(f"{self.name}: manager_preset replaces manager_sampling and the effort")
            if self.manager_model_id != upstream_model_id(self.manager_preset):
                raise ValueError(f"{self.name}: manager_model_id differs from {self.manager_preset}")
            mode = "thinking" if is_thinking(self.manager_preset) else "non_thinking"
            if self.manager_reasoning_mode != mode:
                raise ValueError(f"{self.name}: {self.manager_preset} is a {mode} preset")
        if self.subagent_preset is not None:
            if self.subagent_sampling is not None:
                raise ValueError(f"{self.name}: subagent_preset replaces subagent_sampling")
            if SUBAGENT_PRESET_GRAPHS.get(self.subagent_preset) != self.subagent_assistant_id:
                raise ValueError(f"{self.name}: no subagent graph {self.subagent_assistant_id!r} for {self.subagent_preset}")

    @property
    def chat_template_kwargs(self) -> dict[str, bool]:
        if self.manager_reasoning_mode == "thinking":
            return {"enable_thinking": True, "preserve_thinking": True}
        return {"enable_thinking": False, "preserve_thinking": False}

    @property
    def _reasoning_effort(self) -> str:
        if self.manager_reasoning_mode != "thinking":
            return "none"
        return self.manager_reasoning_effort or "xhigh"

    @property
    def manager_proxy_extra_body(self) -> dict[str, Any]:
        """Sampling for `gyms.remote_model_proxy --extra-body-json` (llm_proxy only).

        The proxy merges this server-side and overrides caller keys, so it is the one
        place the manager's sampling is decided. Gym's manager speaks the Responses API,
        where the shared proxy ignores ``chat_template_kwargs``: thinking is switched by
        ``reasoning.effort`` instead (measured on Qwen3.8-Flash-Next: "none" gives no
        reasoning items, the default is "xhigh"). ``chat_template_kwargs`` is kept for
        Chat Completions callers.
        """
        if self.manager_backend != "llm_proxy":
            return {}
        if self.manager_preset is not None:
            return manager_responses_body(self.manager_preset)
        if self.manager_sampling is None:
            return {}
        sampling = self.manager_sampling
        return {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "presence_penalty": sampling.presence_penalty,
            "repetition_penalty": sampling.repetition_penalty,
            "include_reasoning": self.manager_reasoning_mode == "thinking",
            "chat_template_kwargs": self.chat_template_kwargs,
            "reasoning": {"effort": self._reasoning_effort},
        }


def _qwen38_flash_teacher(
    reasoning: Literal["non_thinking", "thinking"], effort: ReasoningEffort | None = None
) -> Tau2Experiment:
    return Tau2Experiment(
        name=f"qwen38_flash_teacher_{reasoning}" + (f"_{effort}" if effort else ""),
        description=(
            f"Qwen3.8 Flash Next ({reasoning.replace('_', '-')}"
            + (f", effort {effort}" if effort else "")
            + ") manager with the shared Decomposer prompt: SFT traces."
        ),
        manager_backend="llm_proxy",
        manager_model_id=QWEN38_FLASH_MODEL_ID,
        pool=TRAIN_POOL,
        manager_reasoning_mode=reasoning,
        manager_sampling=QWEN38_THINKING if reasoning == "thinking" else QWEN38_NON_THINKING,
        manager_reasoning_effort=effort,
    )


EXPERIMENTS: tuple[Tau2Experiment, ...] = (
    Tau2Experiment(
        name="deepseek_v4_flash_teacher",
        description="DeepSeek-v4-flash (reasoning effort max) manager with the shared Decomposer prompt.",
        manager_backend="openrouter",
        manager_model_id=DEEPSEEK_V4_FLASH_MODEL_ID,
        pool=TRAIN_POOL,
        manager_extra_body={"reasoning": {"effort": "max"}},
        concurrency=4,
    ),
    _qwen38_flash_teacher("non_thinking"),
    _qwen38_flash_teacher("thinking"),
    # Shorter reasoning than the "xhigh" default: faster traces and shorter SFT targets.
    _qwen38_flash_teacher("thinking", effort="low"),
    # SFT teacher traces: effort low without the presence penalty, and subagents on the
    # unlooped model's own non-thinking settings (0.7/0.8/20, no penalties) with an 8192-token cap.
    Tau2Experiment(
        name="qwen38_flash_thinking_low_teacher_qwen35_4b_unlooped",
        description=(
            "Qwen3.8 Flash Next (thinking, effort low, no presence penalty) manager with the "
            "shared Decomposer prompt; Qwen3.5-4B-unlooped non-thinking subagents on their recommended "
            "sampling. SFT teacher traces."
        ),
        manager_backend="llm_proxy",
        manager_model_id=QWEN38_FLASH_MODEL_ID,
        pool=TRAIN_POOL,
        manager_reasoning_mode="thinking",
        manager_reasoning_effort="low",
        manager_sampling=QWEN38_TEACHER_THINKING,
        subagent_sampling=QWEN35_UNLOOPED_NON_THINKING,
        upstream_replays_reasoning=True,
    ),
    # Teacher traces with the models.py presets: Qwen3.8 Flash Next non-thinking as
    # the manager and Qwen3.5-4B-unlooped thinking subagents (no output cap), over the
    # broad pool.
    Tau2Experiment(
        name="qwen38_flash_non_thinking_teacher_qwen35_4b_unlooped_thinking",
        description=(
            "Qwen3.8 Flash Next non-thinking manager (models.py preset) with the teacher "
            "prompt; Qwen3.5-4B-unlooped thinking subagents (models.py preset). Teacher "
            "traces over the broad pool."
        ),
        manager_backend="llm_proxy",
        manager_model_id=QWEN38_FLASH_MODEL_ID,
        pool=BROAD_POOL,
        manager_reasoning_mode="non_thinking",
        manager_preset=QWEN38_FLASH_NON_THINKING_PRESET,
        subagent_preset=QWEN35_UNLOOPED_THINKING_PRESET,
        subagent_type_id="subagent_thinking",
        subagent_assistant_id=SUBAGENT_PRESET_GRAPHS[QWEN35_UNLOOPED_THINKING_PRESET],
    ),
    Tau2Experiment(
        name="qwen35_4b_base_student",
        description="Untuned Qwen3.5-4B manager with the shared Decomposer prompt.",
        manager_backend="local_vllm",
        manager_model_id="decomposer/qwen35-4b-base-manager",
        pool=EVAL_POOL,
        manager_reasoning_mode="non_thinking",
        manager_sampling=QWEN35_GENERAL_NON_THINKING,
        manager_checkpoint=QWEN35_4B_BASE,
    ),
    Tau2Experiment(
        name="qwen35_4b_sft_mixed_v3_student",
        description="Qwen3.5-4B manager SFT'd on the v5 student-prompt release (no tau2 data).",
        manager_backend="local_vllm",
        manager_model_id="decomposer/qwen35-4b-sft-student",
        pool=EVAL_POOL,
        manager_reasoning_mode="non_thinking",
        manager_sampling=QWEN35_GENERAL_NON_THINKING,
        manager_checkpoint=SFT_CHECKPOINTS_ROOT / "qwen35-4b-nonthinking-mixed-v3-student-4gpu" / "final",
    ),
    Tau2Experiment(
        name="qwen35_4b_student_checkpoint",
        description=(
            "Any Qwen3.5-4B manager checkpoint (`run.py --manager-checkpoint`), "
            "with evaluation sampling: SFT checkpoint evaluations."
        ),
        manager_backend="local_vllm",
        manager_model_id="decomposer/qwen35-4b-student",
        pool=EVAL_POOL,
        manager_reasoning_mode="non_thinking",
        manager_sampling=QWEN35_GENERAL_NON_THINKING,
    ),
)

EXPERIMENTS_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}
if len(EXPERIMENTS_BY_NAME) != len(EXPERIMENTS):
    raise RuntimeError("Duplicate tau2 experiment names")


def get_experiment(name: str) -> Tau2Experiment:
    try:
        return EXPERIMENTS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown tau2 experiment {name!r}; known: {sorted(EXPERIMENTS_BY_NAME)}") from None
