"""Recommended sampling presets for Qwen model families."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class QwenSamplingParams:
    """OpenAI-compatible sampling parameters for one Qwen generation mode."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float

    @property
    def extra_body(self) -> dict[str, int | float]:
        """Parameters that OpenAI clients forward through ``extra_body``."""

        return {
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }


QWEN35_GENERAL_THINKING = QwenSamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)
QWEN35_GENERAL_NON_THINKING = QwenSamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)
QWEN36_NON_THINKING = QwenSamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)
QWEN36_THINKING = QwenSamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)


# Qwen3.8 ships no presets of its own yet; these are the Qwen3.5/3.6 family values.
QWEN38_NON_THINKING = QwenSamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)
QWEN38_THINKING = QwenSamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)
# The thinking teacher for Decomposer traces: the family values without the
# presence penalty, which is an anti-repetition workaround, not a teacher setting.
QWEN38_TEACHER_THINKING = QwenSamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)
# On-policy training rollouts sample from the raw policy: the per-token estimators
# (reverse KL, importance ratios) assume the sampled token came from pi itself, which
# top-k/top-p truncation and penalties would violate.
UNTRUNCATED_SAMPLING = QwenSamplingParams(
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)

def qwen35_general_sampling(*, thinking: bool) -> QwenSamplingParams:
    """Return Qwen3.5's recommended general-task preset for ``thinking`` mode."""

    if thinking:
        return QWEN35_GENERAL_THINKING
    return QWEN35_GENERAL_NON_THINKING


def qwen36_non_thinking_sampling() -> QwenSamplingParams:
    """Return Qwen3.6's recommended instruct/non-thinking preset."""

    return QWEN36_NON_THINKING


def qwen36_thinking_sampling() -> QwenSamplingParams:
    """Return Qwen3.6's recommended general-task thinking preset."""

    return QWEN36_THINKING


SUBAGENT_SAMPLING_ENV = "DECOMPOSER_SUBAGENT_SAMPLING_JSON"
SUBAGENT_MAX_COMPLETION_TOKENS_ENV = "DECOMPOSER_SUBAGENT_MAX_COMPLETION_TOKENS"


@dataclass(frozen=True)
class SubagentSampling:
    """Sampling for a non-thinking subagent, chosen per experiment.

    Unset optional fields are not sent, so the server's defaults apply; a preset
    that leaves the penalties unset therefore runs without them.
    """

    temperature: float
    top_p: float
    top_k: int
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    # Exported through DECOMPOSER_SUBAGENT_MAX_COMPLETION_TOKENS, which the subagent
    # graphs already read; None leaves the length to the provider.
    max_completion_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_completion_tokens is not None and self.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be at least 1")

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


# Qwen3.5-4B-unlooped (answer-ready cut DPO against looping), non-thinking mode.
# Its authors recommend the base model's non-thinking values and evaluated the
# model without any penalty, so none is sent. Thinking mode would be 0.6/0.95/20
# with 8192 tokens; Decomposer subagents do not think.
# The cap is 8192, not the doc's non-thinking 2048: in the Workplace smoke
# (2026-09-25) 2048 cut 2 of 16 subagent runs, both legitimate full-record reports
# rather than loops, and the base model rarely loops without thinking.
QWEN35_UNLOOPED_NON_THINKING = SubagentSampling(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_completion_tokens=8192,
)


def subagent_sampling_environment(sampling: SubagentSampling | None) -> dict[str, str]:
    """Environment that makes the subagent graphs use ``sampling``.

    ``None`` exports nothing, so the graphs keep their built-in preset.
    """

    if sampling is None:
        return {}
    record = sampling.as_record()
    max_completion_tokens = record.pop("max_completion_tokens")
    environment = {SUBAGENT_SAMPLING_ENV: json.dumps(record, sort_keys=True)}
    if max_completion_tokens is not None:
        environment[SUBAGENT_MAX_COMPLETION_TOKENS_ENV] = str(max_completion_tokens)
    return environment


def non_thinking_subagent_sampling_kwargs(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """``ChatVLLM`` sampling kwargs for a non-thinking Qwen3.5 subagent.

    Without ``DECOMPOSER_SUBAGENT_SAMPLING_JSON`` this is Qwen3.5's general
    non-thinking preset, exactly as the graphs sent it before the setting existed.
    The completion cap is not included: the graphs read it from its own variable.
    """

    environ = os.environ if environ is None else environ
    raw = environ.get(SUBAGENT_SAMPLING_ENV)
    if raw is None:
        preset = qwen35_general_sampling(thinking=False)
        kwargs: dict[str, Any] = {
            "temperature": preset.temperature,
            "top_p": preset.top_p,
            "presence_penalty": preset.presence_penalty,
        }
        extra_body: dict[str, Any] = dict(preset.extra_body)
    else:
        values = json.loads(raw)
        known = {item.name for item in fields(SubagentSampling)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"{SUBAGENT_SAMPLING_ENV} has unknown keys: {unknown}")
        sampling = SubagentSampling(**values)
        kwargs = {"temperature": sampling.temperature, "top_p": sampling.top_p}
        if sampling.presence_penalty is not None:
            kwargs["presence_penalty"] = sampling.presence_penalty
        extra_body = {"top_k": sampling.top_k}
        if sampling.min_p is not None:
            extra_body["min_p"] = sampling.min_p
        if sampling.repetition_penalty is not None:
            extra_body["repetition_penalty"] = sampling.repetition_penalty
    extra_body.update(
        {
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    kwargs["extra_body"] = extra_body
    return kwargs
