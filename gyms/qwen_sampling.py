"""Recommended sampling presets for Qwen model families."""

from __future__ import annotations

from dataclasses import dataclass


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


def qwen35_general_sampling(*, thinking: bool) -> QwenSamplingParams:
    """Return Qwen3.5's recommended general-task preset for ``thinking`` mode."""

    if thinking:
        return QWEN35_GENERAL_THINKING
    return QWEN35_GENERAL_NON_THINKING


def qwen36_non_thinking_sampling() -> QwenSamplingParams:
    """Return Qwen3.6's recommended instruct/non-thinking preset."""

    return QWEN36_NON_THINKING
