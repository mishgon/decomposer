"""Selectable Decomposer system prompts, applied from outside the core.

`create_decomposer_agent` always builds the manager with the core's
`DECOMPOSER_SYSTEM_PROMPT`. A profile swaps that text on every model call through
`system_prompt_middleware`, so experiments, datasets and the SFT trainer can name
the prompt they were made with without the core knowing about profiles.
"""

from typing import Literal, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, dynamic_prompt

from .prompts import DECOMPOSER_SYSTEM_PROMPT

DecomposerPromptProfile = Literal["teacher", "student"]
DECOMPOSER_PROMPT_PROFILES: tuple[DecomposerPromptProfile, ...] = ("teacher", "student")

# Legacy one-line prompt from the student-prompt ablation. Kept selectable, never a default.
DECOMPOSER_STUDENT_SYSTEM_PROMPT = (
    "You are a manager agent. Complete user tasks exclusively by orchestrating "
    "subagents through the provided tools."
)


def resolve_decomposer_system_prompt(profile: str) -> str:
    """Resolve a stable Decomposer prompt profile to its prompt text."""

    prompts: dict[DecomposerPromptProfile, str] = {
        "teacher": DECOMPOSER_SYSTEM_PROMPT,
        "student": DECOMPOSER_STUDENT_SYSTEM_PROMPT,
    }
    try:
        return prompts[cast(DecomposerPromptProfile, profile)]
    except KeyError as error:
        expected = ", ".join(DECOMPOSER_PROMPT_PROFILES)
        raise ValueError(
            f"Unknown Decomposer prompt profile {profile!r}; expected one of: {expected}"
        ) from error


def system_prompt_middleware(system_prompt: str) -> AgentMiddleware:
    """Middleware that makes `system_prompt` the manager's system message."""

    @dynamic_prompt
    def decomposer_system_prompt(request: ModelRequest) -> str:
        return system_prompt

    return decomposer_system_prompt
