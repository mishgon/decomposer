"""GAIA2-specific prompt composition for reproducible ablations."""

from __future__ import annotations

from typing import Literal, cast

from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT, RUN_BUDGET_CONVENTION

Gaia2ManagerPromptAddendumProfile = Literal["gaia2-ambiguity"]

GAIA2_AMBIGUITY_MANAGER_ADDENDUM = (
    "GAIA2 ambiguity policy: Use subagents and available environment information "
    "before deciding that something is ambiguous. Complete all clear parts. For any "
    "remaining ambiguous, contradictory, or impossible part, do not guess or execute "
    "it; ask the user explicitly for the clarification needed."
)

GAIA2_MANAGER_PROMPT_ADDENDA = {
    "gaia2-ambiguity": GAIA2_AMBIGUITY_MANAGER_ADDENDUM,
}


def compose_decomposer_system_prompt(
    addendum_profile: str | None = None,
) -> str:
    """Resolve the base Decomposer prompt and an optional GAIA2 addendum."""

    prompt = DECOMPOSER_SYSTEM_PROMPT.replace(
        "{run_budget_convention}\n\n",
        RUN_BUDGET_CONVENTION.format(
            agent_run_budget_seconds=60.0, agent_shutdown_grace_seconds=60.0,
        ) + "\n\n",
    )
    if addendum_profile is None:
        return prompt
    try:
        addendum = GAIA2_MANAGER_PROMPT_ADDENDA[
            cast(Gaia2ManagerPromptAddendumProfile, addendum_profile)
        ]
    except KeyError as error:
        expected = ", ".join(GAIA2_MANAGER_PROMPT_ADDENDA)
        raise ValueError(
            f"Unknown GAIA2 manager prompt addendum {addendum_profile!r}; "
            f"expected one of: {expected}"
        ) from error
    return f"{prompt}\n\n{addendum}"
