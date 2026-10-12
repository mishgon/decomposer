"""Chat-format schemas of the Decomposer's tools and its system prompt, for building training data."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.utils.function_calling import convert_to_openai_tool

from decomposer.core import AgentType, DecomposerAgentMiddleware
from decomposer.prompts import DECOMPOSER_SYSTEM_PROMPT, RUN_BUDGET_CONVENTION


def build_decomposer_chat_tools(
    agent_types: Sequence[AgentType],
) -> list[dict[str, Any]]:
    """Return the OpenAI/Transformers schemas exposed by Decomposer.

    Dataset preparation uses this helper so canonical SFT records and live
    Decomposer agents cannot silently drift to different tool descriptions or
    argument schemas. Constructing the middleware is side-effect free; clients
    are created lazily only when a tool is invoked.
    """
    middleware = DecomposerAgentMiddleware(
        agent_types,
        agent_recursion_limit=None,
    )
    return [convert_to_openai_tool(tool) for tool in middleware.tools]


def decomposer_system_prompt() -> str:
    """Return the system prompt of a Decomposer agent with the default run budget.

    This repeats what `create_decomposer_agent` does with its default 60-second run
    budget and 60-second shutdown grace, which every teacher collection uses.
    """
    return DECOMPOSER_SYSTEM_PROMPT.replace(
        "{run_budget_convention}\n\n",
        RUN_BUDGET_CONVENTION.format(
            agent_run_budget_seconds=60.0, agent_shutdown_grace_seconds=60.0,
        ) + "\n\n",
    )
