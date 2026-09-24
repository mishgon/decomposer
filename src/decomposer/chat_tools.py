"""Chat-format schemas of the Decomposer's tools, for building training data."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.utils.function_calling import convert_to_openai_tool

from .core import DecomposerAgentMiddleware, SubagentType


def build_decomposer_chat_tools(
    subagent_types: Sequence[SubagentType],
) -> list[dict[str, Any]]:
    """Return the OpenAI/Transformers schemas exposed by Decomposer.

    Dataset preparation uses this helper so canonical SFT records and live
    Decomposer agents cannot silently drift to different tool descriptions or
    argument schemas. Constructing the middleware is side-effect free; clients
    are created lazily only when a tool is invoked.
    """
    middleware = DecomposerAgentMiddleware(
        subagent_types,
        subagent_recursion_limit=None,
    )
    return [convert_to_openai_tool(tool) for tool in middleware.tools]
