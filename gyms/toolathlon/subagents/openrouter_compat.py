from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ToolathlonOpenRouterChat(ChatOpenAI):
    """Stable OpenAI-compatible OpenRouter client for Toolathlon agents."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        **kwargs: Any,
    ):
        # OpenRouter's DeepSeek route is substantially more predictable when
        # the API's default tool policy is sent explicitly.
        return super().bind_tools(
            tools,
            tool_choice="auto" if tool_choice is None else tool_choice,
            **kwargs,
        )


def create_openrouter_model(
    *,
    model: str,
    reasoning: dict[str, Any],
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> ToolathlonOpenRouterChat:
    return ToolathlonOpenRouterChat(
        model=model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get(
            "TOOLATHLON_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
        ),
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        extra_body={"reasoning": reasoning},
    )
