"""Hosted teacher model clients used by Toolathlon evaluation."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def create_lmrouter_teacher(
    *, model: str, max_tokens: int, timeout: float, max_retries: int
) -> ChatOpenAI:
    """Create the hosted Qwen teacher with thinking explicitly disabled."""
    return ChatOpenAI(
        model=model,
        base_url=os.environ["LLM_PROXY_URL"],
        api_key=os.environ["LLM_PROXY_MASTER_KEY"],
        temperature=1.0,
        top_p=0.95,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        use_responses_api=False,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
