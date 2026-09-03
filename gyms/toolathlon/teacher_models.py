"""Hosted teacher model clients used by Toolathlon evaluation."""

from __future__ import annotations

import os

from decomposer.chat_vllm import ChatVLLM


def create_lmrouter_teacher(
    *,
    model: str,
    timeout: float,
    max_retries: int,
    max_tokens: int | None = None,
) -> ChatVLLM:
    """Create the hosted Qwen teacher with thinking explicitly enabled."""
    model_kwargs = {}
    if max_tokens is not None:
        model_kwargs["max_tokens"] = max_tokens

    return ChatVLLM(
        model=model,
        base_url=os.environ["LLM_PROXY_URL"],
        api_key=os.environ["LLM_PROXY_MASTER_KEY"],
        temperature=1.0,
        top_p=0.95,
        timeout=timeout,
        max_retries=max_retries,
        use_responses_api=False,
        preserve_reasoning=True,
        parse_qwen_xml_tool_calls=True,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        **model_kwargs,
    )
