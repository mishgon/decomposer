"""Hosted teacher model clients used by Toolathlon evaluation."""

from __future__ import annotations

import os

from decomposer.chat_vllm import ChatVLLM


def create_vllm_teacher(
    *,
    model: str,
    base_url: str,
    timeout: float,
    max_retries: int,
    thinking: bool = True,
) -> ChatVLLM:
    """Create a local Decomposer model with model-card sampling."""
    model_lower = model.lower()
    if "gemma-4" in model_lower:
        top_k = 64
    elif "qwen" in model_lower:
        top_k = 20
    else:
        raise ValueError(f"Unsupported local Decomposer teacher model: {model!r}")

    extra_body: dict[str, object] = {
        "top_k": top_k,
        "include_reasoning": thinking,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if "qwen" in model_lower:
        extra_body.update({"min_p": 0.0, "repetition_penalty": 1.0})

    temperature = 1.0 if thinking or "gemma-4" in model_lower else 0.7
    top_p = 0.95 if thinking or "gemma-4" in model_lower else 0.8
    model_kwargs: dict[str, object] = {}
    if not thinking and "qwen" in model_lower:
        model_kwargs["presence_penalty"] = 1.5

    return ChatVLLM(
        model=model,
        api_key="EMPTY",
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        max_retries=max_retries,
        use_responses_api=False,
        preserve_reasoning=thinking,
        extra_body=extra_body,
        **model_kwargs,
    )


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
