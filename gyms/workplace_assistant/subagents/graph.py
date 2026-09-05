from __future__ import annotations

import json
import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.graph.state import CompiledStateGraph
from responses_api_agents.decomposer_agent.subagents.graph import (
    SYSTEM_PROMPT,
    NeMoGymSubagentMiddleware,
)

from decomposer.chat_vllm import ChatVLLM
from gyms.qwen_sampling import qwen35_general_sampling

REQUEST_TIMEOUT_SECONDS = 300.0
MODEL_BASE_URLS_ENV = "WORKPLACE_ASSISTANT_MODEL_BASE_URLS_JSON"
SUBAGENT_MAX_MODEL_CALLS = int(
    os.environ.get("DECOMPOSER_SUBAGENT_MAX_MODEL_CALLS", "100")
)
if SUBAGENT_MAX_MODEL_CALLS < 1:
    raise ValueError("DECOMPOSER_SUBAGENT_MAX_MODEL_CALLS must be at least 1")
_max_completion_tokens = os.environ.get("DECOMPOSER_SUBAGENT_MAX_COMPLETION_TOKENS")
SUBAGENT_MAX_COMPLETION_TOKENS = (
    int(_max_completion_tokens) if _max_completion_tokens is not None else None
)
if (
    SUBAGENT_MAX_COMPLETION_TOKENS is not None
    and SUBAGENT_MAX_COMPLETION_TOKENS < 1
):
    raise ValueError(
        "DECOMPOSER_SUBAGENT_MAX_COMPLETION_TOKENS must be at least 1"
    )


def _completion_kwargs() -> dict[str, int]:
    if SUBAGENT_MAX_COMPLETION_TOKENS is None:
        return {}
    return {"max_completion_tokens": SUBAGENT_MAX_COMPLETION_TOKENS}


def _model_base_url(model_id: str, default_port: int) -> str:
    raw = os.environ.get(MODEL_BASE_URLS_ENV)
    if raw is None:
        return f"http://127.0.0.1:{default_port}/v1"
    try:
        configured = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{MODEL_BASE_URLS_ENV} must contain valid JSON") from error
    if not isinstance(configured, dict):
        raise ValueError(f"{MODEL_BASE_URLS_ENV} must contain a JSON object")
    value = configured.get(model_id)
    if value is None:
        return f"http://127.0.0.1:{default_port}/v1"
    if not isinstance(value, str) or not value:
        raise ValueError(f"{MODEL_BASE_URLS_ENV}[{model_id!r}] must be a URL string")
    return value


def _create_subagent(model: ChatVLLM) -> CompiledStateGraph:
    return create_agent(
        model=model,
        tools=[],
        middleware=[
            NeMoGymSubagentMiddleware(),
            ModelCallLimitMiddleware(
                run_limit=SUBAGENT_MAX_MODEL_CALLS,
                exit_behavior="error",
            ),
        ],
        system_prompt=SYSTEM_PROMPT,
    )


def qwen35_4b_non_thinking() -> CompiledStateGraph:
    sampling = qwen35_general_sampling(thinking=False)
    model = ChatVLLM(
        model="Qwen/Qwen3.5-4B",
        base_url=_model_base_url("Qwen/Qwen3.5-4B", 8025),
        api_key="EMPTY",
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        presence_penalty=sampling.presence_penalty,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=False,
        **_completion_kwargs(),
        extra_body={
            **sampling.extra_body,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return _create_subagent(model)


def _gemma_subagent(
    *,
    model_id: str,
    default_port: int,
    thinking: bool,
) -> CompiledStateGraph:
    extra_body: dict[str, Any] = {
        "top_k": 64,
        "include_reasoning": thinking,
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            **({"preserve_thinking": True} if thinking else {}),
        },
    }
    model = ChatVLLM(
        model=model_id,
        base_url=_model_base_url(model_id, default_port),
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=thinking,
        **_completion_kwargs(),
        extra_body=extra_body,
    )
    return _create_subagent(model)


def gemma_4_2b_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-E2B-it", default_port=8020, thinking=True
    )


def gemma_4_2b_non_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-E2B-it", default_port=8020, thinking=False
    )


def gemma_4_4b_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-E4B-it", default_port=8021, thinking=True
    )


def gemma_4_4b_non_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-E4B-it", default_port=8021, thinking=False
    )


def gemma_4_12b_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-12B-it", default_port=8022, thinking=True
    )


def gemma_4_12b_non_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-12B-it", default_port=8022, thinking=False
    )


def gemma_4_26b_a4b_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-26B-A4B-it", default_port=8023, thinking=True
    )


def gemma_4_26b_a4b_non_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-26B-A4B-it", default_port=8023, thinking=False
    )


def gemma_4_31b_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-31B-it", default_port=8027, thinking=True
    )


def gemma_4_31b_non_thinking() -> CompiledStateGraph:
    return _gemma_subagent(
        model_id="google/gemma-4-31B-it", default_port=8027, thinking=False
    )
