from __future__ import annotations

import os

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
        base_url="http://127.0.0.1:8025/v1",
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


def gemma_4_4b_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-E4B-it",
        base_url="http://127.0.0.1:8021/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=True,
        **_completion_kwargs(),
        extra_body={
            "top_k": 64,
            "include_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return _create_subagent(model)
