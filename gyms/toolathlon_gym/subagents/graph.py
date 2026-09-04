import os

import httpx
from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from decomposer.chat_vllm import ChatVLLM
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from model_logging import durable_model_call_log
from webapp import get_tools, truncate_mcp_tool_output


REQUEST_TIMEOUT_SECONDS = 600.0
REQUEST_MAX_RETRIES = 2


def configured_non_thinking() -> CompiledStateGraph:
    return _create_subagent(
        os.environ.get(
            "DECOMPOSER_SUBAGENT_MODEL", "google/gemma-4-26B-A4B-it"
        ),
        "DECOMPOSER_SUBAGENT_BASE_URL",
        8023,
        thinking=False,
    )


def _create_subagent(
    model_id: str,
    base_url_env: str,
    default_port: int,
    *,
    thinking: bool,
) -> CompiledStateGraph:
    extra_body = {"top_k": 64}
    if not thinking:
        extra_body.update(
            {
                "include_reasoning": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )

    base_url = os.environ.get(
        base_url_env,
        f"http://host.docker.internal:{default_port}/v1",
    )
    model = ChatVLLM(
        model=model_id,
        base_url=base_url,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=REQUEST_MAX_RETRIES,
        http_async_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=0)
        ),
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=thinking,
        extra_body=extra_body,
    )
    return create_agent(
        model=model,
        tools=get_tools(),
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        middleware=[durable_model_call_log, truncate_mcp_tool_output],
    )


def gemma_4_2b_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-E2B-it",
        "GEMMA_4_E2B_BASE_URL",
        8020,
        thinking=True,
    )


def gemma_4_2b_non_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-E2B-it",
        "GEMMA_4_E2B_BASE_URL",
        8020,
        thinking=False,
    )


def gemma_4_4b_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-E4B-it",
        "GEMMA_4_E4B_BASE_URL",
        8021,
        thinking=True,
    )


def gemma_4_4b_non_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-E4B-it",
        "GEMMA_4_E4B_BASE_URL",
        8021,
        thinking=False,
    )


def gemma_4_12b_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-12B-it",
        "GEMMA_4_12B_BASE_URL",
        8022,
        thinking=True,
    )


def gemma_4_12b_non_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-12B-it",
        "GEMMA_4_12B_BASE_URL",
        8022,
        thinking=False,
    )


def gemma_4_26b_a4b_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-26B-A4B-it",
        "GEMMA_4_26B_A4B_BASE_URL",
        8023,
        thinking=True,
    )


def gemma_4_26b_a4b_non_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-26B-A4B-it",
        "GEMMA_4_26B_A4B_BASE_URL",
        8023,
        thinking=False,
    )


def qwen_3_5_4b_non_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "Qwen/Qwen3.5-4B",
        "QWEN_3_5_4B_BASE_URL",
        8030,
        thinking=False,
    )
