import os

from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from decomposer.chat_vllm import ChatVLLM
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from webapp import get_tools, truncate_mcp_tool_output


REQUEST_TIMEOUT_SECONDS = 300.0


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
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=thinking,
        extra_body=extra_body,
    )
    return create_agent(
        model=model,
        tools=get_tools(),
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        middleware=[truncate_mcp_tool_output],
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
    # https://huggingface.co/Qwen/Qwen3.5-4B#best-practices
    model = ChatVLLM(
        model="Qwen/Qwen3.5-4B",
        base_url=os.environ.get(
            "QWEN_3_5_4B_BASE_URL",
            "http://host.docker.internal:8024/v1",
        ),
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=False,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_agent(
        model=model,
        tools=get_tools(),
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        middleware=[truncate_mcp_tool_output],
    )
