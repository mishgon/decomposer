import os

import httpx
from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from decomposer.chat_vllm import ChatVLLM
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
try:
    from .openrouter_compat import create_openrouter_model
    from .model_logging import durable_model_call_log
    from .webapp import get_tools, truncate_mcp_tool_output
except ImportError:  # Loaded by `langgraph dev` with this directory on sys.path.
    from openrouter_compat import create_openrouter_model
    from model_logging import durable_model_call_log
    from webapp import get_tools, truncate_mcp_tool_output


REQUEST_TIMEOUT_SECONDS = 600.0
REQUEST_MAX_RETRIES = 2
MAX_OUTPUT_TOKENS = 8192
DEEPSEEK_REASONING_EFFORT = os.environ.get(
    "TOOLATHLON_DEEPSEEK_REASONING_EFFORT", "high"
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
        f"http://127.0.0.1:{default_port}/v1",
    )
    model = ChatVLLM(
        model=model_id,
        base_url=base_url,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        temperature=1.0,
        top_p=0.95,
        # A malformed turn can otherwise generate until the server-wide model
        # limit and monopolize one DP worker for minutes. Toolathlon tool calls
        # and final answers are comfortably below this ceiling.
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=REQUEST_MAX_RETRIES,
        # vLLM's local data-parallel frontend assigns accepted connections to
        # API workers. Reconnecting each turn prevents a long-running episode
        # from remaining pinned to one GPU after the other episodes finish.
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


def qwen_3_5_4b_non_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "Qwen/Qwen3.5-4B",
        "QWEN_3_5_4B_BASE_URL",
        8030,
        thinking=False,
    )


def deepseek_openrouter() -> CompiledStateGraph:
    model = create_openrouter_model(
        model=os.environ.get(
            "TOOLATHLON_OPENROUTER_MODEL",
            "deepseek/deepseek-v4-flash-0731",
        ),
        reasoning={"effort": DEEPSEEK_REASONING_EFFORT},
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=REQUEST_MAX_RETRIES,
    )
    return create_agent(
        model=model,
        tools=get_tools(),
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        middleware=[durable_model_call_log, truncate_mcp_tool_output],
    )
