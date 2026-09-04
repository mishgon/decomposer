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
DEEPSEEK_REASONING_EFFORT = os.environ.get(
    "TOOLATHLON_DEEPSEEK_REASONING_EFFORT", "high"
)


def _explicit_system_prompt(value: object | None) -> str | None:
    """Ignore RunnableConfig injected positionally by the LangGraph server."""
    return value if isinstance(value, str) else None


def _create_subagent(
    model_id: str,
    base_url_env: str,
    default_port: int,
    *,
    thinking: bool,
    system_prompt: str | None = None,
) -> CompiledStateGraph:
    qwen_non_thinking = "qwen3.5" in model_id.lower() and not thinking
    if qwen_non_thinking:
        # Official Qwen3.5 recommendation for general non-thinking tasks.
        temperature = 0.7
        top_p = 0.8
        presence_penalty = 1.5
        extra_body = {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    else:
        temperature = 1.0
        top_p = 0.95
        presence_penalty = None
        extra_body = {"top_k": 64}
        if thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
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
    model_kwargs = {}
    if presence_penalty is not None:
        model_kwargs["presence_penalty"] = presence_penalty
    model = ChatVLLM(
        model=model_id,
        base_url=base_url,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        temperature=temperature,
        top_p=top_p,
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
        **model_kwargs,
    )
    return create_agent(
        model=model,
        tools=get_tools(),
        system_prompt=(
            system_prompt
            or os.environ.get("TOOLATHLON_AGENT_SYSTEM_PROMPT")
            or SUBAGENT_SYSTEM_PROMPT
        ),
        middleware=[durable_model_call_log, truncate_mcp_tool_output],
    )


def qwen_3_5_4b_non_thinking(
    system_prompt: object | None = None,
) -> CompiledStateGraph:
    return _create_subagent(
        "Qwen/Qwen3.5-4B",
        "QWEN_3_5_4B_BASE_URL",
        8030,
        thinking=False,
        system_prompt=_explicit_system_prompt(system_prompt),
    )


def gemma_4_e4b_thinking(
    system_prompt: object | None = None,
) -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-E4B-it",
        "GEMMA_4_E4B_BASE_URL",
        8030,
        thinking=True,
        system_prompt=_explicit_system_prompt(system_prompt),
    )


def gemma_4_31b_thinking(
    system_prompt: object | None = None,
) -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-31B-it",
        "GEMMA_4_31B_BASE_URL",
        8030,
        thinking=True,
        system_prompt=_explicit_system_prompt(system_prompt),
    )


def gemma_4_26b_a4b_non_thinking(
    system_prompt: object | None = None,
) -> CompiledStateGraph:
    return _create_subagent(
        "google/gemma-4-26B-A4B-it",
        "GEMMA_4_26B_A4B_BASE_URL",
        8030,
        thinking=False,
        system_prompt=_explicit_system_prompt(system_prompt),
    )


def deepseek_openrouter(
    system_prompt: object | None = None,
) -> CompiledStateGraph:
    model = create_openrouter_model(
        model=os.environ.get(
            "TOOLATHLON_OPENROUTER_MODEL",
            "deepseek/deepseek-v4-flash-0731",
        ),
        reasoning={"effort": DEEPSEEK_REASONING_EFFORT},
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=REQUEST_MAX_RETRIES,
    )
    return create_agent(
        model=model,
        tools=get_tools(),
        system_prompt=(
            _explicit_system_prompt(system_prompt)
            or os.environ.get("TOOLATHLON_AGENT_SYSTEM_PROMPT")
            or SUBAGENT_SYSTEM_PROMPT
        ),
        middleware=[durable_model_call_log, truncate_mcp_tool_output],
    )
