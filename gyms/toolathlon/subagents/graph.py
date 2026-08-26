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
        f"http://127.0.0.1:{default_port}/v1",
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


def qwen_3_5_4b_non_thinking() -> CompiledStateGraph:
    return _create_subagent(
        "Qwen/Qwen3.5-4B",
        "QWEN_3_5_4B_BASE_URL",
        8030,
        thinking=False,
    )
