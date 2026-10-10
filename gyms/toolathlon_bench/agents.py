from decomposer.core import create_decomposer_agent
from decomposer.models import create_model
from langchain.agents import create_agent


QWEN_3_5_4B_THINKING_MODEL_ID = "lmrouter/qwen_3_5_4b_unlooped_thinking"
DECOMPOSER_MODEL_ID = "lmrouter/qwen_3_8_flash_next_non_thinking"


def qwen_3_5_4b_thinking():
    from model_logging import durable_model_call_log
    from tools import get_runtime, get_tools, truncate_mcp_tool_output

    return create_agent(
        model=create_model(QWEN_3_5_4B_THINKING_MODEL_ID),
        tools=get_tools(),
        # The task prompt names the workspace directory and the completion tool.
        system_prompt=get_runtime()["agent_system_prompt"],
        middleware=[durable_model_call_log, truncate_mcp_tool_output],
    )


def decomposer():
    from tools import get_runtime

    return create_decomposer_agent(
        decomposer_model=create_model(DECOMPOSER_MODEL_ID),
        agent_types=[{
            "agent_type_id": "qwen_3_5_4b_thinking",
            "description": "Qwen3.5-4B thinking agent equipped with all the available tools.",
            "assistant_id": "qwen_3_5_4b_thinking",
            "url": get_runtime()["agent_server_url"],
        }],
        agent_recursion_limit=410,
    )
