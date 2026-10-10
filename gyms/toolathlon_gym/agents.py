from decomposer.core import create_decomposer_agent
from decomposer.run_budget import RunBudgetMiddleware
from decomposer.models import create_model
from decomposer.prompts import AGENT_SYSTEM_PROMPT
from langchain.agents import create_agent


QWEN_3_5_4B_THINKING_MODEL_ID = "lmrouter/qwen_3_5_4b_unlooped_thinking"
DECOMPOSER_MODEL_ID = "lmrouter/qwen_3_8_flash_next_non_thinking"


def qwen_3_5_4b_thinking():
    from model_logging import durable_model_call_log
    from tools import get_tools, truncate_mcp_tool_output

    return create_agent(
        model=create_model(QWEN_3_5_4B_THINKING_MODEL_ID),
        tools=get_tools(),
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=[RunBudgetMiddleware(), durable_model_call_log, truncate_mcp_tool_output],
    )


def decomposer(*, middleware=()):
    return create_decomposer_agent(
        decomposer_model=create_model(DECOMPOSER_MODEL_ID),
        agent_types=[{
            "agent_type_id": "qwen_3_5_4b_thinking",
            "description": "Qwen3.5-4B thinking agent equipped with all the available tools.",
            "assistant_id": "qwen_3_5_4b_thinking",
            "url": "http://127.0.0.1:2024",
        }],
        agent_recursion_limit=410,
        middleware=middleware,
    )
