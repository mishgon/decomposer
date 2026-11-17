from decomposer.core import create_decomposer_agent
from decomposer.run_budget import RunBudgetContext, RunBudgetMiddleware
from prompts import AGENT_SYSTEM_PROMPT
from decomposer.models import create_model
from langchain.agents import create_agent


OCC_4B_NON_THINKING_MODEL_ID = "lmrouter/occ_4b_non_thinking"
DECOMPOSER_MODEL_ID = "lmrouter/qwen_3_8_flash_next_non_thinking"


def occ_4b_non_thinking():
    from model_logging import durable_model_call_log
    from tools import get_runtime, get_tools, truncate_mcp_tool_output

    return create_agent(
        context_schema=RunBudgetContext,
        model=create_model(OCC_4B_NON_THINKING_MODEL_ID),
        tools=get_tools(),
        system_prompt=AGENT_SYSTEM_PROMPT.format(workspace_dir=get_runtime()["agent_workspace"]),
        middleware=[RunBudgetMiddleware(), durable_model_call_log, truncate_mcp_tool_output],
    )


def decomposer():
    from tools import get_runtime

    return create_decomposer_agent(
        decomposer_model=create_model(DECOMPOSER_MODEL_ID),
        agent_types=[{
            "agent_type_id": "occ_4b_non_thinking",
            "description": "OCC-4B non-thinking agent equipped with all the required tools.",
            "assistant_id": "occ_4b_non_thinking",
            "url": get_runtime()["agent_server_url"],
        }],
        agent_recursion_limit=410,
        agent_run_budget_seconds=60, agent_shutdown_grace_seconds=60,
    )
