import os

from decomposer.core import create_decomposer_agent
from decomposer.models import create_model
from decomposer.prompts import AGENT_SYSTEM_PROMPT
from langchain.agents import create_agent
from model_logging import durable_model_call_log
from tools import get_tools, truncate_mcp_tool_output


def qwen_3_5_4b_unlooped_thinking():
    return create_agent(
        model=create_model("qwen_3_5_4b_unlooped_thinking"),
        tools=get_tools(),
        system_prompt=AGENT_SYSTEM_PROMPT,
        middleware=[durable_model_call_log, truncate_mcp_tool_output],
    )


def decomposer():
    return create_decomposer_agent(
        decomposer_model=create_model(os.environ.get(
            "TOOLATHLON_DECOMPOSER_MODEL", "qwen_3_8_flash_next_low_thinking",
        )),
        agent_types=[{
            "agent_type_id": "qwen_3_5_4b_unlooped_thinking",
            "description": "Qwen3.5-4B unlooped thinking agent with all task tools.",
            "assistant_id": "qwen_3_5_4b_unlooped_thinking",
            "url": "http://127.0.0.1:2024",
        }],
        agent_recursion_limit=410,
    )
