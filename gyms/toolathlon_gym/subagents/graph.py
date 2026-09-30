from decomposer.models import create_model
from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from langchain.agents import create_agent
from model_logging import durable_model_call_log
from webapp import get_tools, truncate_mcp_tool_output


def qwen_3_5_4b_unlooped_thinking():
    return create_agent(
        model=create_model("qwen_3_5_4b_unlooped_thinking"),
        tools=get_tools(),
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        middleware=[durable_model_call_log, truncate_mcp_tool_output],
    )
