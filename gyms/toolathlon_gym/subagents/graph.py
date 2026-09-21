from decomposer.models import create_model
from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from model_logging import durable_model_call_log
from webapp import get_tools, truncate_mcp_tool_output


def validate_transport_numbers(value):
    """Reject model arguments that LangGraph's JSON transport cannot serialize."""
    if type(value) is int and not -(2**63) <= value < 2**64:
        raise ValueError("Invalid tool argument: integer outside the JSON transport's 64-bit range")
    if isinstance(value, dict):
        for item in value.values():
            validate_transport_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_transport_numbers(item)


@wrap_model_call
async def transport_safe_response(request, handler):
    response = await handler(request)
    for message in response.result:
        for call in getattr(message, "tool_calls", []):
            validate_transport_numbers(call["args"])
    return response


def qwen_3_5_4b_unlooped_thinking():
    return create_agent(
        model=create_model("qwen_3_5_4b_unlooped_thinking"),
        tools=get_tools(),
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
        # Validate outside the logger: retain raw output but never checkpoint unsafe arguments.
        middleware=[transport_safe_response, durable_model_call_log, truncate_mcp_tool_output],
    )
