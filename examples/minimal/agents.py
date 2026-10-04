from decomposer.core import create_decomposer_agent
from decomposer.models import create_model
from decomposer.prompts import AGENT_SYSTEM_PROMPT
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph


def qwen_3_5_4b_thinking() -> CompiledStateGraph:
    return create_agent(
        model=create_model("lmrouter/qwen_3_5_4b_unlooped_thinking"),
        tools=[],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


def decomposer() -> CompiledStateGraph:
    return create_decomposer_agent(
        decomposer_model=create_model("lmrouter/qwen_3_8_flash_next_non_thinking"),
        agent_types=[
            {
                "agent_type_id": "qwen_3_5_4b_thinking",
                "description": "Qwen3.5-4B with thinking enabled, without tools.",
                "assistant_id": "qwen_3_5_4b_thinking",
                "url": "http://127.0.0.1:2024",
            }
        ],
    )
