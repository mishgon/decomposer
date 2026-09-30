from decomposer.models import create_model
from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph


def qwen_3_5_4b_unlooped_thinking() -> CompiledStateGraph:
    return create_agent(model=create_model("qwen_3_5_4b_unlooped_thinking"),
                        tools=[], system_prompt=SUBAGENT_SYSTEM_PROMPT)
