from decomposer.models import MODELS
from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph


def qwen_3_5_4b_unlooped_thinking() -> CompiledStateGraph:
    return create_agent(model=MODELS["Qwen/Qwen3.5-4B-unlooped"],
                        tools=[], system_prompt=SUBAGENT_SYSTEM_PROMPT)
