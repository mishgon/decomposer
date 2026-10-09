import asyncio
from dataclasses import dataclass
import os
from pathlib import Path

from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from gyms.synth.executor import execute_saved


@dataclass
class Context:
    directory: str


async def execute(state: AgentState, runtime: Runtime[Context]):
    directory = Path(runtime.context.directory).resolve()
    root = Path(os.environ["SYNTH_ARTIFACT_ROOT"]).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise ValueError("Episode must be inside SYNTH_ARTIFACT_ROOT")
    report = await asyncio.to_thread(execute_saved, directory, state["messages"][-1].content,
                                    runtime.execution_info.thread_id, runtime.execution_info.run_id)
    return {"messages": [AIMessage(content=report)]}


def worker():
    graph = StateGraph(AgentState, context_schema=Context)
    graph.add_node("execute", execute)
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    return graph.compile()
