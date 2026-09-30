import asyncio
from pathlib import Path

from langchain.agents.middleware import AgentState
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph


async def write_later(state):
    path = Path(state["messages"][-1].content)
    path.with_suffix(".started").touch()
    await asyncio.sleep(2)
    path.write_text("Subagent changed workspace")
    return {"messages": [AIMessage(content="Written")]}


builder = StateGraph(AgentState)
builder.add_node("write_later", write_later)
builder.add_edge(START, "write_later")
builder.add_edge("write_later", END)
graph = builder.compile()
