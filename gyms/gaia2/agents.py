from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime

from decomposer.core import create_decomposer_agent
from decomposer.models import create_model
from decomposer.run_budget import RunBudgetMiddleware
from gyms.gaia2.tools import EpisodeContext, _worker_tools


def _model():
    return create_model(
        os.environ["GAIA2_AGENT_MODEL_ID"],
        base_url=os.environ.get("GAIA2_AGENT_BASE_URL") or None,
    )


class Gaia2ToolChoiceAutoMiddleware(AgentMiddleware):
    """Make the worker's vLLM tool-selection policy explicit and reproducible."""

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        if request.tools:
            request = request.override(tool_choice="auto")
        return await handler(request)


async def run_agent(
    state: MessagesState, runtime: Runtime[EpisodeContext]
) -> dict[str, Any]:
    context = runtime.context
    tools = _worker_tools(context, consumer=f"agent-{id(state)}")
    system_prompt = context["worker_system_prompt"]
    middleware: list[AgentMiddleware] = [
        RunBudgetMiddleware(),
        Gaia2ToolChoiceAutoMiddleware(),
    ]
    agent = create_agent(
        model=_model(),
        tools=tools,
        context_schema=EpisodeContext,
        system_prompt=system_prompt,
        middleware=middleware,
        # The outer graph checkpoints the worker's conversation; this per-run agent
        # needs none. Inheriting the thread's checkpointer breaks forked workers:
        # the in-memory runtime copies a thread's checkpoints into a plain dict, so
        # looking up this agent's new `agent:<task id>` namespace raises KeyError.
        checkpointer=False,
    )
    result = await agent.ainvoke({"messages": state["messages"]}, context=context)
    return {"messages": result["messages"][len(state["messages"]) :]}


builder = StateGraph(MessagesState, context_schema=EpisodeContext)
builder.add_node("agent", run_agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
qwen_3_5_4b_thinking = builder.compile()


def decomposer():
    return create_decomposer_agent(
        decomposer_model=create_model(
            os.environ["GAIA2_DECOMPOSER_MODEL_ID"],
            base_url=os.environ.get("GAIA2_DECOMPOSER_BASE_URL") or None,
        ),
        agent_types=[
            {
                "agent_type_id": "qwen_3_5_4b_thinking",
                "description": "Qwen3.5-4B thinking agent equipped with all the required tools.",
                "assistant_id": "qwen_3_5_4b_thinking",
                "url": os.environ["GAIA2_AGENT_SERVER_URL"],
            }
        ],
        context_schema=EpisodeContext,
        agent_recursion_limit=10000,
        agent_run_budget_seconds=float(os.environ["GAIA2_AGENT_RUN_BUDGET_SECONDS"]),
        agent_shutdown_grace_seconds=float(
            os.environ["GAIA2_AGENT_SHUTDOWN_GRACE_SECONDS"]
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--n-jobs-per-worker", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from langgraph_api.cli import run_server

    run_server(
        host=args.host,
        port=args.port,
        reload=False,
        graphs={
            "decomposer": "gyms.gaia2.agents:decomposer",
            "qwen_3_5_4b_thinking": "gyms.gaia2.agents:qwen_3_5_4b_thinking",
        },
        n_jobs_per_worker=args.n_jobs_per_worker,
        open_browser=False,
        disable_persistence=True,
        allow_blocking=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
