"""Named WideSeek agents; model profiles are fixed here, not CLI options."""
from langchain.agents import create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

from decomposer.core import create_decomposer_agent
from decomposer.run_budget import RunBudgetMiddleware
from gyms.wideseek.prompts import AGENT_SYSTEM_PROMPT, WORKER_SYSTEM_PROMPT
from gyms.wideseek.runtime import Context, ModelLog, model, DEFAULT_MODEL, DEFAULT_TEACHER, MODEL_BACKEND
from gyms.wideseek.worker import access, search, access_batch, search_batch

REACT_MODEL = DEFAULT_MODEL
DECOMPOSER_MODEL = DEFAULT_TEACHER
RESEARCHER_MODEL = REACT_MODEL
JUDGE_MODEL = DEFAULT_TEACHER
AGENT_MODELS = {"react": REACT_MODEL, "decomposer": DECOMPOSER_MODEL}
RESEARCHER_ID = "researcher"
AGENT_RUN_BUDGET_SECONDS = 60.0
AGENT_SHUTDOWN_GRACE_SECONDS = 60.0


def context_middleware():
    # Edits affect model requests, preserving the original state and raw tool logs.
    return ([ContextEditingMiddleware(edits=[ClearToolUsesEdit(
        trigger=8000, keep=4,
        placeholder="[Earlier tool result cleared from model context; repeat search/access if needed.]",
    )])] if MODEL_BACKEND == "local" else [])


def researcher():
    return create_agent(model(RESEARCHER_MODEL), tools=[search_batch, access_batch], context_schema=Context,
        middleware=[RunBudgetMiddleware(), ModelLog("researcher"), *context_middleware()],
        system_prompt=WORKER_SYSTEM_PROMPT)


def react(policy, checkpointer, *, middleware=()):
    return create_agent(policy, tools=[search, access], system_prompt=AGENT_SYSTEM_PROMPT,
        context_schema=Context, middleware=[RunBudgetMiddleware(), ModelLog("researcher", final_response=True),
            *context_middleware(), *middleware], checkpointer=checkpointer)


def decomposer(policy, checkpointer, worker_url, *, middleware=()):
    return create_decomposer_agent(decomposer_model=policy,
        agent_types=[{"agent_type_id": "qwen_3_5_4b_thinking", "assistant_id": RESEARCHER_ID,
            "description": "Qwen3.5-4B thinking agent equipped with search and access tools.",
            "url": worker_url}],
        checkpointer=checkpointer, middleware=[ModelLog("decomposer", final_response=True), *context_middleware(), *middleware],
        context_schema=Context, agent_recursion_limit=410,
        agent_run_budget_seconds=AGENT_RUN_BUDGET_SECONDS,
        agent_shutdown_grace_seconds=AGENT_SHUTDOWN_GRACE_SECONDS)
