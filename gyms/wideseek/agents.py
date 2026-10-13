"""Named WideSeek agents; model profiles are fixed here, not CLI options."""
from langchain.agents import create_agent

from decomposer.core import create_decomposer_agent
from decomposer.run_budget import RunBudgetMiddleware
from gyms.wideseek.prompts import AGENT_SYSTEM_PROMPT
from gyms.wideseek.runtime import Context, ModelLog, model
from gyms.wideseek.worker import access, search

REACT_MODEL = "vllm/qwen_3_5_4b_thinking"
DECOMPOSER_MODEL = "vllm/qwen_3_8_flash_next_non_thinking"
RESEARCHER_MODEL = REACT_MODEL
JUDGE_MODEL = DECOMPOSER_MODEL
AGENT_MODELS = {"react": REACT_MODEL, "decomposer": DECOMPOSER_MODEL}
RESEARCHER_ID = "researcher"
AGENT_RUN_BUDGET_SECONDS = 60.0
AGENT_SHUTDOWN_GRACE_SECONDS = 60.0


def researcher():
    return create_agent(model(RESEARCHER_MODEL), tools=[search, access], context_schema=Context,
        middleware=[RunBudgetMiddleware(), ModelLog("researcher")],
        system_prompt=AGENT_SYSTEM_PROMPT)


def react(policy, checkpointer, *, middleware=()):
    return create_agent(policy, tools=[search, access], system_prompt=AGENT_SYSTEM_PROMPT,
        context_schema=Context, middleware=[RunBudgetMiddleware(), ModelLog("researcher"), *middleware], checkpointer=checkpointer)


def decomposer(policy, checkpointer, worker_url, *, middleware=()):
    return create_decomposer_agent(decomposer_model=policy,
        agent_types=[{"agent_type_id": "qwen_3_5_4b_thinking", "assistant_id": RESEARCHER_ID,
            "description": "Qwen3.5-4B thinking agent equipped with search and access tools.",
            "url": worker_url}],
        checkpointer=checkpointer, middleware=[ModelLog("decomposer"), *middleware],
        context_schema=Context, agent_recursion_limit=410,
        agent_run_budget_seconds=AGENT_RUN_BUDGET_SECONDS,
        agent_shutdown_grace_seconds=AGENT_SHUTDOWN_GRACE_SECONDS)
