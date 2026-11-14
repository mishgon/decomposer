"""Named WideSeek agents; model profiles are fixed here, not CLI options."""
from langchain.agents import create_agent

from decomposer.core import create_decomposer_agent
from gyms.wideseek import agentic_rag
from gyms.wideseek.runtime import Context, ModelLog, model

REACT_MODEL = "lmrouter/occ_4b_non_thinking"
DECOMPOSER_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
RESEARCHER_MODEL = REACT_MODEL
STUDENT_MODEL = "vllm/qwen_3_5_4b_non_thinking_rl_sampling"
JUDGE_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
AGENT_MODELS = {"react": REACT_MODEL, "decomposer": DECOMPOSER_MODEL, "student": STUDENT_MODEL}
SUBAGENT_MODELS = dict.fromkeys(AGENT_MODELS, RESEARCHER_MODEL)
RESEARCHER_IDS = dict.fromkeys(("decomposer", "student"), "researcher")
AGENT_RUN_BUDGET_SECONDS = 180.0
AGENT_SHUTDOWN_GRACE_SECONDS = 180.0
RUN_BUDGETS = dict.fromkeys(AGENT_MODELS, (AGENT_RUN_BUDGET_SECONDS, AGENT_SHUTDOWN_GRACE_SECONDS))
RESEARCHER_RECURSION_LIMITS = dict.fromkeys(AGENT_MODELS, 10000)


def researcher():
    return react(model(RESEARCHER_MODEL), None)


def react(policy, checkpointer, *, middleware=()):
    return create_agent(policy, tools=agentic_rag.TOOLS, system_prompt=agentic_rag.SYSTEM_PROMPT,
        context_schema=Context, middleware=[agentic_rag.ResearchProtocol(middleware), ModelLog("researcher"), *middleware],
        checkpointer=checkpointer)


def decomposer(policy, checkpointer, worker_url, *, middleware=(), agent_recursion_limit=None,
               agent="decomposer"):
    run_budget, shutdown_grace = RUN_BUDGETS[agent]
    agent_recursion_limit = agent_recursion_limit or RESEARCHER_RECURSION_LIMITS.get(agent, 410)
    return create_decomposer_agent(decomposer_model=policy,
        agent_types=[{"agent_type_id": "occ_4b_non_thinking", "assistant_id": RESEARCHER_IDS[agent],
            "description": "OCC-4B non-thinking agent equipped with search, open, find and finish tools.",
            "url": worker_url}],
        checkpointer=checkpointer, middleware=[ModelLog("decomposer"), *middleware],
        context_schema=Context, agent_recursion_limit=agent_recursion_limit,
        agent_run_budget_seconds=run_budget,
        agent_shutdown_grace_seconds=shutdown_grace)
