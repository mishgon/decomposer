"""Named WideSeek agents; model profiles are fixed here, not CLI options."""
from langchain.agents import create_agent

from decomposer.core import create_decomposer_agent
from decomposer.run_budget import RunBudgetMiddleware
from gyms.wideseek.prompts import AGENT_SYSTEM_PROMPT
from gyms.wideseek.runtime import Context, ModelLog, model
from gyms.wideseek.worker import access, search

REACT_MODEL = "lmrouter/qwen_3_5_4b_thinking"
DECOMPOSER_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
RESEARCHER_MODEL = REACT_MODEL
# The student is an SFT manager checkpoint that local vLLM serves as Qwen/Qwen3.5-4B, over
# non-thinking hosted OCC-4b researchers.
STUDENT_MODEL = "vllm/qwen_3_5_4b_non_thinking_rl_sampling"
STUDENT_RESEARCHER_MODEL = "lmrouter/occ_4b_non_thinking"
JUDGE_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
AGENT_MODELS = {"react": REACT_MODEL, "decomposer": DECOMPOSER_MODEL, "student": STUDENT_MODEL}
SUBAGENT_MODELS = {"react": RESEARCHER_MODEL, "decomposer": RESEARCHER_MODEL,
                   "student": STUDENT_RESEARCHER_MODEL}
RESEARCHER_IDS = {"decomposer": "researcher", "student": "researcher_non_thinking"}
AGENT_RUN_BUDGET_SECONDS = 180.0
AGENT_SHUTDOWN_GRACE_SECONDS = 180.0
# Run budget and shutdown grace in seconds. The student's are the ones its training prompt
# names: create_decomposer_agent's defaults, which the SFT teacher collections used.
RUN_BUDGETS = {"react": (AGENT_RUN_BUDGET_SECONDS, AGENT_SHUTDOWN_GRACE_SECONDS),
               "decomposer": (AGENT_RUN_BUDGET_SECONDS, AGENT_SHUTDOWN_GRACE_SECONDS),
               "student": (60.0, 60.0)}


def _researcher(profile):
    return create_agent(model(profile), tools=[search, access], context_schema=Context,
        middleware=[RunBudgetMiddleware(), ModelLog("researcher")],
        system_prompt=AGENT_SYSTEM_PROMPT)


def researcher():
    return _researcher(RESEARCHER_MODEL)


def researcher_non_thinking():
    return _researcher(STUDENT_RESEARCHER_MODEL)


def react(policy, checkpointer, *, middleware=()):
    return create_agent(policy, tools=[search, access], system_prompt=AGENT_SYSTEM_PROMPT,
        context_schema=Context, middleware=[RunBudgetMiddleware(), ModelLog("researcher"), *middleware], checkpointer=checkpointer)


def decomposer(policy, checkpointer, worker_url, *, middleware=(), agent_recursion_limit=410,
               agent="decomposer"):
    # The student keeps the type and description its training records offer.
    run_budget, shutdown_grace = RUN_BUDGETS[agent]
    return create_decomposer_agent(decomposer_model=policy,
        agent_types=[{"agent_type_id": "qwen_3_5_4b_thinking", "assistant_id": RESEARCHER_IDS[agent],
            "description": "Qwen3.5-4B thinking agent equipped with search and access tools.",
            "url": worker_url}],
        checkpointer=checkpointer, middleware=[ModelLog("decomposer"), *middleware],
        context_schema=Context, agent_recursion_limit=agent_recursion_limit,
        agent_run_budget_seconds=run_budget,
        agent_shutdown_grace_seconds=shutdown_grace)
