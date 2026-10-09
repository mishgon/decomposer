"""Named WideSeek agents; model profiles are fixed here, not CLI options."""
from langchain.agents import create_agent

from decomposer.core import create_decomposer_agent
from gyms.wideseek.runtime import Context, ModelLog, model
from gyms.wideseek.worker import SYSTEM_PROMPT, access, search

REACT_MODEL = "lmrouter/qwen_3_5_4b_unlooped_non_thinking"
DECOMPOSER_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
RESEARCHER_MODEL = "lmrouter/qwen_3_5_4b_unlooped_thinking"
JUDGE_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
AGENT_MODELS = {"react": REACT_MODEL, "decomposer": DECOMPOSER_MODEL}
RESEARCHER_ID = "researcher"


def researcher():
    return create_agent(model(RESEARCHER_MODEL), tools=[search, access], context_schema=Context,
        middleware=[ModelLog("researcher")], system_prompt=SYSTEM_PROMPT)


def react(policy, checkpointer, *, middleware=()):
    return create_agent(policy, tools=[search, access], system_prompt=SYSTEM_PROMPT,
        context_schema=Context, middleware=[ModelLog("researcher"), *middleware], checkpointer=checkpointer)


def decomposer(policy, checkpointer, worker_url, *, middleware=()):
    return create_decomposer_agent(decomposer_model=policy,
        agent_types=[{"agent_type_id": RESEARCHER_ID, "assistant_id": RESEARCHER_ID,
            "description": "Qwen3.5-4B unlooped thinking researcher with offline Wiki-2018 search and access tools.",
            "url": worker_url}],
        checkpointer=checkpointer, middleware=[ModelLog("decomposer"), *middleware],
        context_schema=Context, agent_recursion_limit=410)
