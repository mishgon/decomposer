"""Use WideSearch's English instructions and live web tools for all researchers."""
from langchain.agents import create_agent

from gyms.browsecomp import agents
from gyms.browsecomp.agents import decomposer, decomposer_4b_non_thinking, decomposer_sft_1_2_0
from decomposer.models import create_model
from decomposer.prompts import AGENT_SYSTEM_PROMPT
from decomposer.run_budget import RunBudgetContext, RunBudgetMiddleware
from external.WideSearch.src.agent.prompt import get_system_prompt
from gyms.browsecomp.model_logging import log_model_call
from gyms.widesearch.tools import tools

PROMPT = get_system_prompt("en")


def _researcher(model_id, *, subagent=False):
    return create_agent(
        create_model(model_id), tools=tools(), context_schema=RunBudgetContext,
        system_prompt=PROMPT + ("\n\n" + AGENT_SYSTEM_PROMPT if subagent else ""),
        middleware=[RunBudgetMiddleware(), log_model_call],
    )


def researcher():
    return _researcher(agents.RESEARCHER_MODEL, subagent=True)


def researcher_4b_thinking():
    return _researcher(agents.RESEARCHER_MODEL)


def researcher_27b_thinking():
    return _researcher("lmrouter/qwen_3_8_27b_medium_thinking")
