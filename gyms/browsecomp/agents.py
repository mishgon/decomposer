import os
from pathlib import Path

from langchain.agents import create_agent
from decomposer.core import create_decomposer_agent
from decomposer.models import create_model
from decomposer.prompts import AGENT_SYSTEM_PROMPT
from decomposer.run_budget import RunBudgetContext, RunBudgetMiddleware
from gyms.browsecomp.model_logging import log_model_call
from gyms.browsecomp.tools import ResearchState, search, fetch

RESEARCHER_MODEL = "lmrouter/qwen_3_5_4b_unlooped_thinking"
DECOMPOSER_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
JUDGE_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
BASE_DECOMPOSER_MODEL = "vllm/qwen_3_5_4b_non_thinking"
BASE_DECOMPOSER_URL = "http://127.0.0.1:8026/v1"
AGENT_MODELS = {"researcher": RESEARCHER_MODEL, "decomposer": DECOMPOSER_MODEL,
                "decomposer_qwen4b": BASE_DECOMPOSER_MODEL,
                "researcher_flash_next": DECOMPOSER_MODEL}
REACT_AGENTS = {"researcher", "researcher_flash_next"}


def researcher():
    return create_agent(
        create_model(RESEARCHER_MODEL), tools=[search, fetch],
        state_schema=ResearchState,
        context_schema=RunBudgetContext,
        system_prompt=Path(__file__).with_name("prompt.txt").read_text() + "\n\n" + AGENT_SYSTEM_PROMPT,
        middleware=[RunBudgetMiddleware(), log_model_call],
    )


def researcher_flash_next():
    return create_agent(
        create_model(DECOMPOSER_MODEL), tools=[search, fetch],
        state_schema=ResearchState,
        context_schema=RunBudgetContext,
        system_prompt=Path(__file__).with_name("prompt.txt").read_text() + "\n\n" + AGENT_SYSTEM_PROMPT,
        middleware=[RunBudgetMiddleware(), log_model_call],
    )


def decomposer():
    return create_decomposer_agent(
        decomposer_model=create_model(DECOMPOSER_MODEL),
        agent_types=[{
            "agent_type_id": "researcher", "assistant_id": "researcher",
            "description": "Researcher with search and fetch over the BrowseComp-Plus corpus.",
            "url": os.environ["BROWSECOMP_AGENT_URL"],
        }],
        agent_recursion_limit=410, middleware=[log_model_call],
    )


def decomposer_qwen4b():
    return create_decomposer_agent(
        decomposer_model=create_model(BASE_DECOMPOSER_MODEL, base_url=BASE_DECOMPOSER_URL),
        agent_types=[{
            "agent_type_id": "researcher", "assistant_id": "researcher",
            "description": "Researcher with search and fetch over the BrowseComp-Plus corpus.",
            "url": os.environ["BROWSECOMP_AGENT_URL"],
        }],
        agent_recursion_limit=410, middleware=[log_model_call],
    )
