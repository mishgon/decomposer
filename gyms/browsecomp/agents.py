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
QWEN27_MODEL = "lmrouter/qwen_3_8_27b_medium_thinking"
SFT_MODEL = "qwen35-4b-manager-sft-2.1.0-checkpoint-8988"
FULL_SFT_MODEL = "qwen35-4b-manager-full-sft-2.1.0-checkpoint-8980"
AGENT_MODELS = {"researcher": RESEARCHER_MODEL, "decomposer": DECOMPOSER_MODEL,
                "decomposer_qwen4b": BASE_DECOMPOSER_MODEL,
                "researcher_flash_next": DECOMPOSER_MODEL,
                "researcher_qwen27b": QWEN27_MODEL,
                "decomposer_sft": SFT_MODEL,
                "decomposer_full_sft": FULL_SFT_MODEL}
REACT_AGENTS = {"researcher", "researcher_flash_next", "researcher_qwen27b"}


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


def researcher_qwen27b():
    return create_agent(
        create_model(QWEN27_MODEL), tools=[search, fetch],
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


def decomposer_sft():
    # Shared deployment: goncharov-qwen35-4b-sft-8988, checkpoint-8988-merged.
    model = create_model(BASE_DECOMPOSER_MODEL, base_url="http://127.0.0.1:8028/v1")
    return create_decomposer_agent(
        decomposer_model=model,
        agent_types=[{
            "agent_type_id": "researcher", "assistant_id": "researcher",
            "description": "Researcher with search and fetch over the BrowseComp-Plus corpus.",
            "url": os.environ["BROWSECOMP_AGENT_URL"],
        }],
        agent_recursion_limit=410, middleware=[log_model_call],
    )


def decomposer_full_sft():
    model = create_model(BASE_DECOMPOSER_MODEL, base_url="http://127.0.0.1:8029/v1")
    model = model.model_copy(update={"model_name": FULL_SFT_MODEL})
    return create_decomposer_agent(
        decomposer_model=model,
        agent_types=[{
            "agent_type_id": "researcher", "assistant_id": "researcher",
            "description": "Researcher with search and fetch over the BrowseComp-Plus corpus.",
            "url": os.environ["BROWSECOMP_AGENT_URL"],
        }],
        agent_recursion_limit=410, middleware=[log_model_call],
    )
