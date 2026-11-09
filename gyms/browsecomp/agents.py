import os
from pathlib import Path

from langchain.agents import create_agent
from decomposer.core import create_decomposer_agent
from decomposer.models import create_model
from decomposer.prompts import AGENT_SYSTEM_PROMPT
from decomposer.run_budget import RunBudgetContext, RunBudgetMiddleware
from gyms.browsecomp.model_logging import log_model_call
from gyms.browsecomp.tools import ResearchState, search, fetch

RESEARCHER_MODEL = "lmrouter/qwen_3_5_4b_thinking"
DECOMPOSER_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
JUDGE_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"


def _researcher(model_id, *, prompt=None):
    return create_agent(
        create_model(model_id), tools=[search, fetch],
        state_schema=ResearchState,
        context_schema=RunBudgetContext,
        system_prompt=(prompt if prompt is not None else Path(__file__).with_name("prompt.txt").read_text()) + "\n\n" + AGENT_SYSTEM_PROMPT,
        middleware=[RunBudgetMiddleware(), log_model_call],
    )


def _decomposer(model_id):
    return create_decomposer_agent(
        decomposer_model=create_model(model_id),
        agent_types=[{
            "agent_type_id": "qwen_3_5_4b_thinking", "assistant_id": "researcher",
            "description": "Qwen3.5-4B thinking agent equipped with search and access tools.",
            "url": os.environ["BROWSECOMP_AGENT_URL"],
        }],
        agent_recursion_limit=410, middleware=[log_model_call],
    )


def researcher():
    return _researcher(RESEARCHER_MODEL)


def decomposer():
    return _decomposer(DECOMPOSER_MODEL)


def researcher_4b_thinking():
    return _researcher(RESEARCHER_MODEL)


def researcher_27b_thinking():
    return _researcher("lmrouter/qwen_3_8_27b_medium_thinking")


def decomposer_4b_non_thinking():
    return _decomposer("lmrouter/qwen_3_5_4b_non_thinking")


def decomposer_sft_1_0_0():
    return _decomposer("vllm/decomposer_manager_qwen35_4b_sft_1_0_0")


def decomposer_sft_1_2_0():
    return _decomposer("vllm/decomposer_manager_qwen35_4b_sft_1_2_0")


def decomposer_sft_1_4_0():
    return _decomposer("vllm/decomposer_manager_qwen35_4b_sft_1_4_0")


# Model and harness recorded in traces, indexed by the LangGraph assistant ID.
AGENTS = {
    "researcher": (RESEARCHER_MODEL, "react"),
    "decomposer": (DECOMPOSER_MODEL, "decomposer"),
    "researcher_4b_thinking": (RESEARCHER_MODEL, "react"),
    "researcher_27b_thinking": ("lmrouter/qwen_3_8_27b_medium_thinking", "react"),
    "decomposer_4b_non_thinking": ("lmrouter/qwen_3_5_4b_non_thinking", "decomposer"),
    "decomposer_sft_1_0_0": ("vllm/decomposer_manager_qwen35_4b_sft_1_0_0", "decomposer"),
    "decomposer_sft_1_2_0": ("vllm/decomposer_manager_qwen35_4b_sft_1_2_0", "decomposer"),
    "decomposer_sft_1_4_0": ("vllm/decomposer_manager_qwen35_4b_sft_1_4_0", "decomposer"),
}
