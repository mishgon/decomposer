"""The single-agent baseline and two managers sharing the same thinking worker."""
import os

from langchain.agents import create_agent
from decomposer.core import create_decomposer_agent
from decomposer.models import create_model
from decomposer.prompts import AGENT_SYSTEM_PROMPT
from gyms.masbench import RULES
from gyms.masbench.model_logging import log_model_call

AGENTS = {
    "single": "vllm/qwen_3_5_4b_thinking",
    "base_manager": "vllm/qwen_3_5_4b_non_thinking",
    "sft_manager": "vllm/decomposer_manager_qwen35_4b_sft_1_5_0",
}


def single():
    return create_agent(
        create_model(AGENTS["single"], base_url=os.environ["MASBENCH_BASE_URL"]),
        tools=[], system_prompt=AGENT_SYSTEM_PROMPT + "\n\n" + RULES,
        middleware=[log_model_call],
    )


def manager(agent):
    url = os.environ["MASBENCH_SFT_URL" if agent == "sft_manager" else "MASBENCH_BASE_URL"]
    return create_decomposer_agent(
        decomposer_model=create_model(AGENTS[agent], base_url=url),
        agent_types=[{
            "agent_type_id": "qwen_3_5_4b_thinking", "assistant_id": "single",
            "description": "Qwen3.5-4B thinking agent for reasoning tasks. No external tools.",
            "url": os.environ["MASBENCH_AGENT_URL"],
        }],
        agent_recursion_limit=410, agent_run_budget_seconds=None,
        middleware=[log_model_call],
    )


def base_manager():
    return manager("base_manager")


def sft_manager():
    return manager("sft_manager")
