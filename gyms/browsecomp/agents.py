import os
from pathlib import Path

from langchain.agents import create_agent
from decomposer.core import create_decomposer_agent
from decomposer.models import create_model
from gyms.browsecomp.model_logging import log_model_call
from gyms.browsecomp.tools import ResearchState, search, fetch

RESEARCHER_MODEL = "lmrouter/qwen_3_5_4b_unlooped_thinking"
DECOMPOSER_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"
JUDGE_MODEL = "lmrouter/qwen_3_8_flash_next_non_thinking"


def researcher():
    return create_agent(
        create_model(RESEARCHER_MODEL), tools=[search, fetch],
        state_schema=ResearchState,
        system_prompt=Path(__file__).with_name("prompt.txt").read_text(),
        middleware=[log_model_call],
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
