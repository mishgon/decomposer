"""Reuse BrowseComp agents, changing only the final answer format."""
from pathlib import Path

from gyms.browsecomp import agents
from gyms.browsecomp.agents import decomposer, decomposer_4b_non_thinking, decomposer_sft_1_2_0

PROMPT = Path(agents.__file__).with_name("prompt.txt").read_text().split(
    "When asked for the final benchmark answer,"
)[0] + """When asked for the final benchmark answer, return the requested Markdown table.
When assigned a research subtask, report its findings and sources to the caller.
"""


def researcher():
    return agents._researcher(agents.RESEARCHER_MODEL, prompt=PROMPT)


def researcher_4b_thinking():
    return researcher()


def researcher_27b_thinking():
    return agents._researcher("lmrouter/qwen_3_8_27b_medium_thinking", prompt=PROMPT)
