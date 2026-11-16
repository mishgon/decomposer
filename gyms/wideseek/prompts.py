from external.wideseek_tools.prompt import SYSTEM_PROMPT_WORKER, SYSTEM_PROMPT_SINGLE_AGENT_NOSHOT


# The local model runs without thinking; preserve the remaining upstream role text.
THINKING_INSTRUCTION = "You must conduct reasoning inside <think> and </think> first every time you get new information.\n\n"

CORPUS_INSTRUCTIONS = """The search and access tools use an offline Wikipedia corpus from 2018. Do not claim to have checked sources you did not read.

Search results contain up to 1500 characters per passage. For access, specify
info_to_extract: the question or facts needed from the page. The tool uses Qwen
to extract relevant information from the first 6000 characters. A fact absent
from this excerpt may still exist later in the page; search for other evidence.
Stop searching and return the requested
answer as soon as the retrieved evidence is sufficient. Use exact URLs returned
by search. If a page is missing, search for another URL instead of repeating the
same failed access. Keep a concise running
record of verified facts and source URLs in your messages before more tool calls;
older tool results can be cleared from the model context. If a cleared result is
needed again, repeat its search or access call. Do not invent missing cells.

"""

FINAL_ANSWER_INSTRUCTIONS = """

Follow the format requested by the task, including \\boxed{...} for a short QA answer.
For a final table answer, return only one fenced Markdown block starting with
```markdown and ending with ```. Use the requested column headers exactly.
Do not add commentary outside that block. Mark unavailable facts as unknown
instead of estimating missing values.
"""


AGENT_SYSTEM_PROMPT = (SYSTEM_PROMPT_SINGLE_AGENT_NOSHOT.replace(THINKING_INSTRUCTION, "")
                       .format(FINAL_ANSWER_INSTRUCTIONS) + "\n\n" + CORPUS_INSTRUCTIONS)
WORKER_SYSTEM_PROMPT = (SYSTEM_PROMPT_WORKER.replace(THINKING_INSTRUCTION, "")
                        + "\n\n" + CORPUS_INSTRUCTIONS)
