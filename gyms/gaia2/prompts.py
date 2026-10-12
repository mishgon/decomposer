from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from decomposer.prompts import AGENT_SYSTEM_PROMPT

GAIA2_AGENT_SYSTEM_PROMPT = """The current date and time in the environment is {current_time} UTC.

{agent_system_prompt}"""


def render_worker_system_prompt(scenario: Any) -> str:
    prompt = GAIA2_AGENT_SYSTEM_PROMPT.format(
        agent_system_prompt=AGENT_SYSTEM_PROMPT,
        current_time=datetime.fromtimestamp(
            getattr(scenario, "start_time", None) or 0,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S"),
    )
    additional = getattr(scenario, "additional_system_prompt", None)
    if additional:
        prompt += "\n" + additional
    return prompt
