"""System prompts for the Decomposer's GAIA2 workers.

The workers are the agents that act in the ARE environment, as ARE's native agent
does in the simple baseline. They get that agent's system prompt, rendered for the
scenario exactly as ARE renders it (`are_simulation_main.py:init_system_prompt` and
the native-tools renderer), with one change: ARE's paragraph telling the agent to
answer through `AgentUserInterface__send_message_to_user` is replaced, because a
worker cannot call that tool and its final message goes back to the manager.

Rendering needs ARE and runs in the ARE process (the proxy); ARE is imported lazily
so the worker graph can import `LEGACY_WORKER_SYSTEM_PROMPT` without it.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

# The worker prompt before `are_native` (1b11854), kept to reproduce older runs.
LEGACY_WORKER_SYSTEM_PROMPT = (
    "You are a GAIA2 worker. Solve the delegated subtask using the supplied "
    "ARE tools. State is shared between workers and tool calls are serialized. "
    "Never call the user-interface final-response tool; report findings and "
    "actions back to the manager. SystemApp__wait_for_notification is the "
    "canonical way to advance simulated time."
)

# ARE sends its system prompt through this message template
# (agents/default_agent/base_agent.py DEFAULT_STEP_2_MESSAGE["system_prompt"]).
ARE_SYSTEM_MESSAGE_TEMPLATE = "{content}\n"

# Where ARE's native-tools block turns from tool use to the user channel.
_USER_CHANNEL_PARAGRAPH_START = "\n\nTo say anything to the user"

WORKER_CHANNEL_PARAGRAPH = (
    "Your final message is returned to the manager who delegated this task to you; "
    "only the manager communicates with the user, and you cannot message the user "
    "yourself. Where these instructions say to message the user or to ask the user "
    "for clarification, state it in your final message instead. Other workers may "
    "act on the same environment at the same time."
)


def _agent_instructions() -> tuple[str, str]:
    """ARE's native-tools block and the worker's version of it."""

    from are.simulation.agents.default_agent.prompts.system_prompt import (
        NATIVE_TOOLS_LOOP_SYSTEM_PROMPT,
    )

    if NATIVE_TOOLS_LOOP_SYSTEM_PROMPT.count(_USER_CHANNEL_PARAGRAPH_START) != 1:
        raise ValueError(
            "ARE's native-tools prompt no longer has exactly one user-channel "
            "paragraph; review gyms/gaia2/worker_prompt.py"
        )
    tool_use, _ = NATIVE_TOOLS_LOOP_SYSTEM_PROMPT.split(_USER_CHANNEL_PARAGRAPH_START)
    return NATIVE_TOOLS_LOOP_SYSTEM_PROMPT, f"{tool_use}\n\n{WORKER_CHANNEL_PARAGRAPH}"


def render_native_system_prompt(scenario: Any, notification_system: Any) -> str:
    """The system message ARE's `native_tools` agent's model gets for this scenario."""

    from are.simulation.agents.default_agent.prompts.notification_system import (
        get_notification_system_prompt,
    )
    from are.simulation.agents.default_agent.prompts.system_prompt import (
        DEFAULT_ARE_SIMULATION_NATIVE_SYSTEM_PROMPT,
    )
    from are.simulation.agents.default_agent.tools.native_tools import (
        make_native_update_system_prompt_tools,
    )

    # Same steps, in the same order, as ARESimulationAgent.init_system_prompt.
    prompt = str(DEFAULT_ARE_SIMULATION_NATIVE_SYSTEM_PROMPT)
    additional = getattr(scenario, "additional_system_prompt", None)
    if additional is not None:
        prompt += "\n\n" + additional
    extra = os.environ.get("ARE_EXTRA_SYSTEM_PROMPT")
    extra_file = os.environ.get("ARE_EXTRA_SYSTEM_PROMPT_FILE")
    if not extra and extra_file and os.path.exists(extra_file):
        with open(extra_file) as handle:
            extra = handle.read()
    if extra:
        prompt += "\n\n" + extra
    prompt = prompt.replace(
        "<<notification_system_description>>",
        get_notification_system_prompt(notification_system, getattr(scenario, "apps", None)),
    )
    date_str = datetime.fromtimestamp(
        getattr(scenario, "start_time", None) or 0, tz=timezone.utc
    ).strftime("%Y-%m-%d %H")
    prompt = prompt.replace(
        "<<curent_time_description>>",
        f"Today's date in 'YYYY-MM-DD HH' format is {date_str}",
    )
    prompt = prompt.replace("<<agent_reminder_description>>", "")
    # The native agent's first initialize() swaps the text tool list for this note.
    prompt = make_native_update_system_prompt_tools()({"system_prompt": prompt}, [])[
        "system_prompt"
    ]
    return ARE_SYSTEM_MESSAGE_TEMPLATE.format(content=prompt)


def render_worker_system_prompt(scenario: Any, notification_system: Any) -> str:
    """ARE's native system prompt with the user-channel paragraph replaced."""

    native_block, worker_block = _agent_instructions()
    prompt = render_native_system_prompt(scenario, notification_system)
    if prompt.count(native_block) != 1:
        raise ValueError(
            "ARE's native system prompt no longer contains its native-tools block "
            "exactly once; review gyms/gaia2/worker_prompt.py"
        )
    return prompt.replace(native_block, worker_block)
