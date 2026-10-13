from types import SimpleNamespace

from gyms.gaia2 import prompts as worker_prompt


def _scenario(additional="Scenario-specific instructions."):
    return SimpleNamespace(start_time=1728975600.0, additional_system_prompt=additional)


def test_worker_prompt_preserves_our_instructions_and_scenario_context():
    from decomposer.prompts import AGENT_SYSTEM_PROMPT

    prompt = worker_prompt.render_worker_system_prompt(_scenario())
    assert prompt.startswith(
        "The current date and time in the environment is 2024-10-15 07:00:00 UTC.\n\n"
        + AGENT_SYSTEM_PROMPT
    )
    assert "Scenario-specific instructions." in prompt
    for obsolete in (
        "## Environment",
        "Other agents",
        "write as the user",
        "keep going until the whole task is done",
        "Only message the user when completely done",
        "Try alternative approaches before reporting failure",
        "send_message_to_user",
        "manager",
        "Return your response to the current task in your final message.",
        "Messages, Chats, and Email send messages to contacts.",
    ):
        assert obsolete not in prompt


def test_worker_prompt_without_scenario_instructions():
    prompt = worker_prompt.render_worker_system_prompt(_scenario(None))
    assert "{current_time}" not in prompt
    assert "None" not in prompt
