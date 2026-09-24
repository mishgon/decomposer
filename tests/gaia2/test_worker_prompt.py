from __future__ import annotations

import ast
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

GAIA2_REPO = Path(__file__).parents[2] / "external" / "gaia2"
if not (GAIA2_REPO / "are").is_dir():
    pytest.skip("external/gaia2 is not checked out", allow_module_level=True)
sys.path.insert(0, str(GAIA2_REPO))

from are.simulation.agents.default_agent.prompts import system_prompt  # noqa: E402
from are.simulation.agents.default_agent.prompts.notification_system import (  # noqa: E402
    get_notification_system_prompt,
)

from gyms.gaia2 import worker_prompt  # noqa: E402

START_TIME = 1728975600.0  # 2024-10-15 07:00 UTC


def _scenario(additional: str | None = "Scenario-specific instructions."):
    apps = [SimpleNamespace(name="EmailClientV2"), SimpleNamespace(name="Calendar")]
    return SimpleNamespace(
        start_time=START_TIME, apps=apps, additional_system_prompt=additional
    )


def _notifications(notified_tools=None):
    return SimpleNamespace(
        config=SimpleNamespace(
            notified_tools={"EmailClientV2": ["create_and_add_email"]}
            if notified_tools is None
            else notified_tools
        )
    )


@pytest.fixture(autouse=True)
def _no_extra_prompt(monkeypatch):
    monkeypatch.delenv("ARE_EXTRA_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("ARE_EXTRA_SYSTEM_PROMPT_FILE", raising=False)


def test_native_prompt_is_rendered_like_ares_own_agent():
    scenario, notifications = _scenario(), _notifications()

    prompt = worker_prompt.render_native_system_prompt(scenario, notifications)

    assert "<<" not in prompt
    date = datetime.fromtimestamp(START_TIME, tz=timezone.utc).strftime("%Y-%m-%d %H")
    assert f"Today's date in 'YYYY-MM-DD HH' format is {date}" in prompt
    assert get_notification_system_prompt(notifications, scenario.apps) in prompt
    assert "EmailClientV2__create_and_add_email" in prompt
    assert system_prompt.NATIVE_TOOLS_LOOP_SYSTEM_PROMPT in prompt
    assert "You have a set of tools available via function-calling." in prompt
    assert prompt.endswith("\n\nScenario-specific instructions.\n")
    assert prompt.startswith("<general_instructions>")


def test_worker_prompt_replaces_only_the_user_channel_paragraph():
    scenario, notifications = _scenario(), _notifications()

    native = worker_prompt.render_native_system_prompt(scenario, notifications)
    worker = worker_prompt.render_worker_system_prompt(scenario, notifications)

    tool_use = system_prompt.NATIVE_TOOLS_LOOP_SYSTEM_PROMPT.split(
        "\n\nTo say anything to the user"
    )[0]
    assert worker == native.replace(
        system_prompt.NATIVE_TOOLS_LOOP_SYSTEM_PROMPT,
        f"{tool_use}\n\n{worker_prompt.WORKER_CHANNEL_PARAGRAPH}",
    )
    assert "send_message_to_user" not in worker
    assert "Work step by step" in worker
    assert "ENVIRONMENT CHARACTERISTICS" in worker


def test_scenarios_without_notified_tools_or_extras_render_the_plain_policy():
    scenario, notifications = _scenario(additional=None), _notifications({})

    prompt = worker_prompt.render_native_system_prompt(scenario, notifications)

    assert "environment events will not be notified to you" in prompt
    assert prompt.endswith(
        system_prompt.DEFAULT_ARE_SIMULATION_NATIVE_SYSTEM_PROMPT.rsplit(
            "<<curent_time_description>>"
        )[1]
        + "\n"
    )


def test_extra_system_prompt_follows_the_scenario_prompt_as_in_are(monkeypatch):
    monkeypatch.setenv("ARE_EXTRA_SYSTEM_PROMPT", "Extra run-wide instructions.")

    prompt = worker_prompt.render_native_system_prompt(_scenario(), _notifications())

    assert prompt.endswith(
        "\n\nScenario-specific instructions.\n\nExtra run-wide instructions.\n"
    )


def test_changed_are_native_tools_prompt_fails_loudly(monkeypatch):
    monkeypatch.setattr(
        system_prompt, "NATIVE_TOOLS_LOOP_SYSTEM_PROMPT", "Only one paragraph."
    )

    with pytest.raises(ValueError, match="user-channel paragraph"):
        worker_prompt.render_worker_system_prompt(_scenario(), _notifications())


def test_system_message_template_matches_ares():
    tree = ast.parse(
        (GAIA2_REPO / "are/simulation/agents/default_agent/base_agent.py").read_text(
            encoding="utf-8"
        )
    )
    (templates,) = [
        dict(ast.literal_eval(node.value.args[0]))
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "DEFAULT_STEP_2_MESSAGE" for target in node.targets)
    ]

    assert worker_prompt.ARE_SYSTEM_MESSAGE_TEMPLATE == templates["system_prompt"]

