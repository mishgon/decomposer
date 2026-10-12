import ast
import re
from pathlib import Path

import pytest

from gyms.gaia2 import are as trace

ARE_AGENT_DIR = (
    Path(__file__).parents[2]
    / "external"
    / "gaia2"
    / "are"
    / "simulation"
    / "agents"
    / "default_agent"
)
AUI_MESSAGE = (
    "Received at: 2026-01-01 09:00:00\nSender: User\n"
    "Message: Send the Q3 report to Ana.\nAlready read: False"
)


def test_uncollected_subagent_is_reported_as_outstanding():
    summaries, outstanding = trace._agent_summary(
        {
            "agents": {
                "s1": {"agent_id": "s1", "agent_type_id": "worker"},
            },
            "agent_runs": {
                "running": {"agent_id": "s1", "status": "running"},
                "uncollected": {
                    "agent_id": "s1",
                    "status": "responded",
                    "response": "ok",
                },
                "collected": {
                    "agent_id": "s1",
                    "status": "responded",
                    "response": "ok",
                    "response_sequence_number": 1,
                    "messages": [
                        {
                            "type": "ai",
                            "content": "",
                            "additional_kwargs": {"reasoning_content": "private"},
                        }
                    ],
                },
            },
        }
    )
    assert outstanding == ["running", "uncollected"]
    assert all("messages" not in summary for summary in summaries)
    # The type lives on the persistent subagent, not on the run.
    assert {summary["agent_type_id"] for summary in summaries} == {"worker"}
    assert summaries[2]["response"] == "ok"


def _context():
    return {
        "tool_schemas": [],
        "broker_url": "http://broker/session",
        "session_token": "secret",
        "policy": "shared_serialized",
        "scenario_id": "scenario",
        "run_number": 1,
        "notification_cursor": 0,
        "worker_system_prompt": "GAIA2 worker prompt",
    }


def _notification(
    kind: str, message: str, timestamp: str = "2026-01-01T09:00:00+00:00"
):
    return {"type": kind, "message": message, "simulated_timestamp": timestamp}


def _request(*notifications):
    return list(notifications)


def test_turn_templates_match_ares_own_agent():
    base_agent = ARE_AGENT_DIR / "base_agent.py"
    steps = ARE_AGENT_DIR / "steps" / "are_simulation.py"
    if not base_agent.is_file() or not steps.is_file():
        pytest.skip("external/gaia2 is not checked out")
    tree = ast.parse(base_agent.read_text(encoding="utf-8"))
    (templates,) = [
        dict(ast.literal_eval(node.value.args[0]))
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", None) == "DEFAULT_STEP_2_MESSAGE"
            for target in node.targets
        )
    ]

    assert trace.ARE_TASK_TEMPLATE == templates["task"]
    assert (
        trace.ARE_ENVIRONMENT_NOTIFICATIONS_TEMPLATE
        == templates["environment_notifications"]
    )
    (time_format,) = re.findall(
        r"def format_notification.*?strftime\('([^']+)'\)",
        steps.read_text(encoding="utf-8"),
        flags=re.S,
    )
    assert trace.ARE_NOTIFICATION_TIME_FORMAT == time_format


def test_user_turn_is_ares_task_text_and_nothing_else():
    messages = trace._turn_messages(
        _request(_notification("USER_MESSAGE", AUI_MESSAGE))
    )

    assert messages == [{"role": "user", "content": f"[TASK]: \n{AUI_MESSAGE}\n"}]
    assert not re.search(r"GAIA2|Delegate|turn 3|subagent", messages[0]["content"])


def test_several_user_messages_join_into_one_task():
    messages = trace._turn_messages(
        _request(
            _notification("USER_MESSAGE", "first"),
            _notification("USER_MESSAGE", "second"),
        )
    )

    assert messages == [{"role": "user", "content": "[TASK]: \nfirst\nsecond\n"}]


def test_environment_notifications_follow_as_ares_update_block():
    messages = trace._turn_messages(
        _request(
            _notification("USER_MESSAGE", "task"),
            _notification(
                "ENVIRONMENT_NOTIFICATION",
                "EmailClientV2: New email received from ana@example.com",
                "2026-01-01T09:05:30+00:00",
            ),
            _notification("ENVIRONMENT_STOP", "stop"),
        )
    )

    assert messages == [
        {"role": "user", "content": "[TASK]: \ntask\n"},
        {
            "role": "user",
            "content": (
                "Environment notifications updates:\n***\n"
                "[2026-01-01 09:05:30] EmailClientV2: New email received from ana@example.com"
                "\n***\n"
            ),
        },
    ]


def test_environment_only_turn_keeps_ares_empty_task():
    messages = trace._turn_messages(
        _request(_notification("ENVIRONMENT_NOTIFICATION", "Chats: New message"))
    )

    assert messages[0] == {"role": "user", "content": "[TASK]: \n\n"}
    assert messages[1]["content"].startswith(
        "Environment notifications updates:\n***\n[2026-01-01 09:00:00]"
    )


def test_public_context_records_only_a_hash_of_the_worker_prompt():
    context = {**_context(), "worker_system_prompt": "ARE native prompt"}

    public = trace._public_context(context)

    assert "worker_system_prompt" not in public
    assert public["worker_system_prompt_sha256"] == (
        __import__("hashlib").sha256(b"ARE native prompt").hexdigest()
    )
    assert "worker_system_prompt_sha256" in trace._public_context(_context())


@pytest.mark.parametrize(
    "content, expected",
    [
        ("answer", "answer"),
        (
            [
                {"type": "reasoning", "text": "private"},
                {"type": "text", "text": "answer"},
            ],
            "answer",
        ),
        ([{"type": "refusal", "refusal": "cannot"}], "cannot"),
    ],
)
def test_final_text_preserves_visible_blocks(content, expected):
    message = {"type": "ai", "content": content, "usage_metadata": {"output_tokens": 7}}
    result = trace.turn_result({"messages": [message]}, _context(), 2)
    assert result["final_text"] == expected
    assert result["usage"] == {"output_tokens": 7}
    assert result["trace"]["manager_messages"] == [{"type": "ai", "data": message}]


def test_reasoning_only_final_is_rejected():
    with pytest.raises(RuntimeError, match="no visible final"):
        trace.turn_result(
            {
                "messages": [
                    {
                        "type": "ai",
                        "content": [{"type": "reasoning", "text": "private"}],
                    }
                ]
            },
            _context(),
            0,
        )
