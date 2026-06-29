"""Tests for shared ReAct tool-calling behavior."""

from __future__ import annotations

from claw_eval.config import ReActRunConfig
from claw_eval.models.content import TextBlock, ToolUseBlock
from claw_eval.models.message import Message
from claw_eval.models.task import Prompt, TaskDefinition
from claw_eval.models.tool import ToolSpec
from claw_eval.models.trace import TokenUsage, ToolDispatch
from claw_eval.runner.executor_report_tools import SUBMIT_EXECUTOR_REPORT_TOOL
from claw_eval.runner.loop_common import make_local_tool_result
from claw_eval.runner.react_loop import run_react_loop


class FakeProvider:
    def __init__(self, responses: list[tuple[Message, TokenUsage]]) -> None:
        self.model_id = "fake"
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("FakeProvider has no response queued")
        return self.responses.pop(0)


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, tool_use, trace_id):
        self.calls.append(tool_use)
        return (
            make_local_tool_result(tool_use, "ok"),
            ToolDispatch(
                trace_id=trace_id,
                tool_use_id=tool_use.id,
                tool_name=tool_use.name,
                endpoint_url="local://test",
            ),
        )


def _task_with_tool() -> tuple[TaskDefinition, list[ToolSpec]]:
    tool = ToolSpec(
        name="lookup",
        description="Look up a record",
        input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
    )
    task = TaskDefinition(
        task_id="T_test",
        task_name="Test",
        prompt=Prompt(text="Use the lookup tool, then report."),
        tools=[tool],
    )
    return task, [tool]


def test_react_loop_requires_first_tool_then_allows_final_text():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (Message(role="assistant", content=[ToolUseBlock(id="call_1", name="lookup", input={"id": "1"})]), TokenUsage(input_tokens=10, output_tokens=5)),
        (Message(role="assistant", content=[TextBlock(text="done")]), TokenUsage(input_tokens=12, output_tokens=2)),
    ])

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=FakeDispatcher(),
        trace_id="trace",
        max_turns=3,
        timeout_seconds=30,
        min_environment_tool_calls=1,
    )

    assert result.final_report == "done"
    assert provider.calls[0]["kwargs"]["tool_choice"] == "required"
    assert provider.calls[1]["kwargs"]["tool_choice"] is None


def test_react_loop_zero_min_tool_calls_does_not_force_first_tool():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (Message(role="assistant", content=[TextBlock(text="done")]), TokenUsage(input_tokens=10, output_tokens=2)),
    ])

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=FakeDispatcher(),
        trace_id="trace",
        max_turns=3,
        timeout_seconds=30,
    )

    assert result.final_report == "done"
    assert result.environment_tool_count == 0
    assert len(provider.calls) == 1
    assert provider.calls[0]["kwargs"]["tool_choice"] is None


def test_react_loop_does_not_retry_empty_response_by_default():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (Message(role="assistant", content=[]), TokenUsage(input_tokens=10, output_tokens=3)),
    ])

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=FakeDispatcher(),
        trace_id="trace",
        max_turns=3,
        timeout_seconds=30,
    )

    assert result.stopped_reason == "no_tools"
    assert result.final_report == ""
    assert result.empty_visible_response_count == 1
    assert len(provider.calls) == 1
    assert all("Protocol error" not in msg.text for msg in result.messages)


def test_react_loop_retries_once_on_empty_required_tool_response():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (Message(role="assistant", content=[]), TokenUsage(input_tokens=10, output_tokens=3)),
        (Message(role="assistant", content=[ToolUseBlock(id="call_1", name="lookup", input={"id": "1"})]), TokenUsage(input_tokens=11, output_tokens=5)),
        (Message(role="assistant", content=[TextBlock(text="done")]), TokenUsage(input_tokens=12, output_tokens=2)),
    ])

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=FakeDispatcher(),
        trace_id="trace",
        max_turns=4,
        timeout_seconds=30,
        min_environment_tool_calls=1,
        react_cfg=ReActRunConfig(retry_empty_model_response=True),
    )

    assert result.final_report == "done"
    assert len(provider.calls) == 3
    assert provider.calls[0]["kwargs"]["tool_choice"] == "required"
    assert provider.calls[1]["kwargs"]["tool_choice"] == "required"
    assert "Protocol error" in provider.calls[1]["messages"][-1].text


def test_react_loop_retries_once_on_empty_post_tool_response():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (Message(role="assistant", content=[ToolUseBlock(id="call_1", name="lookup", input={"id": "1"})]), TokenUsage(input_tokens=10, output_tokens=5)),
        (Message(role="assistant", content=[]), TokenUsage(input_tokens=11, output_tokens=9)),
        (Message(role="assistant", content=[TextBlock(text="done")]), TokenUsage(input_tokens=12, output_tokens=2)),
    ])

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=FakeDispatcher(),
        trace_id="trace",
        max_turns=4,
        timeout_seconds=30,
        react_cfg=ReActRunConfig(retry_empty_model_response=True),
    )

    assert result.final_report == "done"
    assert len(provider.calls) == 3
    assert provider.calls[1]["kwargs"]["tool_choice"] is None
    assert provider.calls[2]["kwargs"]["tool_choice"] is None
    assert "after receiving a tool result" in provider.calls[2]["messages"][-1].text


def test_react_loop_retries_plain_text_before_required_environment_tool():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (Message(role="assistant", content=[TextBlock(text="I can answer without tools.")]), TokenUsage(input_tokens=10, output_tokens=4)),
        (Message(role="assistant", content=[ToolUseBlock(id="call_1", name="lookup", input={"id": "1"})]), TokenUsage(input_tokens=11, output_tokens=5)),
        (Message(role="assistant", content=[TextBlock(text="done")]), TokenUsage(input_tokens=12, output_tokens=2)),
    ])

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=FakeDispatcher(),
        trace_id="trace",
        max_turns=4,
        timeout_seconds=30,
        min_environment_tool_calls=1,
        react_cfg=ReActRunConfig(retry_missing_required_tool=True),
    )

    assert result.final_report == "done"
    assert result.environment_tool_count == 1
    assert provider.calls[0]["kwargs"]["tool_choice"] == "required"
    assert provider.calls[1]["kwargs"]["tool_choice"] == "required"
    assert "at least 1 environment tool" in provider.calls[1]["messages"][-1].text


def test_react_loop_does_not_retry_missing_required_tool_by_default():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (Message(role="assistant", content=[TextBlock(text="I can answer without tools.")]), TokenUsage(input_tokens=10, output_tokens=4)),
    ])

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=FakeDispatcher(),
        trace_id="trace",
        max_turns=4,
        timeout_seconds=30,
        min_environment_tool_calls=1,
    )

    assert result.stopped_reason == "missing_required_tool"
    assert result.environment_tool_count == 0
    assert len(provider.calls) == 1
    assert provider.calls[0]["kwargs"]["tool_choice"] == "required"
    assert all("Protocol error" not in msg.text for msg in result.messages)


def test_react_loop_caps_environment_tool_dispatches():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (
            Message(role="assistant", content=[
                ToolUseBlock(id="call_1", name="lookup", input={"id": "1"}),
                ToolUseBlock(id="call_2", name="lookup", input={"id": "2"}),
            ]),
            TokenUsage(input_tokens=10, output_tokens=8),
        ),
    ])
    dispatcher = FakeDispatcher()

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=dispatcher,
        trace_id="trace",
        max_turns=3,
        timeout_seconds=30,
        max_environment_tool_calls=1,
    )

    assert result.stopped_reason == "tool_budget"
    assert result.tool_budget_exhausted is True
    assert result.environment_tool_count == 1
    assert result.environment_tool_names == ["lookup"]
    assert len(dispatcher.calls) == 1
    assert "tool budget exhausted" in result.messages[-1].content[1].content[0].text


def test_react_loop_retries_transitional_text_after_tool_result():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (Message(role="assistant", content=[ToolUseBlock(id="call_1", name="lookup", input={"id": "1"})]), TokenUsage(input_tokens=10, output_tokens=5)),
        (Message(role="assistant", content=[TextBlock(text="I found one record. Let me retrieve the details.")]), TokenUsage(input_tokens=12, output_tokens=8)),
        (Message(role="assistant", content=[ToolUseBlock(id="call_2", name="lookup", input={"id": "2"})]), TokenUsage(input_tokens=13, output_tokens=5)),
        (Message(role="assistant", content=[TextBlock(text="done")]), TokenUsage(input_tokens=14, output_tokens=2)),
    ])
    dispatcher = FakeDispatcher()

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=dispatcher,
        trace_id="trace",
        max_turns=5,
        timeout_seconds=30,
        react_cfg=ReActRunConfig(
            retry_transitional_tool_text=True,
            transitional_tool_retry_limit=1,
        ),
    )

    assert result.final_report == "done"
    assert result.transitional_tool_retry_count == 1
    assert result.environment_tool_count == 2
    assert len(dispatcher.calls) == 2
    assert provider.calls[1]["kwargs"]["tool_choice"] is None
    assert provider.calls[2]["kwargs"]["tool_choice"] == "required"
    assert "did not include a native tool call" in provider.calls[2]["messages"][-1].text


def test_react_loop_stops_when_transitional_retry_budget_exhausted():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (Message(role="assistant", content=[ToolUseBlock(id="call_1", name="lookup", input={"id": "1"})]), TokenUsage(input_tokens=10, output_tokens=5)),
        (Message(role="assistant", content=[TextBlock(text="I found one record. Let me retrieve the details.")]), TokenUsage(input_tokens=12, output_tokens=8)),
        (Message(role="assistant", content=[TextBlock(text="Let me retrieve the details now.")]), TokenUsage(input_tokens=13, output_tokens=6)),
    ])

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools,
        dispatcher=FakeDispatcher(),
        trace_id="trace",
        max_turns=5,
        timeout_seconds=30,
        react_cfg=ReActRunConfig(
            retry_transitional_tool_text=True,
            transitional_tool_retry_limit=1,
        ),
    )

    assert result.stopped_reason == "transitional_tool_retry_exhausted"
    assert result.transitional_tool_retry_count == 1
    assert result.environment_tool_count == 1
    assert provider.calls[2]["kwargs"]["tool_choice"] == "required"


def test_react_loop_submit_report_is_local_and_not_environment_tool():
    task, tools = _task_with_tool()
    provider = FakeProvider([
        (
            Message(role="assistant", content=[
                ToolUseBlock(id="call_1", name="lookup", input={"id": "1"}),
            ]),
            TokenUsage(input_tokens=10, output_tokens=5),
        ),
        (
            Message(role="assistant", content=[
                ToolUseBlock(
                    id="call_2",
                    name="submit_report",
                    input={
                        "what_did": "looked up record 1",
                        "key_findings": "invoice #123",
                        "tool_results": "lookup returned ok",
                        "blockers": "none",
                    },
                ),
            ]),
            TokenUsage(input_tokens=12, output_tokens=5),
        ),
    ])
    dispatcher = FakeDispatcher()

    result = run_react_loop(
        initial_messages=[Message(role="user", content=[TextBlock(text="go")])],
        task=task,
        provider=provider,
        task_tools=tools + [SUBMIT_EXECUTOR_REPORT_TOOL],
        dispatcher=dispatcher,
        trace_id="trace",
        max_turns=4,
        timeout_seconds=30,
        min_environment_tool_calls=1,
        enable_submit_report_tool=True,
    )

    assert result.stopped_reason == "submitted_report"
    assert result.submitted_report_count == 1
    assert "invoice #123" in result.submitted_report
    assert result.environment_tool_count == 1
    assert result.environment_tool_names == ["lookup"]
    assert len(dispatcher.calls) == 1
    assert result.assistant_tool_message_count == 2
    assert result.empty_visible_response_count == 0
    assert result.environment_tool_summaries == ["lookup [OK]: ok"]
