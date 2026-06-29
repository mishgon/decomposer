"""Tests for decomposer control tools and trace merge behavior."""

from __future__ import annotations

import json
import time
from pathlib import Path

from claw_eval.models.content import TextBlock, ToolUseBlock
from claw_eval.models.message import Message
from claw_eval.models.task import Environment, Prompt, TaskDefinition
from claw_eval.models.tool import ToolSpec
from claw_eval.models.trace import DecomposerSummary, DelegationEnd, TokenUsage, ToolDispatch, TraceEnd
from claw_eval.config import DecomposerRunConfig, PromptConfig, ReActRunConfig
from claw_eval.runner import decomposer as decomposer_module
from claw_eval.runner.decomposer import run_decomposer_task
from claw_eval.runner.decomposer_tools import (
    DECOMPOSER_TOOLS,
    DELEGATE_SUBTASK_NAME,
    SUBMIT_FINAL_ANSWER_NAME,
)
from claw_eval.runner.loop_common import make_local_tool_result
from claw_eval.runner.system_prompt import build_system_prompt
from claw_eval.trace.reader import load_trace, load_trace_for_grading, read_events
from claw_eval.trace.writer import TraceWriter


class FakeProvider:
    def __init__(self, model_id: str, responses: list[Message] | None = None) -> None:
        self.model_id = model_id
        self.responses = list(responses or [])
        self.calls = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("FakeProvider has no response queued")
        return self.responses.pop(0), TokenUsage(input_tokens=1, output_tokens=1)


def _task(*, tools: list[ToolSpec] | None = None) -> TaskDefinition:
    return TaskDefinition(
        task_id="T_test",
        task_name="Test",
        prompt=Prompt(text="Do a multi-step task."),
        tools=tools or [],
    )


def _text_response(text: str) -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)])


def _tool_response(name: str, input_: dict, *, call_id: str = "call_1") -> Message:
    return Message(role="assistant", content=[ToolUseBlock(id=call_id, name=name, input=input_)])


def _events(trace_path: Path):
    return list(read_events(trace_path))


def _trace_end(trace_path: Path) -> TraceEnd:
    end = next(event for event in _events(trace_path) if isinstance(event, TraceEnd))
    return end


def _decomposer_summary(trace_path: Path) -> DecomposerSummary:
    summary = next(event for event in _events(trace_path) if isinstance(event, DecomposerSummary))
    return summary


def test_decomposer_tools_schema():
    names = {t.name for t in DECOMPOSER_TOOLS}
    assert names == {DELEGATE_SUBTASK_NAME, SUBMIT_FINAL_ANSWER_NAME}
    delegate = next(t for t in DECOMPOSER_TOOLS if t.name == DELEGATE_SUBTASK_NAME)
    assert "subtask" in delegate.input_schema["properties"]
    assert "subtask" in delegate.input_schema["required"]


def test_make_local_tool_result():
    tu = ToolUseBlock(id="call_1", name="submit_final_answer", input={"answer": "done"})
    result = make_local_tool_result(tu, "ok")
    assert result.tool_use_id == "call_1"
    assert result.content[0].text == "ok"


def test_load_trace_skips_decomposer_events(tmp_path: Path):
    trace_path = tmp_path / "test.jsonl"
    events = [
        {"type": "trace_start", "trace_id": "t1", "task_id": "T001", "model": "big+small", "run_mode": "decomposer"},
        {"type": "message", "trace_id": "t1", "message": {"role": "assistant", "content": [{"type": "text", "text": "final"}]}},
        {"type": "tool_dispatch", "trace_id": "t1", "tool_use_id": "x", "tool_name": "kb_search", "endpoint_url": "http://localhost/kb/search"},
        {"type": "delegation_start", "trace_id": "t1", "delegation_index": 1, "subtask": "search"},
        {"type": "delegation_end", "trace_id": "t1", "delegation_index": 1, "report": "found", "executor_turns": 2},
        {"type": "decomposer_summary", "trace_id": "t1", "decomposer_turns": 2, "delegation_count": 1},
        {"type": "trace_end", "trace_id": "t1", "total_turns": 2},
    ]
    with open(trace_path, "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    start, messages, dispatches, _, end, _ = load_trace(trace_path)
    assert start.run_mode == "decomposer"
    assert len(messages) == 1
    assert messages[-1].message.text == "final"
    assert len(dispatches) == 1
    assert dispatches[0].tool_name == "kb_search"
    assert end is not None


def test_load_trace_for_grading_includes_executor_sidecar_dispatches(tmp_path: Path):
    trace_path = tmp_path / "main.jsonl"
    sidecar_path = tmp_path / "main_exec_1.jsonl"
    main_events = [
        {"type": "trace_start", "trace_id": "t1", "task_id": "T001", "model": "big+small", "run_mode": "decomposer"},
        {"type": "message", "trace_id": "t1", "message": {"role": "assistant", "content": [{"type": "text", "text": "final"}]}},
        {
            "type": "delegation_end",
            "trace_id": "t1",
            "delegation_index": 1,
            "report": "found",
            "executor_turns": 2,
            "sidecar_trace": sidecar_path.name,
        },
        {"type": "trace_end", "trace_id": "t1", "total_turns": 2},
    ]
    sidecar_events = [
        {"type": "trace_start", "trace_id": "exec-1", "task_id": "T001", "model": "small"},
        {"type": "message", "trace_id": "exec-1", "message": {"role": "assistant", "content": [{"type": "text", "text": "hidden executor text"}]}},
        {
            "type": "tool_dispatch",
            "trace_id": "exec-1",
            "tool_use_id": "tool-1",
            "tool_name": "kb_search",
            "endpoint_url": "http://localhost/kb/search",
            "request_body": {"query": "x"},
        },
        {"type": "trace_end", "trace_id": "exec-1", "total_turns": 1},
    ]
    for path, events in [(trace_path, main_events), (sidecar_path, sidecar_events)]:
        with open(path, "w") as fh:
            for ev in events:
                fh.write(json.dumps(ev) + "\n")

    _, main_messages, main_dispatches, _, _, _ = load_trace(trace_path)
    _, grading_messages, grading_dispatches, _, _, _ = load_trace_for_grading(trace_path)

    assert len(main_messages) == 1
    assert len(main_dispatches) == 0
    assert len(grading_messages) == 1
    assert grading_messages[0].message.text == "final"
    assert [d.tool_name for d in grading_dispatches] == ["kb_search"]


def test_react_loop_result_final_report():
    from claw_eval.runner.react_loop import ReActLoopResult

    msgs = [
        Message(role="user", content=[TextBlock(text="go")]),
        Message(role="assistant", content=[TextBlock(text="report text")]),
    ]
    result = ReActLoopResult(messages=msgs, turn_count=1)
    assert result.final_report == "report text"


def test_plain_text_decomposer_response_is_protocol_error_not_final(tmp_path: Path):
    decomposer = FakeProvider("manager", [_text_response("I will delegate this next.")])
    executor = FakeProvider("executor")

    trace_path = run_decomposer_task(
        _task(),
        decomposer,
        executor,
        tmp_path,
        decomposer_cfg=DecomposerRunConfig(max_decomposer_turns=1),
    )

    end = _trace_end(trace_path)
    assert "decomposer_protocol_violation" in end.failure_modes
    assert "no_control_tool_call" in end.failure_modes
    assert "zero_delegations" in end.failure_modes
    assert decomposer.calls[0]["kwargs"]["tool_choice"] == "required"
    assert decomposer.calls[0]["kwargs"]["strict_native_tools"] is True


def test_submit_final_answer_requires_prior_delegation(tmp_path: Path):
    decomposer = FakeProvider(
        "manager",
        [_tool_response(SUBMIT_FINAL_ANSWER_NAME, {"answer": "done"})],
    )
    executor = FakeProvider("executor")

    trace_path = run_decomposer_task(
        _task(),
        decomposer,
        executor,
        tmp_path,
        decomposer_cfg=DecomposerRunConfig(max_decomposer_turns=1),
    )

    end = _trace_end(trace_path)
    assert "final_before_min_delegations" in end.failure_modes
    assert "zero_delegations" in end.failure_modes


def test_empty_submit_final_answer_is_rejected_then_can_recover(tmp_path: Path, monkeypatch):
    decomposer = FakeProvider(
        "manager",
        [
            _tool_response(DELEGATE_SUBTASK_NAME, {"subtask": "Inspect the relevant records."}),
            _tool_response(SUBMIT_FINAL_ANSWER_NAME, {}),
            _tool_response(SUBMIT_FINAL_ANSWER_NAME, {"answer": "Final answer"}),
        ],
    )
    executor = FakeProvider("executor")

    monkeypatch.setattr(
        decomposer_module,
        "_run_executor_delegation",
        lambda **kwargs: ("executor report", decomposer_module._ExecutorStats(1, TokenUsage(), 0.0, 0.0)),
    )

    trace_path = run_decomposer_task(
        _task(),
        decomposer,
        executor,
        tmp_path,
        decomposer_cfg=DecomposerRunConfig(max_decomposer_turns=3),
    )

    events = _events(trace_path)
    end = _trace_end(trace_path)
    assert end.failure_modes == []
    assert len(decomposer.calls) == 3
    rejection_texts = []
    for event in events:
        if event.type != "message":
            continue
        for block in event.message.content:
            if block.type == "tool_result":
                rejection_texts.extend(part.text for part in block.content)
            elif block.type == "text":
                rejection_texts.append(block.text)
    assert any(
        "final answer rejected" in text and "non-empty `answer`" in text
        for text in rejection_texts
    )
    assert any(
        event.type == "message"
        and event.message.role == "assistant"
        and event.message.text == "Final answer"
        for event in events
    )


def test_valid_delegate_then_submit_succeeds_and_counts_executor_turns(tmp_path: Path, monkeypatch):
    decomposer = FakeProvider(
        "manager",
        [
            _tool_response(
                DELEGATE_SUBTASK_NAME,
                {"subtask": "Inspect the relevant records.", "context": "Preserve constraints."},
            ),
            _tool_response(SUBMIT_FINAL_ANSWER_NAME, {"answer": "Final answer"}),
        ],
    )
    executor = FakeProvider("executor")

    seen = {}

    def fake_executor_delegation(**kwargs):
        seen["executor_prompt_mode"] = kwargs["executor_prompt_mode"]
        return "executor report", decomposer_module._ExecutorStats(
            turn_count=2,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            model_time_s=1.0,
            tool_time_s=0.5,
        )

    monkeypatch.setattr(decomposer_module, "_run_executor_delegation", fake_executor_delegation)

    trace_path = run_decomposer_task(
        _task(),
        decomposer,
        executor,
        tmp_path,
        decomposer_cfg=DecomposerRunConfig(
            max_decomposer_turns=3,
            executor_prompt_mode="flat_subtask",
        ),
    )

    assert seen["executor_prompt_mode"] == "flat_subtask"
    events = _events(trace_path)
    end = _trace_end(trace_path)
    summary = _decomposer_summary(trace_path)
    assert end.failure_modes == []
    assert end.total_turns == 4
    assert summary.decomposer_turns == 2
    assert summary.executor_turns == 2
    assert summary.delegation_count == 1
    assert any(event.type == "delegation_start" for event in events)
    delegation_end = next(event for event in events if isinstance(event, DelegationEnd))
    assert delegation_end.report_status == "natural"


def test_executor_system_prompt_matches_flat_prompt_and_includes_mock_today(monkeypatch, tmp_path: Path):
    task = TaskDefinition(
        task_id="T_date",
        task_name="Date Test",
        prompt=Prompt(text="Check tomorrow's meetings."),
        environment=Environment(mock_today="2026-03-26"),
        tools=[ToolSpec(name="calendar_list_events", description="Get events")],
    )
    prompt_cfg = PromptConfig()
    expected_system_prompt = build_system_prompt(task, prompt_cfg)
    seen = {}

    def fake_react_loop(**kwargs):
        messages = kwargs["initial_messages"]
        seen["system_prompt"] = messages[0].text
        return decomposer_module.ReActLoopResult(
            messages=[
                *messages,
                Message(role="assistant", content=[TextBlock(text=(
                    "What I did: checked the date-aware executor prompt.\n"
                    "Key findings: the executor received the flat system prompt with Current date.\n"
                    "Blockers: none."
                ))]),
            ],
            turn_count=1,
        )

    monkeypatch.setattr(decomposer_module, "run_react_loop", fake_react_loop)

    report, _ = decomposer_module._run_executor_delegation(
        task=task,
        subtask="Use calendar_list_events for tomorrow.",
        context="",
        executor_provider=FakeProvider("executor"),
        task_tools=[],
        dispatcher=None,
        sandbox_tool_list=[],
        trace_id="trace",
        main_writer=TraceWriter(tmp_path / "main.jsonl"),
        sidecar_path=tmp_path / "sidecar.jsonl",
        wall_start=time.monotonic(),
        prompt_cfg=prompt_cfg,
        executor_model_cfg=None,
        media_cfg=None,
        max_turns=20,
        timeout_seconds=300,
    )

    assert "Current date" in report
    assert seen["system_prompt"] == expected_system_prompt
    assert "Current date: 2026-03-26" in seen["system_prompt"]


def test_decomposer_receives_only_control_tools(tmp_path: Path, monkeypatch):
    env_tool = ToolSpec(
        name="kb_search",
        description="Search the KB",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    decomposer = FakeProvider(
        "manager",
        [
            _tool_response(DELEGATE_SUBTASK_NAME, {"subtask": "Search the KB."}),
            _tool_response(SUBMIT_FINAL_ANSWER_NAME, {"answer": "done"}),
        ],
    )
    executor = FakeProvider("executor")

    monkeypatch.setattr(
        decomposer_module,
        "_run_executor_delegation",
        lambda **kwargs: ("report", decomposer_module._ExecutorStats(1, TokenUsage(), 0.0, 0.0)),
    )

    run_decomposer_task(
        _task(tools=[env_tool]),
        decomposer,
        executor,
        tmp_path,
        decomposer_cfg=DecomposerRunConfig(max_decomposer_turns=3),
    )

    tool_names = {tool.name for tool in decomposer.calls[0]["tools"]}
    assert tool_names == {DELEGATE_SUBTASK_NAME, SUBMIT_FINAL_ANSWER_NAME}


def test_decomposer_direct_environment_tool_call_is_protocol_error(tmp_path: Path):
    decomposer = FakeProvider(
        "manager",
        [_tool_response("kb_search", {"query": "vpn"})],
    )
    executor = FakeProvider("executor")

    trace_path = run_decomposer_task(
        _task(tools=[
            ToolSpec(
                name="kb_search",
                description="Search the KB",
                input_schema={"type": "object"},
            )
        ]),
        decomposer,
        executor,
        tmp_path,
        decomposer_cfg=DecomposerRunConfig(max_decomposer_turns=1),
    )

    events = _events(trace_path)
    end = _trace_end(trace_path)
    assert "unknown_decomposer_tool" in end.failure_modes
    assert not any(event.type == "tool_dispatch" for event in events)


class ExecutorDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, tool_use, trace_id):
        self.calls.append(tool_use)
        return (
            make_local_tool_result(tool_use, "lookup result: invoice #123"),
            ToolDispatch(
                trace_id=trace_id,
                tool_use_id=tool_use.id,
                tool_name=tool_use.name,
                endpoint_url="local://test",
            ),
        )


def _env_tool() -> ToolSpec:
    return ToolSpec(
        name="kb_search",
        description="Search the KB",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )


def test_executor_delegation_strict_uses_natural_action_loop_report(tmp_path: Path):
    tool = _env_tool()
    natural_report = (
        "- What you did: searched the KB.\n"
        "- Key findings / outputs: invoice #123.\n"
        "- Tool results that matter for next steps: lookup result was relevant.\n"
        "- Blockers / open questions: none."
    )
    executor = FakeProvider(
        "executor",
        [
            _tool_response("kb_search", {"query": "invoice"}),
            _text_response(natural_report),
        ],
    )
    dispatcher = ExecutorDispatcher()
    main_path = tmp_path / "main.jsonl"
    sidecar_path = tmp_path / "exec.jsonl"

    with TraceWriter(main_path) as main_writer:
        report, stats = decomposer_module._run_executor_delegation(
            task=_task(tools=[tool]),
            subtask="Find the invoice.",
            context="",
            executor_provider=executor,
            task_tools=[tool],
            dispatcher=dispatcher,
            sandbox_tool_list=[],
            trace_id="trace",
            main_writer=main_writer,
            sidecar_path=sidecar_path,
            prompt_cfg=None,
            executor_model_cfg=None,
            media_cfg=None,
            max_turns=3,
            timeout_seconds=30,
            wall_start=time.monotonic(),
            min_tool_calls=1,
            max_environment_tool_calls=20,
            report_max_tokens=256,
        )

    assert report == natural_report
    assert stats.turn_count == 2
    assert stats.report_status == "natural"
    assert stats.executor_stopped_reason == "no_tools"
    assert len(executor.calls) == 2
    assert executor.calls[-1]["tools"] == [tool]
    assert not main_path.exists() or "tool_dispatch" not in main_path.read_text()
    assert any(isinstance(event, ToolDispatch) for event in _events(sidecar_path))


def test_executor_delegation_structured_uses_submit_report_tool(tmp_path: Path):
    tool = _env_tool()
    executor = FakeProvider(
        "executor",
        [
            _tool_response("kb_search", {"query": "invoice"}, call_id="call_1"),
            _tool_response(
                "submit_report",
                {
                    "what_did": "searched the KB",
                    "key_findings": "invoice #123",
                    "tool_results": "lookup result was relevant",
                    "blockers": "none",
                },
                call_id="call_2",
            ),
        ],
    )
    dispatcher = ExecutorDispatcher()

    with TraceWriter(tmp_path / "main.jsonl") as main_writer:
        report, stats = decomposer_module._run_executor_delegation(
            task=_task(tools=[tool]),
            subtask="Find the invoice.",
            context="",
            executor_provider=executor,
            task_tools=[tool],
            dispatcher=dispatcher,
            sandbox_tool_list=[],
            trace_id="trace",
            main_writer=main_writer,
            sidecar_path=tmp_path / "exec.jsonl",
            prompt_cfg=None,
            executor_model_cfg=None,
            media_cfg=None,
            max_turns=3,
            timeout_seconds=30,
            wall_start=time.monotonic(),
            min_tool_calls=1,
            max_environment_tool_calls=20,
            report_max_tokens=256,
            report_mode="structured",
        )

    assert "invoice #123" in report
    assert stats.report_status == "structured"
    assert stats.executor_stopped_reason == "submitted_report"
    assert stats.environment_tool_count == 1
    assert stats.environment_tool_names == ["kb_search"]
    assert stats.assistant_tool_message_count == 2
    assert len(dispatcher.calls) == 1
    assert {tool.name for tool in executor.calls[0]["tools"]} == {"kb_search", "submit_report"}
    assert "plain text final reports are invalid" in executor.calls[0]["messages"][-1].text


def test_executor_delegation_passes_compact_tool_evidence(tmp_path: Path):
    tool = _env_tool()
    executor = FakeProvider(
        "executor",
        [
            _tool_response("kb_search", {"query": "invoice"}),
            _text_response("Done."),
        ],
    )
    dispatcher = ExecutorDispatcher()

    with TraceWriter(tmp_path / "main.jsonl") as main_writer:
        report, stats = decomposer_module._run_executor_delegation(
            task=_task(tools=[tool]),
            subtask="Find the invoice.",
            context="",
            executor_provider=executor,
            task_tools=[tool],
            dispatcher=dispatcher,
            sandbox_tool_list=[],
            trace_id="trace",
            main_writer=main_writer,
            sidecar_path=tmp_path / "exec.jsonl",
            prompt_cfg=None,
            executor_model_cfg=None,
            media_cfg=None,
            max_turns=4,
            timeout_seconds=30,
            wall_start=time.monotonic(),
            min_tool_calls=1,
            max_environment_tool_calls=20,
            report_max_tokens=256,
            report_mode="strict",
            evidence_mode="tool_summary",
            evidence_max_chars=500,
        )

    assert stats.report_status == "synthetic_failure"
    assert "Executor attempted the delegated subtask" in report
    assert "[Compact tool evidence passed through from Executor trace]" in report
    assert "kb_search [OK]: lookup result: invoice #123" in report


def test_executor_delegation_runs_no_tools_report_phase(tmp_path: Path):
    tool = _env_tool()
    executor = FakeProvider(
        "executor",
        [
            _tool_response("kb_search", {"query": "invoice"}),
            _text_response("I will prepare the report now."),
            _text_response("- What you did: searched the KB.\n- Key findings / outputs: invoice #123.\n- Tool results that matter for next steps: lookup result was relevant.\n- Blockers / open questions: none."),
        ],
    )
    dispatcher = ExecutorDispatcher()

    with TraceWriter(tmp_path / "main.jsonl") as main_writer:
        report, stats = decomposer_module._run_executor_delegation(
            task=_task(tools=[tool]),
            subtask="Find the invoice.",
            context="",
            executor_provider=executor,
            task_tools=[tool],
            dispatcher=dispatcher,
            sandbox_tool_list=[],
            trace_id="trace",
            main_writer=main_writer,
            sidecar_path=tmp_path / "exec.jsonl",
            prompt_cfg=None,
            executor_model_cfg=None,
            media_cfg=None,
            max_turns=4,
            timeout_seconds=30,
            wall_start=time.monotonic(),
            min_tool_calls=1,
            max_environment_tool_calls=20,
            report_max_tokens=256,
            report_mode="repair",
        )

    assert "invoice #123" in report
    assert stats.turn_count == 3
    assert stats.report_status == "repaired"
    assert len(dispatcher.calls) == 1
    assert executor.calls[-1]["tools"] == []
    assert executor.calls[-1]["kwargs"]["max_tokens"] == 256
    assert "Stop using tools" in executor.calls[-1]["messages"][-1].text


def test_executor_delegation_strict_accepts_transitional_looking_report(tmp_path: Path):
    tool = _env_tool()
    final_report = "I found one result. Let me search for details."
    executor = FakeProvider(
        "executor",
        [
            _tool_response("kb_search", {"query": "invoice"}),
            _text_response(final_report),
        ],
    )
    dispatcher = ExecutorDispatcher()

    with TraceWriter(tmp_path / "main.jsonl") as main_writer:
        report, stats = decomposer_module._run_executor_delegation(
            task=_task(tools=[tool]),
            subtask="Find the invoice.",
            context="",
            executor_provider=executor,
            task_tools=[tool],
            dispatcher=dispatcher,
            sandbox_tool_list=[],
            trace_id="trace",
            main_writer=main_writer,
            sidecar_path=tmp_path / "exec.jsonl",
            prompt_cfg=None,
            executor_model_cfg=None,
            media_cfg=None,
            max_turns=4,
            timeout_seconds=30,
            wall_start=time.monotonic(),
            min_tool_calls=1,
            max_environment_tool_calls=20,
            report_max_tokens=256,
        )

    assert report == final_report
    assert "Executor attempted the delegated subtask" not in report
    assert stats.turn_count == 2
    assert stats.report_status == "natural"
    assert stats.executor_stopped_reason == "no_tools"
    assert len(executor.calls) == 2


def test_executor_delegation_strict_rejects_too_short_report(tmp_path: Path):
    tool = _env_tool()
    executor = FakeProvider(
        "executor",
        [
            _tool_response("kb_search", {"query": "invoice"}),
            _text_response("Done."),
        ],
    )
    dispatcher = ExecutorDispatcher()

    with TraceWriter(tmp_path / "main.jsonl") as main_writer:
        report, stats = decomposer_module._run_executor_delegation(
            task=_task(tools=[tool]),
            subtask="Find the invoice.",
            context="",
            executor_provider=executor,
            task_tools=[tool],
            dispatcher=dispatcher,
            sandbox_tool_list=[],
            trace_id="trace",
            main_writer=main_writer,
            sidecar_path=tmp_path / "exec.jsonl",
            prompt_cfg=None,
            executor_model_cfg=None,
            media_cfg=None,
            max_turns=4,
            timeout_seconds=30,
            wall_start=time.monotonic(),
            min_tool_calls=1,
            max_environment_tool_calls=20,
            report_max_tokens=256,
        )

    assert "Executor attempted the delegated subtask" in report
    assert "executor_finished_without_report" in report
    assert "empty or too-short text" in report
    assert stats.turn_count == 2
    assert stats.report_status == "synthetic_failure"
    assert stats.executor_stopped_reason == "no_tools"


def test_executor_delegation_strict_can_disable_synthetic_failure_report(tmp_path: Path):
    tool = _env_tool()
    executor = FakeProvider(
        "executor",
        [
            _tool_response("kb_search", {"query": "invoice"}),
            _text_response(""),
        ],
    )
    dispatcher = ExecutorDispatcher()

    with TraceWriter(tmp_path / "main.jsonl") as main_writer:
        report, stats = decomposer_module._run_executor_delegation(
            task=_task(tools=[tool]),
            subtask="Find the invoice.",
            context="",
            executor_provider=executor,
            task_tools=[tool],
            dispatcher=dispatcher,
            sandbox_tool_list=[],
            trace_id="trace",
            main_writer=main_writer,
            sidecar_path=tmp_path / "exec.jsonl",
            prompt_cfg=None,
            executor_model_cfg=None,
            media_cfg=None,
            max_turns=4,
            timeout_seconds=30,
            wall_start=time.monotonic(),
            min_tool_calls=1,
            max_environment_tool_calls=20,
            report_max_tokens=256,
            synthetic_failure_report=False,
        )

    end = _trace_end(tmp_path / "exec.jsonl")
    assert report == ""
    assert stats.turn_count == 2
    assert stats.report_status == "missing_report"
    assert stats.executor_stopped_reason == "no_tools"
    assert stats.empty_visible_response_count == 1
    assert end.failure_modes == ["executor_failed_no_report"]


def test_executor_delegation_uses_react_transitional_retry(tmp_path: Path):
    tool = _env_tool()
    executor = FakeProvider(
        "executor",
        [
            _tool_response("kb_search", {"query": "invoice"}, call_id="call_1"),
            _text_response("I found one result. Let me retrieve the details."),
            _tool_response("kb_search", {"query": "invoice details"}, call_id="call_2"),
            _text_response("Action complete."),
            _text_response("- What you did: searched twice.\n- Key findings / outputs: invoice #123.\n- Tool results that matter for next steps: details were retrieved.\n- Blockers / open questions: none."),
        ],
    )
    dispatcher = ExecutorDispatcher()

    with TraceWriter(tmp_path / "main.jsonl") as main_writer:
        report, stats = decomposer_module._run_executor_delegation(
            task=_task(tools=[tool]),
            subtask="Find the invoice.",
            context="",
            executor_provider=executor,
            task_tools=[tool],
            dispatcher=dispatcher,
            sandbox_tool_list=[],
            trace_id="trace",
            main_writer=main_writer,
            sidecar_path=tmp_path / "exec.jsonl",
            prompt_cfg=None,
            executor_model_cfg=None,
            media_cfg=None,
            max_turns=5,
            timeout_seconds=30,
            wall_start=time.monotonic(),
            min_tool_calls=1,
            max_environment_tool_calls=20,
            report_max_tokens=256,
            report_mode="repair",
            react_cfg=ReActRunConfig(
                retry_transitional_tool_text=True,
                transitional_tool_retry_limit=1,
            ),
        )

    assert "invoice #123" in report
    assert stats.turn_count == 5
    assert stats.report_status == "repaired"
    assert len(dispatcher.calls) == 2
    assert executor.calls[2]["kwargs"]["tool_choice"] == "required"
    assert executor.calls[-1]["tools"] == []
