"""Tests for flat runner wiring."""

from __future__ import annotations

from claw_eval.config import ReActRunConfig
from claw_eval.models.content import TextBlock
from claw_eval.models.message import Message
from claw_eval.models.task import Prompt, TaskDefinition
from claw_eval.models.trace import TokenUsage
from claw_eval.runner import loop as loop_module
from claw_eval.runner.react_loop import ReActLoopResult


class FakeProvider:
    model_id = "fake-model"


def test_flat_run_task_forwards_react_budget_caps(monkeypatch, tmp_path):
    seen = {}

    def fake_run_react_loop(**kwargs):
        seen.update(kwargs)
        return ReActLoopResult(
            messages=[Message(role="assistant", content=[TextBlock(text="done")])],
            turn_count=1,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            stopped_reason="no_tools",
        )

    monkeypatch.setattr(loop_module, "run_react_loop", fake_run_react_loop)

    task = TaskDefinition(
        task_id="T_test",
        task_name="Test",
        prompt=Prompt(text="Do the task."),
    )
    react_cfg = ReActRunConfig(
        max_turns=352,
        max_environment_tool_calls=320,
    )

    loop_module.run_task(
        task,
        FakeProvider(),
        trace_dir=tmp_path,
        react_cfg=react_cfg,
    )

    assert seen["max_turns"] == 352
    assert seen["max_environment_tool_calls"] == 320
    assert seen["react_cfg"] is react_cfg
