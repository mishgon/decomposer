"""Tests for persistent batch trace grading."""

from __future__ import annotations

import json
from pathlib import Path

from claw_eval.cli import main


def _write_temp_task(tasks_dir: Path) -> Path:
    task_dir = tasks_dir / "T001_test_batch_grade"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: T001_test_batch_grade
task_name: Batch Grade Test
prompt:
  text: Produce a final answer.
  language: en
primary_dimensions: [completion, robustness, communication]
""".strip()
        + "\n"
    )
    (task_dir / "grader.py").write_text(
        """
from claw_eval.graders.base import AbstractGrader
from claw_eval.models.trace import DimensionScores


class BatchGradeTestGrader(AbstractGrader):
    def grade(self, messages, dispatches, task, audit_data=None, judge=None, media_events=None, env_snapshot=None):
        return DimensionScores(completion=1.0, robustness=0.5, communication=0.25, safety=1.0)
""".lstrip()
    )
    return task_dir


def _write_dispatch_gate_task(tasks_dir: Path) -> Path:
    task_dir = tasks_dir / "T001_test_batch_grade"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
task_id: T001_test_batch_grade
task_name: Batch Grade Test
prompt:
  text: Produce a final answer.
  language: en
primary_dimensions: [completion, robustness, communication]
""".strip()
        + "\n"
    )
    (task_dir / "grader.py").write_text(
        """
from claw_eval.graders.base import AbstractGrader
from claw_eval.models.trace import DimensionScores


class BatchGradeTestGrader(AbstractGrader):
    def grade(self, messages, dispatches, task, audit_data=None, judge=None, media_events=None, env_snapshot=None):
        saw_tool = any(d.tool_name == "sidecar_tool" for d in dispatches)
        return DimensionScores(completion=1.0 if saw_tool else 0.0, robustness=1.0, communication=0.0, safety=1.0)
""".lstrip()
    )
    return task_dir


def _write_trace(trace_dir: Path) -> Path:
    trace_path = trace_dir / "T001_test_batch_grade_abcd1234.jsonl"
    events = [
        {
            "type": "trace_start",
            "trace_id": "trace-1",
            "task_id": "T001_test_batch_grade",
            "model": "fake-model",
        },
        {
            "type": "message",
            "trace_id": "trace-1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "final answer"}],
            },
        },
        {
            "type": "trace_end",
            "trace_id": "trace-1",
            "total_turns": 1,
            "model_input_tokens": 10,
            "model_output_tokens": 4,
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "model_time_s": 1.5,
            "tool_time_s": 0.25,
            "other_time_s": 0.1,
            "wall_time_s": 1.85,
        },
    ]
    with open(trace_path, "w") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")
    return trace_path


def _write_decomposer_trace_with_sidecar(trace_dir: Path) -> Path:
    trace_path = trace_dir / "T001_test_batch_grade_decomp1234.jsonl"
    sidecar_path = trace_dir / "T001_test_batch_grade_decomp1234_exec_1.jsonl"
    main_events = [
        {
            "type": "trace_start",
            "trace_id": "trace-1",
            "task_id": "T001_test_batch_grade",
            "model": "fake-manager+executor",
            "run_mode": "decomposer",
        },
        {
            "type": "message",
            "trace_id": "trace-1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "final answer"}],
            },
        },
        {
            "type": "delegation_end",
            "trace_id": "trace-1",
            "delegation_index": 1,
            "report": "tool used",
            "executor_turns": 1,
            "sidecar_trace": sidecar_path.name,
        },
        {
            "type": "trace_end",
            "trace_id": "trace-1",
            "total_turns": 1,
            "model_input_tokens": 10,
            "model_output_tokens": 4,
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
        },
    ]
    sidecar_events = [
        {
            "type": "trace_start",
            "trace_id": "trace-1-exec",
            "task_id": "T001_test_batch_grade",
            "model": "fake-executor",
        },
        {
            "type": "tool_dispatch",
            "trace_id": "trace-1-exec",
            "tool_use_id": "tool-1",
            "tool_name": "sidecar_tool",
            "endpoint_url": "http://localhost/tool",
            "request_body": {"x": 1},
        },
        {"type": "trace_end", "trace_id": "trace-1-exec", "total_turns": 1},
    ]
    for path, events in [(trace_path, main_events), (sidecar_path, sidecar_events)]:
        with open(path, "w") as fh:
            for event in events:
                fh.write(json.dumps(event) + "\n")
    return trace_path


def _write_batch_results(trace_dir: Path, trace_path: Path) -> None:
    results = [
        {
            "task_id": "T001_test_batch_grade",
            "task_name": "Batch Grade Test",
            "difficulty": "simple",
            "trials": [
                {
                    "trace": str(trace_path),
                    "model_input_tokens": 10,
                    "model_output_tokens": 4,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "tokens": 14,
                    "model_time_s": 1.5,
                    "tool_time_s": 0.25,
                    "other_time_s": 0.1,
                    "wall_time_s": 1.85,
                    "task_score": None,
                    "passed": None,
                    "skip_grade": True,
                }
            ],
            "error": None,
            "grading_skipped": True,
        }
    ]
    (trace_dir / "batch_results.json").write_text(json.dumps(results, indent=2))


def _count_grading_results(trace_path: Path) -> int:
    return sum(1 for line in trace_path.read_text().splitlines() if json.loads(line).get("type") == "grading_result")


def test_grade_batch_persists_results_and_summary(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    _write_temp_task(tasks_dir)
    trace_path = _write_trace(trace_dir)
    _write_batch_results(trace_dir, trace_path)

    main([
        "grade-batch",
        "--trace-dir",
        str(trace_dir),
        "--tasks-dir",
        str(tasks_dir),
        "--no-judge",
        "--no-vllm-wait",
    ])

    assert _count_grading_results(trace_path) == 1
    results = json.loads((trace_dir / "batch_results.json").read_text())
    trial = results[0]["trials"][0]
    assert trial["completion"] == 1.0
    assert trial["robustness"] == 0.5
    assert trial["communication"] == 0.25
    assert trial["task_score"] == 0.9
    assert trial["passed"] is True
    assert "skip_grade" not in trial
    assert results[0]["avg_score"] == 0.9
    assert results[0]["avg_passed"] is True
    assert "grading_skipped" not in results[0]

    summary = json.loads((trace_dir / "batch_summary.json").read_text())
    assert summary["tasks"] == 1
    assert summary["graded_tasks"] == 1
    assert summary["trace_only_tasks"] == 0
    assert summary["avg_score"] == 0.9
    assert summary["pass_hat_1"] == 1
    assert summary["pass_at_1"] == 1


def test_grade_batch_skips_existing_grades_unless_forced(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    _write_temp_task(tasks_dir)
    trace_path = _write_trace(trace_dir)
    _write_batch_results(trace_dir, trace_path)

    args = [
        "grade-batch",
        "--trace-dir",
        str(trace_dir),
        "--tasks-dir",
        str(tasks_dir),
        "--no-judge",
        "--no-vllm-wait",
    ]
    main(args)
    main(args)
    assert _count_grading_results(trace_path) == 1

    main(args + ["--force"])
    assert _count_grading_results(trace_path) == 2


def test_grade_batch_counts_decomposer_sidecar_dispatches(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    _write_dispatch_gate_task(tasks_dir)
    trace_path = _write_decomposer_trace_with_sidecar(trace_dir)
    _write_batch_results(trace_dir, trace_path)

    main([
        "grade-batch",
        "--trace-dir",
        str(trace_dir),
        "--tasks-dir",
        str(tasks_dir),
        "--no-judge",
        "--no-vllm-wait",
    ])

    results = json.loads((trace_dir / "batch_results.json").read_text())
    trial = results[0]["trials"][0]
    assert trial["completion"] == 1.0
    assert trial["task_score"] == 1.0
    assert trial["passed"] is True
