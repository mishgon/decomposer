"""Tests for explicit batch task selection."""

from __future__ import annotations

import argparse
import builtins
import json
from concurrent.futures import Future
from types import SimpleNamespace

import claw_eval.cli as cli
from claw_eval.cli import _resolve_batch_task_dirs


def _make_task(root, name: str) -> None:
    task_dir = root / name
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text("task_id: test\n", encoding="utf-8")


def test_resolve_batch_task_dirs_accepts_ids_and_paths(tmp_path):
    _make_task(tmp_path, "T112_expense_email_check")
    _make_task(tmp_path, "T116_ticket_kb_suggestion")

    args = argparse.Namespace(tasks="T112", task=[str(tmp_path / "T116_ticket_kb_suggestion")])
    resolved = _resolve_batch_task_dirs(args, tmp_path)

    assert resolved == [
        str(tmp_path / "T112_expense_email_check"),
        str(tmp_path / "T116_ticket_kb_suggestion"),
    ]


def test_resolve_batch_task_dirs_deduplicates(tmp_path):
    _make_task(tmp_path, "T112_expense_email_check")

    args = argparse.Namespace(tasks="T112,T112_expense_email_check", task=["T112"])
    resolved = _resolve_batch_task_dirs(args, tmp_path)

    assert resolved == [str(tmp_path / "T112_expense_email_check")]



def test_main_list_does_not_import_decomposer(monkeypatch, tmp_path, capsys):
    _make_task(tmp_path, "T001_demo")
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "cli_decomposer" in name:
            raise AssertionError("stable commands must not import cli_decomposer")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    cli.main(["list", "--tasks-dir", str(tmp_path)])

    assert "T001_demo" in capsys.readouterr().out


def test_batch_summary_counts_task_errors_as_failed_graded_tasks(monkeypatch, tmp_path):
    _make_task(tmp_path, "T001_ok")
    _make_task(tmp_path, "T002_error")
    trace_root = tmp_path / "traces"
    batch_trace_dir = trace_root / "batch"
    cfg = SimpleNamespace(
        defaults=SimpleNamespace(trace_dir=str(trace_root)),
        model=SimpleNamespace(model_id="test-model"),
    )

    import claw_eval.config as config_module

    monkeypatch.setattr(config_module, "load_config", lambda _path=None: cfg)
    monkeypatch.setattr(cli, "_make_trace_dir", lambda _base, _model: batch_trace_dir)

    fake_results = iter([
        {
            "task_id": "T001_ok",
            "trials": [
                {
                    "task_score": 1.0,
                    "passed": True,
                    "completion": 1.0,
                    "robustness": 1.0,
                    "communication": 1.0,
                    "safety": 1.0,
                    "tokens": 0,
                }
            ],
        },
        {"task_id": "T002_error", "error": "boom", "trials": []},
    ])

    class FakePool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, *args, **kwargs):
            future = Future()
            future.set_result(next(fake_results))
            return future

    monkeypatch.setattr(cli, "ProcessPoolExecutor", FakePool)
    args = argparse.Namespace(
        tasks_dir=str(tmp_path),
        tasks=None,
        task=[],
        filter=None,
        tag=None,
        range=None,
        rerun_errors=None,
        continue_dir=None,
        parallel=1,
        trials=1,
        config=None,
        model=None,
        api_key=None,
        base_url=None,
        trace_dir=str(trace_root),
        no_judge=True,
        judge_model=None,
        proxy=None,
        port_base_offset=0,
        sandbox=False,
        no_sandbox=True,
        sandbox_image=None,
        sandbox_tools=False,
        no_vllm_wait=True,
        skip_grade=False,
        launch_vllm=False,
    )

    cli.cmd_batch(args)

    summary = json.loads((batch_trace_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["graded_tasks"] == 2
    assert summary["errored"] == 1
    assert summary["avg_score"] == 0.5
    assert summary["pass_hat_1"] == 1
    assert summary["pass_at_1"] == 1
