"""Checks for report data accounting; plotting dependencies are script-local."""

import importlib.util
import json
from pathlib import Path

import pytest

for dependency in ("matplotlib", "reportlab", "tensorboard", "yaml"):
    pytest.importorskip(dependency)

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

spec = importlib.util.spec_from_file_location(
    "pdf_report", Path(__file__).parents[1] / "scripts/pdf-report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def events(run, name, points):
    writer = EventFileWriter(str(run / "train/tensorboard" / name))
    for step, timestamp, value in points:
        writer.add_event(
            Event(
                step=step,
                wall_time=timestamp,
                summary=Summary(value=[Summary.Value(tag="training/global_step", simple_value=value)]),
            )
        )
    writer.close()


def trajectory(run, name, *, source="wideseek/train", weights=None, result=None):
    path = run / "train/episodes" / name / "trajectory.json"
    save(
        path,
        {
            "task_id": "depth-1",
            "data_source": source,
            "weight_versions": weights or {"global_steps": 8},
            "response_logprobs": [-1.0, -100.0, -3.0],
            "response_mask": [1, 0, 1],
            "result": result,
        },
    )
    return path


def test_resume_steps_masks_mixed_versions_and_latest_retry(tmp_path):
    events(tmp_path, "original", [(2, 100, 1), (3, 110, 2)])
    events(tmp_path, "resume", [(3, 120, 3)])
    trajectory(tmp_path, "train")
    trajectory(tmp_path, "mixed", weights={"global_steps": 8, "min_global_steps": 7, "max_global_steps": 8})
    trajectory(
        tmp_path, "a-new", source="wideseek/heldout", result={"finished_at": 150, "status": "finished"}
    )
    trajectory(tmp_path, "z-old", source="wideseek/heldout", result={"finished_at": 140, "status": "timeout"})
    workplace = tmp_path / "train/workplace-episodes/one"
    save(workplace / "result.json", {"task_id": 1, "error": "TimeoutError: "})
    (workplace / "policy_calls.jsonl").write_text(json.dumps({"extra_fields": {"global_steps": 8}}) + "\n")
    data = report.load_run(tmp_path)
    assert [r["value"] for r in data["series"]["training/global_step"]] == [1, 3]
    assert data["proxy"]["rows"] == [
        {"version": 8, "episodes": 1, "tokens": 2, "nll_sum": 4.0, "mean_nll": 2.0}
    ]
    assert data["proxy"]["excluded_mixed_version_episodes"] == 1
    assert data["depth"]
    assert not report.timeout(data["panels"][("wideseek/heldout", 8)]["depth-1"])
    assert report.timeout(data["panels"][("workplace-assistant/validation20", 8)]["1"])


def test_bad_token_alignment_is_explicit(tmp_path):
    events(tmp_path, "one", [(2, 100, 1)])
    path = trajectory(tmp_path, "one")
    data = json.loads(path.read_text())
    data["response_mask"] = [1]
    save(path, data)
    with pytest.raises(ValueError, match="alignment mismatch"):
        report.load_run(tmp_path)


def test_planned_updates_uses_run_configuration():
    data = {
        "run_metadata": {},
        "manifest": {"train_tasks": 20, "attempts_per_task_per_epoch": 1},
        "config": {
            "rollout": {"total_rollout_steps": 512},
            "trainer": {"total_epochs": 4},
            "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 8}},
            "async_training": {"require_batches": 1, "trigger_parameter_sync_step": 1},
        },
    }
    assert report.planned_updates(data) == 10
    data["run_metadata"]["optimizer_updates"] = 64
    assert report.planned_updates(data) == 64


def test_completed_training_overrides_parent_still_running(tmp_path):
    events(tmp_path, "one", [(2, 100, 1)])
    save(tmp_path / "local-run.json", {"status": "running", "started_at": 1})
    save(tmp_path / "train/run.json", {"status": "completed", "started_at": 10, "finished_at": 200})
    metadata = report.load_run(tmp_path)["run_metadata"]
    assert metadata == {"status": "completed", "started_at": 1, "finished_at": 200}


def test_repeated_validation_uses_actual_logged_mean():
    tags = report.validation_tags({"val-core/synth/heldout/reward/mean@4": []}, "synth")
    assert tags[1] == "val-core/synth/heldout/reward/mean@4"
