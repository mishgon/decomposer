import json
import subprocess
from pathlib import Path

from gyms.toolathlon_gym import adaptive_scheduler as scheduler, batch


def evaluation(tmp_path: Path, name: str, value: dict) -> str:
    path = tmp_path / name / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def episode(task: str, repetition: int, *attempts: dict) -> dict:
    status = attempts[-1]["status"] if attempts else "pending"
    return {
        "key": f"{task}::rep-{repetition:03d}",
        "task": task,
        "repetition": repetition,
        "status": status,
        "attempts": list(attempts),
    }


def attempt(evaluation_path: str | None, **extra) -> dict:
    return {
        "status": "completed" if evaluation_path else "failed",
        "evaluation_path": evaluation_path,
        **extra,
    }


def test_extract_partial_score_supports_native_and_common_stdout() -> None:
    native = scheduler.extract_partial_score(
        {"native_result": {"total_passed": 9, "total_checks": 10}}
    )
    fraction = scheduler.extract_partial_score(
        {"native_result": None, "stdout": "=== Results: 17/20 passed ==="}
    )
    pass_fail = scheduler.extract_partial_score(
        {"native_result": None, "stdout": "Passed: 18\nFailed: 2"}
    )

    assert native and native.fraction == 0.9
    assert fraction and fraction.fraction == 0.85
    assert pass_fail and pass_fail.fraction == 0.9


def test_extract_partial_score_has_narrow_marker_fallback() -> None:
    markers = scheduler.extract_partial_score(
        {
            "native_result": None,
            "stdout": "[PASS] file exists\n[OK] sheet exists\n[FAIL] wrong row",
        }
    )
    status_lines = scheduler.extract_partial_score(
        {"native_result": None, "stdout": "Checking one\nPASS\nChecking two\nFAIL"}
    )

    assert markers and markers.fraction == 2 / 3
    assert status_lines and status_lines.fraction == 1 / 2


def test_qualification_is_strictly_greater_than_threshold(tmp_path: Path) -> None:
    exactly = evaluation(
        tmp_path,
        "exactly",
        {"pass": False, "native_result": {"passed": 9, "failed": 1}},
    )
    above = evaluation(
        tmp_path,
        "above",
        {"pass": False, "native_result": {"passed": 91, "failed": 9}},
    )
    strict = evaluation(tmp_path, "strict", {"pass": True, "native_result": None})

    assert not scheduler.load_launch_outcome("task", exactly).qualifies(0.9)
    assert scheduler.load_launch_outcome("task", above).qualifies(0.9)
    assert scheduler.load_launch_outcome("task", strict).qualifies(0.9)


def test_interrupted_attempts_do_not_consume_launch_budget() -> None:
    manifest = {
        "episodes": [
            episode(
                "alpha",
                1,
                attempt(None, error={"interrupted": True}),
                attempt(None, error={"type": "EpisodeTimeout"}),
            )
        ]
    }

    assert len(scheduler.task_launches(manifest, "alpha")) == 1
    added = scheduler.append_episodes_to_target(manifest, ["alpha"], 3)
    assert added == ["alpha::rep-002", "alpha::rep-003"]


def test_bottom_cull_only_selects_zero_success_tasks(tmp_path: Path) -> None:
    tasks = [f"task-{index:02d}" for index in range(10)]
    episodes = []
    for index, task in enumerate(tasks):
        result = evaluation(
            tmp_path,
            task,
            {
                "pass": index == 0,
                "native_result": {"passed": index, "failed": 10 - index},
            },
        )
        episodes.append(episode(task, 1, attempt(result)))
    manifest = {"episodes": episodes}

    culled = scheduler.choose_bottom_tasks(
        manifest, tasks, threshold=0.9, fraction=0.1
    )

    assert culled == ["task-01"]
    assert "task-00" not in culled  # Strict pass is always protected.


def test_bottom_cull_protects_unparsed_native_evaluations(tmp_path: Path) -> None:
    unknown = evaluation(
        tmp_path,
        "unknown",
        {"pass": False, "native_result": {"errors": ["one"]}},
    )
    zero = evaluation(
        tmp_path,
        "zero",
        {"pass": False, "native_result": {"passed": 0, "failed": 10}},
    )
    manifest = {
        "episodes": [
            episode("unknown", 1, attempt(unknown)),
            episode("zero", 1, attempt(zero)),
        ]
    }

    assert scheduler.choose_bottom_tasks(
        manifest, ["unknown", "zero"], threshold=0.9, fraction=0.5
    ) == ["zero"]


def test_one_plus_two_plus_three_then_drop_zero_success(tmp_path: Path) -> None:
    tasks = [f"task-{index:02d}" for index in range(10)]
    first_results = {}
    episodes = []
    for index, task in enumerate(tasks):
        first_results[task] = evaluation(
            tmp_path,
            f"first-{task}",
            {
                "pass": index == 9,
                "native_result": {"passed": index, "failed": 10 - index},
            },
        )
        episodes.append(episode(task, 1, attempt(first_results[task])))
    manifest = {"episodes": episodes}
    manifest["adaptive_scheduler"] = scheduler.new_scheduler_state(
        tasks, threshold=0.9, cull_fraction=0.1, target_successes=4
    )

    first_wave = scheduler.plan_next_wave(manifest)
    assert len(first_wave) == 20
    assert {item["repetition"] for item in manifest["episodes"]} == {1, 2, 3}

    for item in manifest["episodes"]:
        if item["status"] == "pending":
            item.update(
                status="failed",
                attempts=[attempt(None, error={"type": "EpisodeTimeout"})],
            )
    second_wave = scheduler.plan_next_wave(manifest)

    assert "task-00" not in manifest["adaptive_scheduler"]["active_tasks"]
    assert len(second_wave) == 27  # Three launches for each of nine survivors.

    for item in manifest["episodes"]:
        if item["status"] == "pending":
            item.update(
                status="failed",
                attempts=[attempt(None, error={"type": "EpisodeTimeout"})],
            )
    scheduler.plan_next_wave(manifest)

    # task-09 had a strict pass; every remaining zero-success task is retired
    # after six consumed launches.
    assert manifest["adaptive_scheduler"]["active_tasks"] == ["task-09"]
    assert manifest["adaptive_scheduler"]["phase"] == "balance_successes"


def test_adaptive_batch_runs_three_plus_three_without_changing_models(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = [f"task-{index:02d}" for index in range(10)]
    toolathlon_root = tmp_path / "toolathlon"
    for task in tasks:
        (toolathlon_root / "tasks" / "finalpool" / task).mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(batch, "new_run_id", lambda: "adaptive-run")
    calls = []

    def fake_execute_episode(args, **kwargs):
        calls.append((args.model, args.subagent_model, kwargs["episode"]["task"]))
        task = kwargs["episode"]["task"]
        repetition = kwargs["episode"]["repetition"]
        result_path = artifacts / "evals" / task / str(repetition) / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "pass": True,
                    "native_result": {"total_passed": 10, "total_checks": 10},
                }
            ),
            encoding="utf-8",
        )
        return {
            "attempt": kwargs["attempt"],
            "status": "completed",
            "score": True,
            "artifact_path": f"/traces/{task}/{repetition}",
            "evaluation_path": str(result_path),
            "started_at": "start",
            "finished_at": "finish",
            "duration_seconds": 1.0,
            "returncode": 0,
            "error": None,
        }

    monkeypatch.setattr(batch, "execute_episode", fake_execute_episode)
    manifest = batch.main(
        [
            "--all",
            "--adaptive",
            "--purpose",
            "trace-generation",
            "--model",
            "teacher-model",
            "--subagent-model",
            "student-model",
            "--gym-artifacts-dir",
            str(artifacts),
            "--concurrency",
            "4",
        ],
        repo_root=tmp_path,
        toolathlon_root=toolathlon_root,
        default_artifacts_dir=artifacts,
        default_image="image",
        default_model="default-teacher",
        default_subagent_model="default-student",
        default_subagent_port=8023,
        start_vllm=lambda **kwargs: None,
        stop_vllm=lambda process: None,
        docker=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )

    assert len(calls) == 60
    assert {(teacher, student) for teacher, student, _task in calls} == {
        ("teacher-model", "student-model")
    }
    assert manifest["adaptive_scheduler"]["phase"] == "complete"
    assert manifest["adaptive_scheduler"]["culled_tasks"] == []
