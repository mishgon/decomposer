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


def test_coverage_first_prioritizes_tasks_without_a_success(tmp_path: Path) -> None:
    tasks = [f"task-{index:02d}" for index in range(10)]
    episodes = []
    for index, task in enumerate(tasks):
        result = evaluation(
            tmp_path,
            f"first-{task}",
            {
                "pass": index < 5,
                "native_result": {"passed": index, "failed": 10 - index},
            },
        )
        episodes.append(episode(task, 1, attempt(result)))
    manifest = {"episodes": episodes}
    manifest["adaptive_scheduler"] = scheduler.new_scheduler_state(
        tasks, threshold=0.9, cull_fraction=0.1, target_successes=4
    )

    first_wave = scheduler.plan_next_wave(manifest)
    assert first_wave == [f"task-{index:02d}::rep-002" for index in range(5, 10)]


def test_coverage_first_water_fills_success_counts(tmp_path: Path) -> None:
    one = evaluation(tmp_path, "one", {"pass": True, "native_result": None})
    two = evaluation(tmp_path, "two", {"pass": True, "native_result": None})
    manifest = {
        "episodes": [
            episode("alpha", 1, attempt(one)),
            episode("alpha", 2, attempt(two)),
            episode("beta", 1, attempt(one)),
        ]
    }
    manifest["adaptive_scheduler"] = scheduler.new_scheduler_state(
        ["alpha", "beta"], threshold=0.9, cull_fraction=0.1, target_successes=4
    )

    first_wave = scheduler.plan_next_wave(manifest)
    assert first_wave == ["beta::rep-002"]

    pending = manifest["episodes"][-1]
    pending.update(status="completed", attempts=[attempt(two)])
    second_wave = scheduler.plan_next_wave(manifest)
    assert second_wave == ["alpha::rep-003", "beta::rep-003"]


def test_coverage_first_drops_known_zero_after_six_launches(tmp_path: Path) -> None:
    zero = evaluation(
        tmp_path,
        "zero",
        {"pass": False, "native_result": {"passed": 0, "failed": 10}},
    )
    manifest = {
        "episodes": [episode("alpha", i, attempt(zero)) for i in range(1, 7)]
    }
    manifest["adaptive_scheduler"] = scheduler.new_scheduler_state(
        ["alpha"], threshold=0.9, cull_fraction=0.1, target_successes=4
    )

    assert scheduler.plan_next_wave(manifest) == []
    assert manifest["adaptive_scheduler"]["active_tasks"] == []
    assert manifest["adaptive_scheduler"]["phase"] == "complete"


def test_coverage_first_honors_per_task_limits_and_can_cap_unparseable(
    tmp_path: Path,
) -> None:
    known = evaluation(
        tmp_path,
        "known",
        {"pass": False, "native_result": {"passed": 0, "failed": 10}},
    )
    unknown = evaluation(
        tmp_path,
        "unknown",
        {"pass": False, "native_result": {"errors": ["unparsed"]}},
    )
    manifest = {
        "episodes": [
            episode("known", 1, attempt(known)),
            episode("known", 2, attempt(known)),
            episode("unknown", 1, attempt(unknown)),
            episode("unknown", 2, attempt(unknown)),
            episode("unknown", 3, attempt(unknown)),
        ]
    }
    state = scheduler.new_scheduler_state(
        ["known", "unknown"], threshold=0.9, cull_fraction=0.1, target_successes=4
    )
    state["zero_success_launch_limits"] = {"known": 2, "unknown": 3}
    state["protect_unscored_evaluations"] = False
    manifest["adaptive_scheduler"] = state

    assert scheduler.plan_next_wave(manifest) == []
    assert state["active_tasks"] == []
    assert {item["task"]: item["after_launches"] for item in state["culled_tasks"]} == {
        "known": 2,
        "unknown": 3,
    }


def test_migration_removes_only_unstarted_legacy_queue(tmp_path: Path) -> None:
    done = evaluation(tmp_path, "done", {"pass": True, "native_result": None})
    manifest = {
        "episodes": [
            episode("alpha", 1, attempt(done)),
            episode("alpha", 2),
            episode("beta", 1, attempt(None, error={"interrupted": True})),
            episode("beta", 2),
        ],
        "adaptive_scheduler": {
            "schema_version": 1,
            "phase": "drop_unsolved_after_six",
            "success_threshold": 0.9,
            "cull_fraction": 0.1,
            "target_successes": 4,
            "original_tasks": ["alpha", "beta", "culled"],
            "active_tasks": ["alpha", "beta"],
            "culled_tasks": [{"task": "culled"}],
            "rounds": [
                {
                    "phase": "six_total",
                    "episode_keys": ["alpha::rep-002", "beta::rep-001"],
                }
            ],
        },
    }

    removed = scheduler.migrate_to_coverage_first(manifest)

    assert removed == 2
    assert [item["key"] for item in manifest["episodes"]] == [
        "alpha::rep-001",
        "beta::rep-001",
    ]
    state = manifest["adaptive_scheduler"]
    assert state["policy"] == "coverage_first"
    assert state["phase"] == "coverage_first"
    assert state["active_tasks"] == ["alpha", "beta"]
    assert state["culled_tasks"] == [{"task": "culled"}]
    assert state["rounds"][0]["episode_keys"] == ["beta::rep-001"]


def test_adaptive_batch_water_fills_without_changing_models(
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

    assert len(calls) == 40
    assert {(teacher, student) for teacher, student, _task in calls} == {
        ("teacher-model", "student-model")
    }
    assert manifest["adaptive_scheduler"]["phase"] == "complete"
    assert manifest["adaptive_scheduler"]["culled_tasks"] == []
