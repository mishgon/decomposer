import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from sft.wideseek.scheduler import new_state, plan_next_wave, qualifies


def outcome(task, attempt, score, status="finished"):
    return {"task_id": task, "attempt": attempt, "status": status,
            "evaluation": {"status": "scored" if score is not None else "judge_error", "score": score}}


@pytest.mark.parametrize("score,expected", [(0.9, False), (0.91, True), (1., True),
    (None, False), (float("nan"), False), (float("inf"), False), (True, False), (1.1, False)])
def test_qualification_matches_toolathlon_rule(score, expected):
    assert qualifies(outcome("a", 1, score), .9) is expected
    assert not qualifies(outcome("a", 1, score, "timeout"), .9)


def test_coverage_skips_solved_and_resumes_same_wave():
    state = new_state(["a", "b", "c"])
    rows = [outcome("a", 1, .95), outcome("b", 1, .2), outcome("c", 1, .9)]
    assert plan_next_wave(state, rows) == [["b", 2], ["c", 2]]
    resumed = json.loads(json.dumps(state))
    rows.append(outcome("b", 2, .99))
    assert plan_next_wave(resumed, rows) == [["c", 2]]
    assert len(resumed["waves"]) == 1
    rows.append(outcome("c", 2, .5))
    assert plan_next_wave(resumed, rows) == [["c", 3]]


def test_six_failures_cull_even_unscored_tasks():
    state = new_state(["solved", "failed", "unscored"], target_successes=1)
    rows = [outcome("solved", 1, 1.)]
    for attempt in range(1, 7):
        assert plan_next_wave(state, rows) == [["failed", attempt], ["unscored", attempt]]
        rows.extend([outcome("failed", attempt, .99, "context_exceeded"),
                     outcome("unscored", attempt, None)])
    assert plan_next_wave(state, rows) == []
    assert state["phase"] == "complete"
    assert state["covered_tasks"] == 1
    assert state["culled_tasks"] == ["failed", "unscored"]


def test_balance_water_fills_lowest_success_count():
    state = new_state(["a", "b", "c"])
    rows = [outcome(task, n, 1.) for task, count in (("a", 1), ("b", 2), ("c", 3))
            for n in range(1, count + 1)]
    assert plan_next_wave(state, rows) == [["a", 2]]
    rows.append(outcome("a", 2, 1.))
    assert plan_next_wave(state, rows) == [["a", 3], ["b", 3]]
    rows.extend([outcome("a", 3, 1.), outcome("b", 3, 1.)])
    assert plan_next_wave(state, rows) == [["a", 4], ["b", 4], ["c", 4]]
    rows.extend([outcome(t, 4, 1.) for t in ("a", "b", "c")])
    assert plan_next_wave(state, rows) == []


def test_balance_stops_after_24_additional_failures():
    state = new_state(["a"])
    rows = [outcome("a", 1, 1.)]
    for attempt in range(2, 26):
        assert plan_next_wave(state, rows) == [["a", attempt]]
        rows.append(outcome("a", attempt, .5))
    assert plan_next_wave(state, rows) == []
    assert state["phase"] == "complete"
    assert state["culled_tasks"] == []  # A once-solved task is retained, never culled.


def test_coverage_only_stops_without_balance_launches():
    state = new_state(["a"], target_successes=1)
    assert plan_next_wave(state, [outcome("a", 1, 1.)]) == []
    assert state["waves"] == []


def test_wave_limit_does_not_append_unstarted_work():
    state = new_state(["a"])
    assert plan_next_wave(state, [outcome("a", 1, .1)], allow_new_wave=False) == []
    assert state["phase"] == "coverage_first"
    assert state["waves"] == []


def test_full_width_split_can_be_planned_and_culled_without_launching():
    tasks = [f"width-{i:05d}" for i in range(20000)]
    state = new_state(tasks)
    assert len(plan_next_wave(state, [])) == 20000
    rows = [outcome(task, attempt, None) for task in tasks for attempt in range(1, 7)]
    assert plan_next_wave(state, rows) == []
    assert state["phase"] == "complete"
    assert state["culled_tasks"] == tasks


def test_collector_resume_counts_saved_traces_and_keeps_failures(tmp_path):
    from sft.wideseek.run import main, completed_results
    tasks = [{"task_id": "a"}, {"task_id": "b"}]
    args = SimpleNamespace(success_threshold=.9, target_successes=1, adaptive=True,
                           n=1, max_waves=1, mode="decomposer")

    def save_result(task, attempt, score):
        path = tmp_path / "decomposer" / task / f"attempt-{attempt:03d}"
        execution = path / "execution-test"
        execution.mkdir(parents=True)
        (execution / "trace.json").write_text('{}')
        (path / "result.json").write_text(json.dumps({**outcome(task, attempt, score),
            "execution_directory": execution.name}))

    save_result("a", 1, .95)
    save_result("b", 1, .2)
    launches = []
    async def run_jobs(tasks, jobs, root, args):
        launches.extend(jobs)
        for task, attempt in jobs:
            save_result(task, attempt, .3 if attempt == 2 else .99)

    with patch("sft.wideseek.run.prepare_run", new=AsyncMock(return_value=(tasks, tmp_path))), \
            patch("sft.wideseek.run.run_jobs", new=run_jobs):
        asyncio.run(main(args))
        assert launches == [["b", 2]]
        assert json.loads((tmp_path / "scheduler.json").read_text())["status"] == "wave_limit"
        asyncio.run(main(args))
    assert launches == [["b", 2], ["b", 3]]
    state = json.loads((tmp_path / "scheduler.json").read_text())
    assert state["status"] == "completed"
    assert state["covered_tasks"] == 2
    assert len(completed_results(tmp_path, "decomposer")) == 4
    assert len((tmp_path / "successful-traces.jsonl").read_text().splitlines()) == 2
    args.target_successes = 4
    with patch("sft.wideseek.run.prepare_run", new=AsyncMock(return_value=(tasks, tmp_path))):
        with pytest.raises(ValueError, match="target_successes"):
            asyncio.run(main(args))
