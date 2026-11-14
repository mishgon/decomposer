import json
import sys

import pytest

from evals.wideseek.band import compute, main
from gyms.wideseek.runtime import save


def _run(root, status, rows):
    save(root / "manifest.json", {"status": status, "repetitions": 2, "settings": {"modes": ["decomposer"]}})
    for (task, attempt), (outcome, evaluation, score) in rows.items():
        save(root / f"decomposer/{task}/attempt-{attempt:03d}/result.json",
             {"task_id": task, "status": outcome, "evaluation": {"status": evaluation, "score": score}})


def test_band_keeps_tasks_solved_on_some_attempts(tmp_path):
    _run(tmp_path, "completed", {
        ("a", 1): ("finished", "scored", 1.0), ("a", 2): ("finished", "scored", 0.0),
        ("b", 1): ("finished", "scored", 1.0), ("b", 2): ("finished", "scored", 1.0),
        ("c", 1): ("timeout", "timeout", None), ("c", 2): ("error", "unscored", None),
        ("d", 1): ("finished", "scored", 0.95), ("d", 2): ("finished", "scored", 0.9),
    })
    band = compute([tmp_path], 0.9)
    assert band["in_band"] == ["a", "d"]
    assert band["passes"] == [["a", 1, 2], ["b", 2, 2], ["c", 0, 2], ["d", 1, 2]]
    assert (band["tasks"], band["never_solved"], band["always_solved"]) == (4, 1, 1)


def test_band_refuses_an_unfinished_run(tmp_path):
    _run(tmp_path, "interrupted", {("a", 1): ("finished", "scored", 1.0)})
    with pytest.raises(ValueError, match="not completed"):
        compute([tmp_path], 0.9)


def test_second_stage_attempts_only_the_undecided_tasks_and_counts_with_the_first(tmp_path, monkeypatch):
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps({"task_id": task, "question": task}) + "\n" for task in "dcbae"))
    first, second, undecided = tmp_path / "first", tmp_path / "second", tmp_path / "undecided.jsonl"
    _run(first, "completed", {
        ("a", 1): ("finished", "scored", 1.0), ("a", 2): ("finished", "scored", 0.0),
        ("b", 1): ("finished", "scored", 1.0), ("b", 2): ("finished", "scored", 1.0),
        ("c", 1): ("finished", "scored", 0.0), ("c", 2): ("timeout", "timeout", None),
        ("d", 1): ("finished", "scored", 1.0), ("d", 2): ("finished", "scored", 1.0),
    })
    monkeypatch.setattr(sys, "argv", ["band", str(first), "--undecided", str(undecided), "--data", str(pool)])
    main()
    assert [json.loads(row)["task_id"] for row in undecided.read_text().splitlines()] == ["d", "c", "b"]
    _run(second, "completed", {
        ("b", 1): ("finished", "scored", 1.0), ("b", 2): ("finished", "scored", 0.0),
        ("c", 1): ("finished", "scored", 0.0), ("c", 2): ("finished", "scored", 0.0),
        ("d", 1): ("finished", "scored", 1.0), ("d", 2): ("finished", "scored", 1.0),
    })
    monkeypatch.setattr(sys, "argv", ["band", str(first), str(second)])
    main()
    band = json.loads((second / "band.json").read_text())
    assert band["in_band"] == ["a", "b"]
    assert band["passes"] == [["a", 1, 2], ["b", 3, 4], ["c", 0, 4], ["d", 4, 4]]
    assert band["run_dirs"] == [str(first), str(second)] and band["repetitions"] == [2, 2]
