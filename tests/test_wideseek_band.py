import pytest

from evals.wideseek.band import compute
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
    band = compute(tmp_path, 0.9)
    assert band["in_band"] == ["a", "d"]
    assert band["passes"] == [["a", 1, 2], ["b", 2, 2], ["c", 0, 2], ["d", 1, 2]]
    assert (band["tasks"], band["never_solved"], band["always_solved"]) == (4, 1, 1)


def test_band_refuses_an_unfinished_run(tmp_path):
    _run(tmp_path, "interrupted", {("a", 1): ("finished", "scored", 1.0)})
    with pytest.raises(ValueError, match="not completed"):
        compute(tmp_path, 0.9)
