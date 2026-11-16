import json
import sys

import pytest

from evals.wideseek.band import compute, main, resolved
from gyms.wideseek.runtime import save


def _run(root, status, rows, repetitions=2):
    save(root / "manifest.json", {"status": status, "repetitions": repetitions,
         "settings": {"modes": ["decomposer"], "tasks": sorted({task for task, _ in rows})}})
    for (task, attempt), (outcome, evaluation, score) in rows.items():
        save(root / f"decomposer/{task}/attempt-{attempt:03d}/result.json",
             {"task_id": task, "status": outcome, "evaluation": {"status": evaluation, "score": score}})


def test_band_keeps_tasks_solved_on_some_attempts(tmp_path):
    _run(tmp_path, "completed", {
        ("a", 1): ("finished", "scored", 1.0), ("a", 2): ("finished", "scored", 0.0),
        ("b", 1): ("finished", "scored", 1.0), ("b", 2): ("finished", "scored", 1.0),
        ("c", 1): ("timeout", "timeout", None), ("c", 2): ("context_exceeded", "unscored", None),
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


def test_judge_error_requires_replacement_and_does_not_create_a_band(tmp_path, monkeypatch):
    first, retry = tmp_path / 'first', tmp_path / 'retry'
    _run(first, 'completed', {
        ('a', 1): ('finished', 'scored', 1),
        ('a', 2): ('finished', 'evaluation_error', None),
        ('b', 1): ('error', 'unscored', None),
        ('b', 2): ('infrastructure_error', 'infrastructure_error', None),
    })
    pool, rerun = tmp_path / 'pool.jsonl', tmp_path / 'rerun.jsonl'
    pool.write_text(''.join(json.dumps({'task_id': task, 'question': task}) + '\n' for task in 'bac'))
    monkeypatch.setattr(sys, 'argv', ['band', str(first), '--rerun', str(rerun), '--data', str(pool)])
    main()
    band = json.loads((first / 'band.json').read_text())
    assert band['passes'] == [['a', 1, 1], ['b', 0, 0]]
    assert band['in_band'] == []
    assert band['never_solved'] == band['always_solved'] == 0
    assert band['missing_attempts'] == {'a': 1, 'b': 2}
    assert len(band['unresolved']) == 3
    assert [json.loads(row)['task_id'] for row in rerun.read_text().splitlines()] == ['b', 'a']
    _run(retry, 'completed', {('a', 1): ('finished', 'scored', 0)}, repetitions=1)
    band = compute([first, retry], .9)
    assert band['passes'] == [['a', 1, 2], ['b', 0, 0]]
    assert band['in_band'] == ['a']
    assert band['rerun'] == ['b']
    assert band['missing_attempts'] == {'b': 2}


def test_mixed_task_with_too_few_valid_attempts_is_rerun_and_undecided(tmp_path, monkeypatch):
    first, retry = tmp_path / 'first', tmp_path / 'retry'
    _run(first, 'completed', {
        ('a', 1): ('finished', 'scored', 1),
        ('a', 2): ('finished', 'scored', 0),
        ('a', 3): ('error', 'unscored', None),
    }, repetitions=3)
    pool, undecided = tmp_path / 'pool.jsonl', tmp_path / 'undecided.jsonl'
    pool.write_text('{"task_id":"a","question":"A"}\n')
    monkeypatch.setattr(sys, 'argv', ['band', str(first), '--undecided', str(undecided), '--data', str(pool)])
    main()
    band = json.loads((first / 'band.json').read_text())
    assert band['in_band'] == []
    assert band['missing_attempts'] == {'a': 1}
    assert undecided.read_text() == pool.read_text()
    _run(retry, 'completed', {('a', 1): ('finished', 'scored', 1)}, repetitions=1)
    band = compute([first, retry], .9)
    assert band['in_band'] == ['a']
    assert band['passes'] == [['a', 2, 3]]
    assert band['rerun'] == []


def test_missing_result_in_completed_manifest_is_not_a_failure(tmp_path):
    _run(tmp_path, 'completed', {('a', 1): ('finished', 'scored', 1)})
    band = compute([tmp_path], .9)
    assert band['passes'] == [['a', 1, 1]]
    assert band['missing_attempts'] == {'a': 1}
    assert band['unresolved'][0]['status'] == 'missing'
    assert band['always_solved'] == 0


@pytest.mark.parametrize('error', [
    'GraphRecursionError: Recursion limit reached',
    'OpenAIContextOverflowError: context too long',
    "BadRequestError: Error code: 400 - This model's maximum context length is 32768 tokens",
    'ValueError: Decomposer requested tools instead of completing graceful shutdown',
    'ValueError: Agent requested tools instead of returning its report',
    'RuntimeError: vLLM exhausted max_completion_tokens before completing its response.',
])
def test_expected_model_limits_are_measured_failures(error):
    assert resolved({'status': 'error', 'error': error, 'evaluation': {'status': 'unscored'}}, .9) is False


@pytest.mark.parametrize('patch', [
    {'cleanup_errors': ['worker still active']}, {'capture_error': 'failed'},
    {'trace_error': 'failed'}, {'export_error': 'failed'}, {'trace_available': False},
    {'evaluation': {'status': 'evaluation_error', 'error': 'judge unavailable', 'score': None}},
    {'evaluation': {'status': 'scored', 'score': float('nan')}},
    {'status': 'error', 'error': 'APIConnectionError: service unavailable'},
    {'status': 'error', 'error': 'BadRequestError: unsupported parameter'},
    {'status': 'error', 'error': 'RuntimeError: unknown failure'},
])
def test_unmeasured_outcomes_are_not_model_failures(patch):
    result = {'status': 'finished', 'evaluation': {'status': 'scored', 'score': 1}, **patch}
    assert resolved(result, .9) is None
