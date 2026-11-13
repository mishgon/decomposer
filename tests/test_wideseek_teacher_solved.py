import json

import pytest

from evals.wideseek.teacher_solved import prepare, summarize


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def result(root, task, attempt, *, model='lmrouter/qwen_3_8_flash_next_non_thinking', score=1, tool='new'):
    path = root / 'decomposer' / task / f'attempt-{attempt:03d}'
    save(path / 'execution-1/trace.json', {'model': model,
         'messages': [{'data': {'tool_calls': [{'name': tool}]}}]})
    save(path / 'result.json', {'task_id': task, 'execution_directory': 'execution-1',
         'evaluation': {'status': 'scored', 'metric': 'qa_accuracy', 'score': score}})


def test_selects_once_only_teacher_solved_new_api_depth(tmp_path):
    source = tmp_path / 'tasks.jsonl'
    source.write_text(''.join(json.dumps({'task_id': f'depth-{i}'}) + '\n' for i in range(5)))
    root = tmp_path / 'teacher'
    result(root, 'depth-0', 1)
    result(root, 'depth-0', 2)
    result(root, 'depth-1', 1, score=0)
    result(root, 'depth-2', 1, model='Qwen/Qwen3.5-4B')
    result(root, 'depth-3', 1, tool='spawn_subagent')
    result(root, 'depth-4', 1)
    a = prepare(source, [root], tmp_path / 'a')
    b = prepare(source, [root], tmp_path / 'b')
    assert set(a['task_ids']) == {'depth-0', 'depth-4'}
    assert a['task_count'] == 2
    assert a['tasks_sha256'] == b['tasks_sha256']
    result(root, 'depth-99', 1)
    with pytest.raises(ValueError, match='missing from source'):
        prepare(source, [root], tmp_path / 'bad')


def test_unscored_infra_is_not_assigned_to_zero_success_bucket(tmp_path):
    for task in ('depth-0', 'depth-1'):
        result(tmp_path, task, 1)
        result(tmp_path, task, 2, score=0)
    save(tmp_path / 'decomposer/depth-0/attempt-003/result.json',
         {'status': 'timeout', 'evaluation': {'status': 'timeout', 'score': None}})
    save(tmp_path / 'decomposer/depth-1/attempt-003/result.json',
         {'status': 'error', 'evaluation': {'status': 'evaluation_error', 'score': None}})
    summary = summarize(tmp_path, ['depth-0', 'depth-1'], 3)
    assert summary['success_count_buckets'] == {1: 1}
    assert summary['tasks'][1]['bucket'] is None
    assert summary['tasks'][1]['unresolved_attempts'] == [3]
