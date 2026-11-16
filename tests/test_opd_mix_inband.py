import hashlib
import json
from collections import Counter

import pandas as pd
import pytest

from gyms.wideseek.prepare import normalize
from opd.mix import overfit, select_inband


def pool(tmp_path, name, outcomes, require_success=False):
    root = tmp_path / name
    run = root / 'eval'
    run.mkdir(parents=True)
    tasks = [normalize({'question': f'{name} question {i}', 'answer': 'SECRET'}, 'depth', i)
             for i in range(len(outcomes))]
    for t in tasks:
        t['task_id'] = name + '-' + t['task_id']
    raw = ''.join(json.dumps(t) + '\n' for t in tasks).encode()
    source = root / 'tasks.jsonl'
    source.write_bytes(raw)
    (root / 'launch.json').write_text(json.dumps({'initialization': {'tag': 'v1.7.0', 'sha256': 'same'},
        'max_pass_rate': .5, 'require_success': require_success}))
    (run / 'manifest.json').write_text(json.dumps({'settings': {'data_sha256': hashlib.sha256(raw).hexdigest(),
        'repetitions': 4, 'modes': ['decomposer'], 'tasks': [t['task_id'] for t in tasks]}}))
    for task, results in zip(tasks, outcomes):
        for number, score in enumerate(results, 1):
            path = run / 'decomposer' / task['task_id'] / f'attempt-{number:03d}' / 'result.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'task_id': task['task_id'], 'status': 'finished', 'finished_at': 1,
                'evaluation': {'status': 'infrastructure_error' if score is None else 'scored', 'score': score}}))
    return run, source


def test_snapshot_requires_four_resolved_attempts_and_preserves_each_pool_rule(tmp_path):
    known = pool(tmp_path, 'known', [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, None, 0], [1, 0]])
    unseen = pool(tmp_path, 'unseen', [[0, 0, 0, 0], [1, 0, 0, 0]], True)
    selected = select_inband.select([known[0], unseen[0]], [known[1], unseen[1]], tmp_path / 'selected')
    assert selected['tasks'] == 3
    assert set(selected['task_ids']) == {'known-depth-00000', 'known-depth-00001', 'unseen-depth-00001'}
    assert [r['qualified'] for r in selected['runs']] == [2, 1]
    assert all(len(row['attempts']) == 4 for row in selected['evidence'])
    assert selected['tasks_sha256'] == hashlib.sha256((tmp_path / 'selected/tasks.jsonl').read_bytes()).hexdigest()
    known[1].write_text(known[1].read_text() + '\n')
    with pytest.raises(ValueError, match='differs'):
        select_inband.select([known[0]], [known[1]], tmp_path / 'bad')


def test_mixed_selection_visits_all_tasks_in_exactly_64_batches(tmp_path, monkeypatch):
    tasks = [normalize({'question': f'Question {i}', 'answer': 'SECRET'}, 'depth', i) for i in range(149)]
    tasks[-1]['source'] = 'deepdive-qa_rl'
    source = tmp_path / 'tasks.jsonl'
    source.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    monkeypatch.setattr(overfit, 'validation_rows', lambda: [])
    data = tmp_path / 'data'
    manifest = overfit.prepare(source, data, steps=64)
    rows = pd.read_parquet(data / 'train.parquet').to_dict('records')
    counts = Counter(r['extra_info']['task_id'] for r in rows)
    assert len(rows) == 512 and manifest['optimizer_steps'] == 64
    assert len(counts) == 149 and max(counts.values()) - min(counts.values()) == 1
    assert {r['data_source'] for r in rows} == {'wideseek/train', 'deepdive/train'}
    assert all('SECRET' not in str(r['prompt']) for r in rows)
    assert len(pd.read_parquet(data / 'evaluation.parquet')) == 6
    with pytest.raises(ValueError, match='visit every'):
        overfit.prepare(source, tmp_path / 'short', steps=1)


def test_holdout_is_stratified_disjoint_and_stable_under_input_reordering(tmp_path, monkeypatch):
    tasks = []
    for source, count in [('depth', 60), ('width', 69), ('deepdive-qa_rl', 47), ('deepdive-qa_sft', 13)]:
        for i in range(count):
            task = normalize({'question': f'{source} question {i}', 'answer': 'SECRET'}, 'depth', len(tasks))
            task.update(source=source, task_id=f'{source}-{i:05d}')
            tasks.append(task)
    source = tmp_path / 'tasks.jsonl'
    source.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    monkeypatch.setattr(overfit, 'validation_rows', lambda: [])
    data = tmp_path / 'data'
    manifest = overfit.prepare(source, data, steps=64, holdout_tasks=20)
    train = pd.read_parquet(data / 'train.parquet').to_dict('records')
    evaluation = pd.read_parquet(data / 'evaluation.parquet').to_dict('records')
    train_ids = {r['extra_info']['task_id'] for r in train}
    heldout_ids = {r['extra_info']['task_id'] for r in evaluation if r['data_source'].endswith('/heldout')}
    probe_ids = {r['extra_info']['task_id'] for r in evaluation if r['data_source'].endswith('/train_probe')}
    assert len(train) == 512 and len(train_ids) == manifest['train_tasks'] == 169
    assert len(heldout_ids) == manifest['heldout_tasks'] == 20
    assert not train_ids & heldout_ids and probe_ids <= train_ids
    assert train_ids | heldout_ids == {t['task_id'] for t in tasks}
    hashes = {t['task_id']: t['question_sha256'] for t in tasks}
    assert not {hashes[i] for i in train_ids} & {hashes[i] for i in heldout_ids}
    assert manifest['heldout_sources'] == {'depth': 6, 'width': 7, 'deepdive-qa_rl': 5, 'deepdive-qa_sft': 2}
    assert not manifest['overfit']
    assert {r['data_source'] for r in evaluation} >= {'wideseek/heldout', 'deepdive/heldout'}
    assert all('SECRET' not in str(r['prompt']) for r in train + evaluation)
    source.write_text(''.join(json.dumps(t) + '\n' for t in reversed(tasks)))
    repeat = overfit.prepare(source, tmp_path / 'repeat', steps=64, holdout_tasks=20)
    assert repeat['heldout_task_ids'] == manifest['heldout_task_ids']
    assert repeat['train_task_ids'] == manifest['train_task_ids']
    for count in (-1, 3, 186):
        with pytest.raises(ValueError, match='every source'):
            overfit.prepare(source, tmp_path / f'bad-{count}', steps=64, holdout_tasks=count)
