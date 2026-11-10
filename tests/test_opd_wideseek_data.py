import json
from pathlib import Path

import pytest

from gyms.wideseek.prepare import normalize
from opd.wideseek.prepare import check, prepare


@pytest.fixture(autouse=True)
def workplace_data(monkeypatch):
    import datasets
    ids = json.loads((Path(__file__).resolve().parents[1] /
                      'gyms/workplace_assistant/validation20.json').read_text())
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **kw: [
        {'id': i, 'responses_create_params': {'input': [{'content': f'Task {i}'}]},
         'ground_truth': [], 'environment_name': 'workplace_assistant'} for i in reversed(ids)])


@pytest.fixture(autouse=True)
def frozen_panel(tmp_path, monkeypatch):
    from opd.wideseek import prepare as module
    tasks = [normalize({"question": f"Find city {i}", "answer": "reference",
                        "unique_columns": ["City"]}, "width", i) for i in range(24)]
    path = tmp_path / "panel.json"
    path.write_text(json.dumps({name: [{k: tasks[i][k] for k in ("task_id", "question_sha256")}
                                     for i in ids]
                               for name, ids in [("train_probe", range(4)), ("heldout", range(16, 24))]}))
    monkeypatch.setattr(module, "VALIDATION_PANEL", path)


def source_file(tmp_path, count=24):
    source = tmp_path / 'tasks.jsonl'
    tasks = [normalize({'question': f'Find city {i}', 'answer': f'secret reference {i}',
                        'unique_columns': ['City']}, 'width', i) for i in range(count)]
    source.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    return source


def test_full_pool_disjoint_questions_and_unchanged_ood_panel(tmp_path):
    pd = pytest.importorskip('pandas')
    pytest.importorskip('pyarrow')
    source = source_file(tmp_path, 40)
    out = tmp_path / 'prepared'
    manifest = prepare(source, out)
    assert check(out) == manifest
    train = pd.read_parquet(out / 'train.parquet')
    validation = pd.read_parquet(out / 'evaluation.parquet')
    assert len(train) == 32 and len(validation) == 32
    assert train.extra_info.map(lambda x: x['task_id']).value_counts().tolist() == [1] * 32
    assert set(manifest['train_task_ids']).isdisjoint(manifest['heldout_task_ids'])
    assert validation.data_source.value_counts().to_dict() == {
        'workplace-assistant/validation20': 20, 'wideseek/heldout': 8, 'wideseek/train_probe': 4}
    assert 'secret reference' not in str(train.to_dict()) + str(validation.to_dict())
    panel = json.loads((Path(__file__).resolve().parents[1] /
                       'gyms/workplace_assistant/validation20.json').read_text())
    ood = validation[validation.data_source == 'workplace-assistant/validation20']
    assert [r['task_id'] for r in ood.extra_info] == [str(t) for t in panel]
    assert [r[0]['content'] for r in ood.prompt] == [f'Task {t}' for t in panel]
    (out / 'tasks.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Prepared data changed'):
        check(out)
    with pytest.raises(FileExistsError):
        prepare(source, out)


def test_duplicate_questions_cannot_leak_across_splits(tmp_path):
    source = source_file(tmp_path, 23)
    with source.open('a') as stream:
        stream.write(source.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='Frozen validation task changed or missing'):
        prepare(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_validation_is_stable_when_source_order_changes(tmp_path):
    source = source_file(tmp_path, 40)
    first = prepare(source, tmp_path / 'first')
    source.write_text('\n'.join(reversed(source.read_text().splitlines())) + '\n')
    second = prepare(source, tmp_path / 'second')
    assert first['heldout_task_ids'] == second['heldout_task_ids']
    assert set(first['train_task_ids']) == set(second['train_task_ids'])


def test_depth_pool_rotates_unique_tasks_and_excludes_frozen_holdout(tmp_path, monkeypatch):
    import pandas as pd
    from opd.wideseek import prepare as module
    tasks = [normalize({'question': f'Depth {i}', 'answer': 'Reference'}, 'depth', i)
             for i in range(40)]
    source = tmp_path / 'depth.jsonl'
    source.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    panel = tmp_path / 'depth-panel.json'
    panel.write_text(json.dumps({name: [{k: tasks[i][k] for k in ('task_id', 'question_sha256')}
                                      for i in ids]
                                for name, ids in [('train_probe', range(4)), ('heldout', range(4, 12))]}))
    monkeypatch.setattr(module, 'DEPTH_VALIDATION_PANEL', panel)
    manifest = prepare(source, tmp_path / 'first')
    assert manifest['source'] == 'depth'
    assert manifest['train_tasks'] == 32 and manifest['attempts_per_task_per_epoch'] == 1
    assert set(manifest['heldout_task_ids']).isdisjoint(manifest['train_task_ids'])
    train = pd.read_parquet(tmp_path / 'first/train.parquet')
    assert len({row['task_id'] for row in train.extra_info}) == 32
    assert all(r'\boxed{}' in prompt[0]['content'] for prompt in train.prompt)
    val = pd.read_parquet(tmp_path / 'first/evaluation.parquet')
    assert len(val) == 32
    for info in val[val.data_source == 'workplace-assistant/validation20'].extra_info:
        assert json.loads(info['task'])['id'] == int(info['task_id'])
    source.write_text(''.join(json.dumps(t) + '\n' for t in reversed(tasks)))
    repeat = prepare(source, tmp_path / 'second')
    assert repeat['heldout_task_ids'] == manifest['heldout_task_ids']
    assert set(repeat['train_task_ids']) == set(manifest['train_task_ids'])
