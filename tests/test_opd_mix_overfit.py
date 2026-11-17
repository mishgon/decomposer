from collections import Counter
import hashlib
import json

import pandas as pd
import pytest

from gyms.wideseek.prepare import normalize
from opd.mix import overfit
from opd.wideseek.prepare import check


def test_overfit_preserves_selection_and_provides_32_full_batches(tmp_path, monkeypatch):
    tasks = [normalize({'question': f'Question {i}', 'answer': 'SECRET ANSWER'}, 'depth', i)
             for i in range(6)]
    source = tmp_path / 'selected.jsonl'
    source.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    monkeypatch.setattr(overfit, 'validation_rows', lambda: [])
    output = tmp_path / 'data'
    manifest = overfit.prepare(source, output)
    assert check(output) == manifest
    rows = pd.read_parquet(output / 'train.parquet').to_dict('records')
    assert len(rows) == 32 * 8
    counts = Counter(r['extra_info']['task_id'] for r in rows)
    assert set(counts) == {t['task_id'] for t in tasks}
    assert max(counts.values()) - min(counts.values()) == 1
    assert all('SECRET ANSWER' not in str(r['prompt']) for r in rows)
    assert len(pd.read_parquet(output / 'evaluation.parquet')) == 6
    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps({'heldout': tasks[:1]}))
    monkeypatch.setattr(overfit, 'DEPTH_VALIDATION_PANEL', panel)
    with pytest.raises(ValueError, match='overlap'):
        overfit.prepare(source, tmp_path / 'leak')
    assert not (tmp_path / 'leak').exists()


def test_frozen_split_keeps_exact_rows_and_rejects_changed_budget_or_data(tmp_path, monkeypatch):
    tasks = [normalize({'question': f'Question {i}', 'answer': 'SECRET'}, 'depth', i)
             for i in range(6)]
    source = tmp_path / 'tasks.jsonl'
    source.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    monkeypatch.setattr(overfit, 'validation_rows', lambda: [])
    data = tmp_path / 'frozen'
    manifest = overfit.prepare(source, data, steps=2, holdout_tasks=2)
    output = tmp_path / 'staged'
    assert overfit.stage_frozen(source, output, directory=data, steps=2, holdout_tasks=2) == manifest
    for name in manifest['files']:
        assert (output / name).read_bytes() == (data / name).read_bytes()
    with pytest.raises(ValueError, match='budget or holdout'):
        overfit.frozen_manifest(source, data, steps=3, holdout_tasks=2)
    with pytest.raises(ValueError, match='budget or holdout'):
        overfit.frozen_manifest(source, data, steps=2, holdout_tasks=1)
    train = pd.read_parquet(data / 'train.parquet')
    train.at[0, 'extra_info'] = {'task_id': manifest['heldout_task_ids'][0]}
    train.to_parquet(data / 'train.parquet', index=False)
    with pytest.raises(ValueError, match='Prepared data changed'):
        overfit.frozen_manifest(source, data, steps=2, holdout_tasks=2)
    manifest['files']['train.parquet'] = hashlib.sha256((data / 'train.parquet').read_bytes()).hexdigest()
    (data / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='budget or holdout'):
        overfit.frozen_manifest(source, data, steps=2, holdout_tasks=2)
