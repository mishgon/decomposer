import json

import pytest

from gyms.wideseek.prepare import normalize
from opd.wideseek.prepare import check, prepare


def test_frozen_data_excludes_references_from_model_input(tmp_path):
    pd = pytest.importorskip('pandas')
    pytest.importorskip('pyarrow')
    source = tmp_path / 'tasks.jsonl'
    task = normalize({'question': 'Find cities', 'answer': 'secret reference', 'unique_columns': ['City']}, 'width', 0)
    source.write_text(json.dumps(task) + '\n')
    out = tmp_path / 'prepared'
    manifest = prepare(source, out, [task['task_id']])
    assert check(out) == manifest
    for split in ('train', 'evaluation'):
        frame = pd.read_parquet(out / f'{split}.parquet')
        assert len(frame) == 1
        assert 'secret reference' not in str(frame.to_dict())
        assert frame.iloc[0]['agent_name'] == 'wideseek_decomposer'
    assert 'secret reference' in (out / 'tasks.jsonl').read_text()
    (out / 'tasks.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Prepared data changed'):
        check(out)
    with pytest.raises(FileExistsError):
        prepare(source, out, [task['task_id']])


def test_duplicate_tasks_rejected_before_creating_output(tmp_path):
    source = tmp_path / 'tasks.jsonl'
    source.write_text(json.dumps({'task_id': 'a', 'question_sha256': 'same'}) + '\n')
    with pytest.raises(ValueError, match='distinct task IDs'):
        prepare(source, tmp_path / 'out', ['a', 'a'])
    assert not (tmp_path / 'out').exists()
