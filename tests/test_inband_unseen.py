import json

from evals.wideseek import unseen
from gyms.wideseek.prepare import normalize


def test_complement_excludes_sft_pool_holdouts_and_question_aliases(tmp_path, monkeypatch):
    tasks = [normalize({'question': f'Q{i}', 'answer': 'A'}, 'depth', i) for i in range(9)]
    tasks[3]['question_sha256'] = tasks[0]['question_sha256']  # SFT alias.
    tasks[4]['question_sha256'] = tasks[1]['question_sha256']  # Existing-pool alias.
    tasks[8]['question_sha256'] = tasks[7]['question_sha256']  # Duplicate unseen question.
    source, pool, index, panel = [tmp_path / f'{name}.json' for name in ('all', 'pool', 'sft', 'holdout')]
    source.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    pool.write_text(json.dumps(tasks[1]) + '\n')
    index.write_text(json.dumps({'task_ids': [tasks[0]['task_id']]}))
    panel.write_text(json.dumps({'heldout': [tasks[2]]}))
    monkeypatch.setattr(unseen, 'DEPTH_VALIDATION_PANEL', panel)
    first = unseen.prepare([source], pool, index, tmp_path / 'first')
    unseen.prepare([source], pool, index, tmp_path / 'second')
    data = (tmp_path / 'first/tasks.jsonl').read_bytes()
    assert data == (tmp_path / 'second/tasks.jsonl').read_bytes()
    assert {json.loads(line)['task_id'] for line in data.splitlines()} == {'depth-00005', 'depth-00006', 'depth-00007'}
    assert first['historical_success_required'] is False
