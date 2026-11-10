import json
from pathlib import Path
from unittest.mock import Mock

from opd.synth.prepare import prepare


def test_frozen_panel_is_resolved_once_and_validation_only(tmp_path, monkeypatch):
    import datasets
    import pandas as pd

    root = Path(__file__).resolve().parents[2]
    task_ids = json.loads((root / 'gyms/workplace_assistant/validation20.json').read_text())
    assert len(task_ids) == len(set(task_ids)) == 20
    # Dataset order must not affect ID selection; unselected tasks stay excluded.
    rows = [{'id': task_id, 'responses_create_params': {'input': [
                {'role': 'user', 'content': f'Task {task_id}'}]},
             'ground_truth': [{'name': 'tool', 'arguments': json.dumps({'id': task_id})}],
             'environment_name': 'workplace_assistant'}
            for task_id in [*reversed(task_ids), -1]]
    load = Mock(return_value=rows)
    monkeypatch.setattr(datasets, 'load_dataset', load)
    prepare(tmp_path / 'data', workplace_validation=True)
    load.assert_called_once_with('nvidia/Nemotron-RL-agent-workplace_assistant',
        revision='f674d9bba17820d90f14eefe9156fdc018a0db60', split='validation')
    train = pd.read_parquet(tmp_path / 'data/train.parquet')
    val = pd.read_parquet(tmp_path / 'data/evaluation.parquet')
    assert set(train.data_source) == {'synth/train'}
    assert len(train) == 128
    workplace = val[val.data_source == 'workplace-assistant/validation20']
    assert len(workplace) == 20 and len(val) == 32
    assert set(workplace.agent_name) == {'workplace_decomposer'}
    assert [int(info['task_id']) for info in workplace.extra_info] == task_ids
    by_id = {row['id']: row for row in rows}
    for row in workplace.itertuples():
        task = json.loads(row.extra_info['task'])
        assert task['id'] == int(row.extra_info['task_id'])
        assert task['prompt'] == row.prompt[0]['content'] == f"Task {task['id']}"
        assert task['ground_truth'] == by_id[task['id']]['ground_truth']
        assert task['environment_name'] == 'workplace_assistant'
