import json
from pathlib import Path

from opd.synth.prepare import prepare
from gyms.workplace_assistant.in_process import Workplace


def test_frozen_panel_is_native_and_validation_only(tmp_path):
    import pandas as pd
    root = Path(__file__).resolve().parents[2]
    panel = json.loads((root / 'gyms/workplace_assistant/validation20.json').read_text())
    assert len(panel) == len({r['id'] for r in panel}) == 20
    for row in panel:
        gym = Workplace()
        assert not gym.score(row['ground_truth'])
        gym.actions = row['ground_truth']
        assert gym.score(row['ground_truth'])
    prepare(tmp_path / 'data', workplace_validation=True)
    train = pd.read_parquet(tmp_path / 'data/train.parquet')
    val = pd.read_parquet(tmp_path / 'data/evaluation.parquet')
    assert set(train.data_source) == {'synth/train'}
    assert len(train) == 128
    workplace = val[val.data_source == 'workplace-assistant/validation20']
    assert len(workplace) == 20 and len(val) == 32
    assert set(workplace.agent_name) == {'workplace_decomposer'}
    assert {int(info['task_id']) for info in workplace.extra_info} == {r['id'] for r in panel}
