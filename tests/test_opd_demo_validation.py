"""Optional OPD integration tests; standalone demo tests cover the gym separately."""
import json
import subprocess
import sys

import pytest

from gyms.decomposer_demo import grade, load_task
from gyms.decomposer_demo.calendar import KEEP_IDS, TASK_ID
from opd.decomposer_demo.reward import compute_score


def expected_events():
    return [e for e in load_task()['calendar']['events'] if e['id'] in KEEP_IDS]


def test_opd_adapter_uses_harness_state_not_model_claim():
    reward = compute_score('decomposer-demo/validation', 'I failed', TASK_ID,
                           {'final_state': expected_events()})
    assert reward == grade(TASK_ID, expected_events())
    assert compute_score('decomposer-demo/validation', 'Everything is correct', TASK_ID,
                         {'final_state': []})['score'] == 0
    with pytest.raises(ValueError, match='final_state'):
        compute_score('decomposer-demo/validation', '', TASK_ID)
    with pytest.raises(ValueError, match='Unknown'):
        compute_score('decomposer-demo/validation', '', '', {'final_state': []})
    with pytest.raises(ValueError, match='validation-only'):
        compute_score('decomposer-demo/train', '', TASK_ID, {'final_state': []})


def test_gym_and_reward_import_without_site_packages():
    subprocess.run([sys.executable, '-S', '-c',
        'from gyms.decomposer_demo import load_task; '
        'from opd.decomposer_demo.reward import compute_score; '
        'assert load_task()["benchmark"] == "decomposer-demo"'], check=True)


def test_validation_dataset_is_opt_in_and_excluded_from_training(tmp_path):
    import pandas as pd
    from opd.synth.prepare import prepare
    from gyms.decomposer_demo.tasks import task_ids, task_spec

    plain, demo = tmp_path / 'plain', tmp_path / 'demo'
    prepare(plain)
    prepare(demo, demo_validation=True)
    train = pd.read_parquet(demo / 'train.parquet')
    pd.testing.assert_frame_equal(train, pd.read_parquet(plain / 'train.parquet'))
    assert set(train.data_source) == {'synth/train'}
    assert len(pd.read_parquet(plain / 'evaluation.parquet')) == 12
    validation = pd.read_parquet(demo / 'evaluation.parquet')
    assert len(validation) == 52
    rows = validation[validation.agent_name == 'demo_decomposer'].to_dict('records')
    assert len(rows) == 40
    assert {(r['extra_info']['task_id'], r['extra_info']['language']) for r in rows} == {
        (task_id, language) for task_id in task_ids() for language in ('en', 'ru')}
    for row in rows:
        task_id, language = row['extra_info']['task_id'], row['extra_info']['language']
        assert row['data_source'] == f'decomposer-demo/{language}'
        assert row['prompt'][0]['content'] == task_spec(task_id, language)['prompt']
        assert row['reward_model']['ground_truth'] == task_id
    manifest = json.loads((demo / 'manifest.json').read_text())
    assert manifest['demo_validation_tasks'] == 20
    assert manifest['demo_validation_rows'] == 40


def test_opd_language_sources_use_same_reward():
    from gyms.decomposer_demo.frontend import seed
    from gyms.decomposer_demo.tasks import task_spec

    task_id = 'frontend-notebook'
    task = task_spec(task_id, 'ru')
    state = seed('ru')
    state['/styles.css'] += '\n' + '\n'.join(
        f'{selector} {{ {key}: {value}; }}' for (selector, key), value in task['targets'].items())
    assert compute_score('decomposer-demo/ru', '', task_id, {'final_state': state})['score'] == 1
    with pytest.raises(ValueError, match='language'):
        compute_score('decomposer-demo/ru', '', task_id, {'final_state': state, 'language': 'en'})
