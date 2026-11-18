import json
import pytest

from opd.reward_summary import RewardSummary, series, summarize, weights


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def panel(tmp_path):
    train = tmp_path / 'train'
    tasks = [{'task_id': f'depth-{i}', 'source': 'depth'} for i in range(8)]
    tasks += [{'task_id': f'deepdive-{i}', 'source': 'deepdive-qa_rl'} for i in range(2)]
    manifest = {'stratify_by': 'dataset_and_passes', 'pool_cell_counts': {'depth/0': 8, 'deepdive/1': 2},
                'train_probe_task_ids': ['depth-0', 'deepdive-0'],
                'heldout_task_ids': ['depth-1', 'depth-2', 'deepdive-1']}
    selection = {'evidence': [{'task_id': t['task_id'], 'passes': int(t['source'].startswith('deepdive'))} for t in tasks]}
    save(train / 'data/manifest.json', manifest)
    (train / 'data/tasks.jsonl').write_text('\n'.join(json.dumps(t) for t in tasks))
    save(tmp_path / 'selected/selection.json', selection)
    for tid, source, score in [('depth-0', 'wideseek/train_probe', 0), ('deepdive-0', 'deepdive/train_probe', 1),
                                ('depth-1', 'wideseek/heldout', 1), ('depth-2', 'wideseek/heldout', None),
                                ('deepdive-1', 'deepdive/heldout', 0)]:
        result = {'status': 'timeout' if score is None else 'finished', 'finished_at': 10,
                  'evaluation': {'score': score}}
        save(train / 'episodes' / tid / 'trajectory.json', {'task_id': tid, 'data_source': source,
             'weight_versions': {'global_steps': 0}, 'result': result})
    logged = {'val-core/wideseek/train_probe/reward/mean@1': 0.,
              'val-core/deepdive/train_probe/reward/mean@1': 1.,
              'val-core/wideseek/heldout/reward/mean@1': .5,
              'val-core/deepdive/heldout/reward/mean@1': 0.,
              'val-core/workplace-assistant/validation20/reward/mean@1': .85}
    return train, manifest, tasks, selection, logged


def test_common_mixture_and_timeout_zero(panel):
    train, manifest, tasks, selection, logged = panel
    w = weights(manifest, tasks, selection)
    assert w['train'] == {'depth-0': .8, 'deepdive-0': .2}
    assert sum(w['validation'].values()) == pytest.approx(1)
    assert RewardSummary(train).metrics(logged, 0) == pytest.approx(
        {'reward/train': .2, 'reward/validation': .4, 'reward/ood': .85})


def test_incomplete_logged_panel_is_not_renormalized(panel):
    train, _, _, _, logged = panel
    del logged['val-core/deepdive/heldout/reward/mean@1']
    assert 'reward/validation' not in RewardSummary(train).metrics(logged, 0)


def test_missing_or_mixed_version_task_is_not_zero(panel):
    train, _, _, _, logged = panel
    p = train / 'episodes/depth-2/trajectory.json'
    d = json.loads(p.read_text()); d['weight_versions']['max_global_steps'] = 1
    save(p, d)
    assert 'reward/validation' not in RewardSummary(train).metrics(logged, 0)


def test_judge_failure_is_explicit(panel):
    train, _, _, _, logged = panel
    p = train / 'episodes/depth-2/trajectory.json'
    d = json.loads(p.read_text());d['result']['status'] = 'error'
    save(p, d)
    with pytest.raises(ValueError, match='Invalid validation score'):
        RewardSummary(train).metrics(logged, 0)


def test_latest_retry_must_match_trainer_mean(panel):
    train, _, _, _, logged = panel
    d = json.loads((train / 'episodes/depth-1/trajectory.json').read_text())
    d['result'].update(finished_at=20, evaluation={'score': 0})
    save(train / 'episodes/retry/trajectory.json', d)
    with pytest.raises(ValueError, match='Raw results disagree'):
        RewardSummary(train).metrics(logged, 0)


def test_absent_stratum_rejected(panel):
    _, manifest, tasks, selection, _ = panel
    manifest['heldout_task_ids'] = ['depth-1', 'depth-2']
    with pytest.raises(ValueError, match='represent every stratum'):
        weights(manifest, tasks, selection)


def test_watcher_can_start_before_manifest(panel):
    train, _, _, _, logged = panel
    path = train / 'data/manifest.json'
    contents = path.read_text()
    path.unlink()
    assert 'reward/train' not in summarize(train, logged, 0)
    path.write_text(contents)
    assert summarize(train, logged, 0)['reward/train'] == pytest.approx(.2)


def test_old_mixed_panels_use_task_counts(tmp_path):
    train = tmp_path / 'train'
    save(train / 'data/manifest.json', {'heldout_task_ids': ['depth-1', 'depth-2', 'deepdive-1']})
    (train / 'data/tasks.jsonl').write_text('\n'.join(json.dumps({'task_id': t, 'source': s}) for t, s in
        [('depth-1', 'depth'), ('depth-2', 'depth'), ('deepdive-1', 'deepdive-qa_rl')]))
    logged = {'val-core/wideseek/heldout/reward/mean@1': [{'step': 0, 'value': 1., 'time': 100}],
              'val-core/deepdive/heldout/reward/mean@1': [{'step': 0, 'value': 0., 'time': 101}]}
    assert series(train, logged)['reward/validation'] == [{'step': 0, 'value': 2/3, 'time': 101}]
