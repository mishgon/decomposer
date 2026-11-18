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
    logged = {'critic/rewards/mean': .625, 'val-core/wideseek/train_probe/reward/mean@1': 0.,
              'val-core/deepdive/train_probe/reward/mean@1': 1.,
              'val-core/wideseek/heldout/reward/mean@1': .5,
              'val-core/deepdive/heldout/reward/mean@1': 0.,
              'val-core/workplace-assistant/validation20/reward/mean@1': .85}
    return train, manifest, tasks, selection, logged


def test_common_mixture_and_timeout_zero(panel):
    train, manifest, tasks, selection, logged = panel
    w = weights(manifest, tasks, selection)
    assert set(w) == {'validation'}
    assert sum(w['validation'].values()) == pytest.approx(1)
    assert RewardSummary(train).metrics(logged, 0) == pytest.approx(
        {'rewards/train': .625, 'rewards/validation': .4, 'rewards/ood': .85})


def test_incomplete_logged_panel_is_not_renormalized(panel):
    train, _, _, _, logged = panel
    del logged['val-core/deepdive/heldout/reward/mean@1']
    assert 'rewards/validation' not in RewardSummary(train).metrics(logged, 0)


def test_missing_or_mixed_version_task_is_not_zero(panel):
    train, _, _, _, logged = panel
    p = train / 'episodes/depth-2/trajectory.json'
    d = json.loads(p.read_text()); d['weight_versions']['max_global_steps'] = 1
    save(p, d)
    assert 'rewards/validation' not in RewardSummary(train).metrics(logged, 0)


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
    assert 'rewards/validation' not in summarize(train, logged, 0)
    path.write_text(contents)
    assert summarize(train, logged, 0)['rewards/validation'] == pytest.approx(.4)


def test_old_mixed_panels_use_task_counts(tmp_path):
    train = tmp_path / 'train'
    save(train / 'data/manifest.json', {'heldout_task_ids': ['depth-1', 'depth-2', 'deepdive-1']})
    (train / 'data/tasks.jsonl').write_text('\n'.join(json.dumps({'task_id': t, 'source': s}) for t, s in
        [('depth-1', 'depth'), ('depth-2', 'depth'), ('deepdive-1', 'deepdive-qa_rl')]))
    logged = {'val-core/wideseek/heldout/reward/mean@1': [{'step': 0, 'value': 1., 'time': 100}],
              'val-core/deepdive/heldout/reward/mean@1': [{'step': 0, 'value': 0., 'time': 101}]}
    assert series(train, logged)['rewards/validation'] == [{'step': 0, 'value': 2/3, 'time': 101}]


def test_training_uses_all_rollout_batches_without_probe_baseline(panel):
    train, _, _, _, _ = panel
    logged = {
        'critic/rewards/mean': [{'step': 1, 'value': .3, 'time': 10}, {'step': 2, 'value': .7, 'time': 20}],
        'val-core/wideseek/train_probe/reward/mean@1': [{'step': 0, 'value': .99, 'time': 1}],
        'reward/train': [{'step': 0, 'value': .88, 'time': 1}],
    }
    assert series(train, logged)['rewards/train'] == logged['critic/rewards/mean']
    assert summarize(train, {'critic/rewards/mean': .6}, 3) == {'rewards/train': .6}


def test_validation_does_not_require_training_probes(panel):
    train, manifest, _, _, logged = panel
    manifest.pop('train_probe_task_ids')
    save(train / 'data/manifest.json', manifest)
    assert summarize(train, logged, 0)['rewards/validation'] == pytest.approx(.4)


@pytest.mark.parametrize('binary_logged', [False, True])
def test_width_binarizes_episodes_before_averaging(tmp_path, binary_logged):
    from opd.wideseek.reward import WIDTH_REWARD

    train = tmp_path / 'train'
    ids = ['width-1', 'width-2']
    manifest = {'heldout_task_ids': ids}
    if binary_logged:
        manifest['width_reward'] = WIDTH_REWARD
    save(train / 'data/manifest.json', manifest)
    (train / 'data/tasks.jsonl').write_text('\n'.join(json.dumps({'task_id': t, 'source': 'width'}) for t in ids))
    for tid, score in zip(ids, [.9, .95]):
        save(train / f'episodes/{tid}/trajectory.json', {
            'task_id': tid, 'data_source': 'wideseek/heldout', 'weight_versions': {'global_steps': 0},
            'result': {'status': 'finished', 'finished_at': 1, 'evaluation': {'score': score}}})
    logged = {'val-core/wideseek/heldout/reward/mean@1': .5 if binary_logged else .925,
              'critic/rewards/mean': .5 if binary_logged else .925}
    result = summarize(train, logged, 0)
    assert result['rewards/validation'] == .5  # Not threshold(mean F1), which would be 1.
    if binary_logged:
        assert result['rewards/train'] == .5
    else:
        assert 'rewards/train' not in result  # No exact historical training-batch membership.
    logged['val-core/wideseek/heldout/reward/mean@1'] = .2
    with pytest.raises(ValueError, match='Raw results disagree'):
        summarize(train, logged, 0)
