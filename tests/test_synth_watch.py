import json
from types import SimpleNamespace

from opd.synth import watch


def test_original_sparkline_and_measured_eta():
    assert watch.trend([1, 2, 3]) == '▁▅█'
    assert watch.trend([2, 2]) == '▄▄'
    assert watch.trend([float('nan')]) == '--'
    points = [SimpleNamespace(step=1, wall_time=100), SimpleNamespace(step=2, wall_time=160)]
    assert watch.update_eta(points, 2, 8) == 360
    assert watch.update_eta(points[:1], 1, 8) is None


def test_job_lifecycle_does_not_claim_stale_manifest_is_running(tmp_path, monkeypatch, capsys):
    run = tmp_path / 'smoke'
    (run / 'baseline').mkdir(parents=True)
    (run / 'train').mkdir()
    (run / 'train-submission.json').write_text(json.dumps({'commit': 'abc', 'result': {'job_name': 'job'}}))
    (run / 'baseline/manifest.json').write_text(json.dumps({'tasks': 8, 'repetitions': 4,
        'status': 'running', 'started_at': 100, 'episodes': [{'score': 1, 'status': 'finished'}]*4}))
    monkeypatch.setattr(watch, 'scalars', lambda _: {})
    monkeypatch.setattr(watch, 'planned_updates', lambda _: None)
    for state in ('pending', 'failed', 'running'):
        monkeypatch.setattr(watch, 'job_status', lambda _, state=state: {'status': state})
        watch.show(tmp_path)
        output = capsys.readouterr().out
        assert state.upper() in output
        assert ('ETA baseline only:' in output) == (state == 'running')
        if state != 'running':
            assert 'ETA: -- (queued, stopped, or status unavailable)' in output


def test_partial_manifest_and_missing_run(tmp_path, capsys):
    path = tmp_path / 'manifest.json'
    path.write_text('{')
    assert watch.read_json(path) == {}
    watch.show(tmp_path)
    assert 'No OPD submissions' in capsys.readouterr().out


def test_async_plan_uses_rollout_budget_not_ignored_training_steps(tmp_path):
    import pytest
    pytest.importorskip('yaml')
    (tmp_path / 'hydra/.hydra').mkdir(parents=True)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/manifest.json').write_text(json.dumps({'train_tasks': 16, 'attempts_per_task_per_epoch': 8}))
    (tmp_path / 'hydra/.hydra/config.yaml').write_text('''
trainer: {total_epochs: 4, total_training_steps: 999}
rollout: {total_rollout_steps: 16}
actor_rollout_ref: {actor: {ppo_mini_batch_size: 8}}
async_training: {require_batches: 1, trigger_parameter_sync_step: 1}
''')
    assert watch.planned_updates(tmp_path) == 2


def test_single_attempt_demo_validation_is_displayed(tmp_path, monkeypatch, capsys):
    run = tmp_path / 'demo'
    run.mkdir()
    (run / 'train-submission.json').write_text(json.dumps({'result': {'job_name': 'job'}}))
    monkeypatch.setattr(watch, 'job_status', lambda _: {'status': 'running'})
    monkeypatch.setattr(watch, 'concurrency', lambda *args: (8, 8, 2, 0))
    monkeypatch.setattr(watch, 'planned_updates', lambda _: 2)
    monkeypatch.setattr(watch, 'scalars', lambda _: {
        'val-core/decomposer-demo/en/reward/mean@1': [SimpleNamespace(step=0, value=0.5)]})
    watch.show(tmp_path)
    output = capsys.readouterr().out
    assert 'decomposer-demo/en/reward/mean@1' in output
    assert '0.500' in output
