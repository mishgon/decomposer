import json
from types import SimpleNamespace

from opd.wideseek import watch


def point(step, value):
    return SimpleNamespace(step=step, value=value, wall_time=step * 60)


def test_resume_history_keeps_saved_progress_and_prefers_new_metrics(tmp_path, monkeypatch):
    old = tmp_path / 'old/train'
    new = tmp_path / 'new/train'
    new.mkdir(parents=True)
    (new / 'run.json').write_text(json.dumps({'resume_from': str(old / 'checkpoints/global_step_31')}))
    monkeypatch.setattr(watch, 'scalars', lambda path: {
        'training/global_step': [point(31, 31), point(32, 32)],
        'reward': [point(0, .1), point(24, .5), point(32, .9)],
    })
    series, step = watch.resume_history(new, {})
    assert step == 31
    assert [p.step for p in series['training/global_step']] == [31]
    assert [p.value for p in series['reward']] == [.1, .5]
    series, step = watch.resume_history(new, {'reward': [point(24, .6), point(32, .7)]})
    assert [p.value for p in series['reward']] == [.1, .6, .7]


def test_fresh_run_has_no_inherited_progress(tmp_path):
    series, step = watch.resume_history(tmp_path, {'reward': [point(0, .3)]})
    assert step == 0
    assert series['reward'][0].value == .3
