import asyncio


def test_collection_uses_gym_agent_without_sequence_limit(monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    from types import SimpleNamespace

    factory = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, 'agents', SimpleNamespace(decomposer=factory))
    path = Path(__file__).resolve().parents[1] / 'sft/toolathlon_gym/agents.py'
    spec = importlib.util.spec_from_file_location('collection_agents_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import inspect

    assert not inspect.signature(module.decomposer).parameters
    assert asyncio.run(module.decomposer()) == {}


def test_skip_is_terminal_but_never_successful(tmp_path):
    import json
    from sft.toolathlon_gym.collection import count_statuses, next_attempt
    from sft.toolathlon_gym.scheduler import TERMINAL_EPISODE_STATUSES, load_launch_outcome
    from gyms.wideseek.metrics import qualifies

    result = tmp_path / 'evaluation.json'
    result.write_text(json.dumps({'pass': True, 'stop_reason': 'sequence_limit'}))
    assert not load_launch_outcome('task', str(result)).qualifies(.9)
    assert 'skipped' in TERMINAL_EPISODE_STATUSES
    assert not qualifies({'status': 'skipped', 'evaluation': {'status': 'scored', 'score': 1.}}, .9)
    saved = {'attempt': 1, 'status': 'skipped', 'stop_reason': 'sequence_limit', 'error': None}
    path = tmp_path / 'attempts/task/rep-001/attempt-001'
    path.mkdir(parents=True)
    (path / 'attempt.json').write_text(json.dumps(saved))
    episode = {'task': 'task', 'repetition': 1, 'attempts': [], 'status': 'pending'}
    assert next_attempt(tmp_path, episode) == (2, True)
    assert episode['status'] == 'skipped'
    counts = count_statuses({'episodes': [episode]})
    assert counts['skipped'] == counts['total'] == 1
    assert counts['failed'] == counts['completed'] == 0


def test_collection_entrypoint_defaults_to_guarded_image(monkeypatch):
    from sft.toolathlon_gym import run
    calls = []
    monkeypatch.setattr(run.collection, 'main', lambda argv, **kw: calls.append((argv, kw)))
    run.main(['--tasks', 'task'])
    assert calls[0][1]['default_image'] == 'decomposer-toolathlon-sft:latest'


def test_collection_accepts_image_without_sequence_limit(tmp_path, monkeypatch):
    import json
    from subprocess import CompletedProcess
    from sft.toolathlon_gym import collection

    (tmp_path / 'tasks/finalpool/task').mkdir(parents=True)
    def docker(*args):
        assert args == ('image', 'inspect', 'plain-gym')
        return CompletedProcess(args, 0, stdout=json.dumps([{'Config': {'Labels': {}}}]))
    monkeypatch.setattr(collection, 'execute_episode', lambda args, **kwargs: {
        'attempt': kwargs['attempt'], 'status': 'completed', 'score': True,
    })
    manifest = collection.main(['--tasks', 'task'], repo_root=tmp_path, toolathlon_root=tmp_path,
                               default_artifacts_dir=tmp_path/'artifacts', default_image='plain-gym', docker=docker)
    assert manifest['status'] == 'completed'
    assert manifest['episodes'][0]['score'] is True
