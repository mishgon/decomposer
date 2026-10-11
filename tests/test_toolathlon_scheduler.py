import json

import pytest

from sft.toolathlon_gym.scheduler import task_rank


@pytest.mark.parametrize('status,trace_exists,successes', [
    ('completed', True, 1), ('failed', True, 0), ('skipped', True, 0),
    ('completed', False, 0),
])
def test_qualifying_trace_requires_completed_attempt_and_saved_trace(tmp_path, status, trace_exists, successes):
    evaluation = tmp_path / 'result.json'
    evaluation.write_text(json.dumps({'pass': True}))
    if trace_exists:
        (tmp_path / 'trace.json').write_text('{}')
    manifest = {'episodes': [{'task': 'task', 'attempts': [{
        'status': status, 'evaluation_path': str(evaluation), 'artifact_path': str(tmp_path),
    }]}]}
    assert task_rank(manifest, 'task', .9)[0] == successes
    assert task_rank(manifest, 'task', .9)[2] == 1


def test_failed_attempt_keeps_diagnostic_score(tmp_path):
    evaluation = tmp_path / 'result.json'
    evaluation.write_text(json.dumps({'pass': False, 'native_result': {'passed': 19, 'total': 20}}))
    (tmp_path / 'trace.json').write_text('{}')
    manifest = {'episodes': [{'task': 'task', 'attempts': [{
        'status': 'failed', 'evaluation_path': str(evaluation), 'artifact_path': str(tmp_path),
    }]}]}
    assert task_rank(manifest, 'task', .9) == (0, .95, 1)


@pytest.mark.parametrize('gym', ['toolathlon_gym', 'toolathlon_bench'])
def test_episode_supervisor_interrupts_process_on_deadline(tmp_path, gym):
    import importlib
    import sys
    import threading

    module = importlib.import_module(f'gyms.{gym}.parallel')
    script = tmp_path / 'episode.py'
    script.write_text('import time\nfrom pathlib import Path\ntry:\n    time.sleep(30)\n'
                      f'except KeyboardInterrupt:\n    Path({str(tmp_path / "stopped")!r}).write_text("yes")\n')
    code, timed_out = module.execute([sys.executable, str(script)], tmp_path / 'logs', .1, threading.Event())
    assert timed_out and code == -1
    assert (tmp_path / 'stopped').read_text() == 'yes'


@pytest.mark.parametrize('gym', ['toolathlon_gym', 'toolathlon_bench'])
def test_toolathlon_cli_has_only_episode_timeout(gym):
    import importlib

    parser = importlib.import_module(f'gyms.{gym}.run').create_parser()
    args = parser.parse_args(['--episode-timeout', '300'])
    assert args.episode_timeout == 300
    assert not hasattr(args, 'agent_timeout')
    with pytest.raises(SystemExit):
        parser.parse_args(['--agent-timeout', '300'])
