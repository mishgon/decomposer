import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gyms.infrastructure import classify, failures
from gyms.browsecomp import run


@pytest.mark.parametrize('error', ['TimeoutError()', 'CancelledError()', 'GraphRecursionError()', 'Agent returned an empty response.'])
def test_agent_limits_are_not_service_failures(error):
    assert classify(error) is None


def test_errors_inside_workers_and_tools_are_detected(tmp_path):
    (tmp_path / 'trace.json').write_text(json.dumps({'agent_runs': {
        'a': {'error': 'OpenAIConnectionError: unavailable'}, 'b': {'error': 'TimeoutError()'}}}))
    (tmp_path / 'web_calls.jsonl').write_text(json.dumps({'error': 'HTTPStatusError: 402 Payment Required'}) + '\n{"error":')
    assert {e['kind'] for e in failures(tmp_path)} == {'credentials_or_quota', 'service_unavailable'}


@pytest.mark.parametrize('error,expected', [('OpenAIConnectionError()', 3), ('AuthenticationError: 401', 1)])
def test_queue_stops_and_resume_retries_only_infrastructure(monkeypatch, tmp_path, error, expected):
    data = tmp_path / 'tasks.jsonl'
    data.write_text(json.dumps({'task_id': 'a', 'question': 'Q', 'answer': 'A'}) + '\n')
    args = run.parser().parse_args(['--data', str(data), '--all', '-n', '5', '--concurrency', '1',
                                  '--agent', 'researcher', '--output-dir', str(tmp_path / 'runs')])
    monkeypatch.setattr(run, 'preflight', AsyncMock())
    calls = []
    failing = True
    async def spawn(*command, **kwargs):
        directory = run.Path(command[command.index('--output-dir') + 1])
        calls.append(directory)
        result = {'status': 'finished', 'evaluation': {'status': 'scored', 'score': 0, 'passed': False}}
        run.save(directory / 'result.json', result)
        run.save(directory / 'trace.json', {'messages': [], 'agent_runs': {'worker': {'error': error if failing else None}}})
        return SimpleNamespace(returncode=0, wait=AsyncMock(return_value=0))
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    root = asyncio.run(run.main(args))
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(calls) == expected and manifest['status'] == 'needs_attention'
    assert all(e['infrastructure_errors'] for e in manifest['episodes'])
    failing = False
    args.resume = root
    asyncio.run(run.main(args))
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert len(manifest['episodes']) == 5
    assert len(manifest['archived_attempts']) == expected
    assert all((root / e['directory'] / 'trace.json').exists() for e in manifest['archived_attempts'])
    assert len(calls) == expected + 5


def test_resume_rejects_changed_provenance(monkeypatch, tmp_path):
    data = tmp_path / 'tasks.jsonl'
    data.write_text(json.dumps({'task_id': 'a', 'question': 'Q'}) + '\n')
    args = run.parser().parse_args(['--data', str(data), '--all', '--resume', str(tmp_path), '--agent', 'researcher'])
    run.save(tmp_path / 'manifest.json', {'status': 'interrupted', 'provenance': {'revision': 'old'}})
    monkeypatch.setattr(run, 'preflight', AsyncMock())
    with pytest.raises(ValueError, match='Resume'):
        asyncio.run(run.main(args))


def test_live_auth_failure_stops_current_process(monkeypatch, tmp_path):
    data = tmp_path / 'tasks.jsonl'
    data.write_text(json.dumps({'task_id': 'a', 'question': 'Q'}) + '\n')
    args = run.parser().parse_args(['--data', str(data), '--all', '-n', '4', '--concurrency', '1',
                                  '--agent', 'researcher', '--output-dir', str(tmp_path / 'runs')])
    monkeypatch.setattr(run, 'preflight', AsyncMock())
    processes = []
    async def spawn(*command, **kwargs):
        directory = run.Path(command[command.index('--output-dir') + 1])
        (directory / 'model_calls.jsonl').write_text(json.dumps({'status': 'error', 'error': 'AuthenticationError: 401'}) + '\n')
        process = SimpleNamespace(returncode=None)
        async def wait():
            if process.returncode is None:
                raise TimeoutError()
            return process.returncode
        def interrupt(sig):
            process.returncode = 1
        process.wait = wait
        process.send_signal = interrupt
        processes.append(process)
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    root = asyncio.run(run.main(args))
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(processes) == 1 and processes[0].returncode == 1
    assert manifest['status'] == 'needs_attention'
    assert (root / manifest['episodes'][0]['directory'] / 'model_calls.jsonl').exists()
