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


def test_widesearch_preserves_connection_error_type(monkeypatch, tmp_path):
    import httpx
    from gyms.widesearch import tools
    monkeypatch.setenv('BROWSECOMP_MODEL_LOG', str(tmp_path / 'model_calls.jsonl'))
    async def failing_search():
        raise httpx.ConnectError('All connection attempts failed')
    asyncio.run(tools.with_errors(failing_search)())
    assert failures(tmp_path)[0]['kind'] == 'service_unavailable'


def test_judge_outage_pauses_without_invalidating_answers(monkeypatch, tmp_path):
    data = tmp_path / 'tasks.jsonl'
    data.write_text(json.dumps({'task_id': 'a', 'question': 'Q'}) + '\n')
    args = run.parser().parse_args(['--data', str(data), '--all', '-n', '5', '--concurrency', '1',
                                  '--agent', 'researcher', '--output-dir', str(tmp_path / 'runs')])
    monkeypatch.setattr(run, 'preflight', AsyncMock())
    async def spawn(*command, **kwargs):
        directory = run.Path(command[command.index('--output-dir') + 1])
        run.save(directory / 'result.json', {'status': 'finished', 'evaluation': {
            'status': 'evaluation_error', 'error': 'APIConnectionError()', 'score': None, 'passed': False}})
        return SimpleNamespace(returncode=0, wait=AsyncMock(return_value=0))
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    root = asyncio.run(run.main(args))
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_attention' and len(manifest['episodes']) == 3
    assert all(not e.get('infrastructure_errors') for e in manifest['episodes'])


def test_judge_retry_normalizes_content_blocks(monkeypatch, tmp_path):
    from evals.browsecomp import run as evaluation
    directory = tmp_path / 'traces/a-r001'
    directory.mkdir(parents=True)
    run.save(directory / 'trace.json', {'messages': [{'type': 'ai', 'content': [{'type': 'text', 'text': 'answer'}]}]})
    run.save(directory / 'task.json', {'task_id': 'a', 'question': 'Q'})
    run.save(tmp_path / 'manifest.json', {'status': 'needs_attention', 'tasks': ['a'], 'repetitions': 1,
        'episodes': [{'task': 'a', 'repetition': 1, 'directory': 'traces/a-r001', 'status': 'finished',
                      'evaluation': {'status': 'evaluation_error', 'score': None, 'passed': False}}]})
    monkeypatch.setattr(evaluation, 'run_episodes', AsyncMock(return_value=tmp_path))
    judge = AsyncMock(return_value={'status': 'scored', 'score': 1, 'passed': True})
    monkeypatch.setattr(evaluation, 'evaluate', judge)
    asyncio.run(evaluation.main(SimpleNamespace(resume=tmp_path)))
    assert judge.call_args.args[1] == 'answer'
    assert json.loads((tmp_path / 'manifest.json').read_text())['status'] == 'completed'


def test_exhausted_quota_stops_before_transient_rate_limits():
    assert classify("RateLimitError: 429 insufficient_quota") == "credentials_or_quota"
    assert classify("RateLimitError: 429 too many requests") == "service_unavailable"
