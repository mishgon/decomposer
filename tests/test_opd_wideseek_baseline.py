import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from opd.wideseek import baseline


@pytest.mark.parametrize('failure', ['timeout', 'evaluation_error', 'cleanup'])
def test_baseline_timeout_completes_but_infrastructure_errors_fail(tmp_path, monkeypatch, failure):
    def prepare(source, output):
        output.mkdir()
        tasks = [{'task_id': str(i)} for i in range(8)]
        (output / 'tasks.jsonl').write_text('\n'.join(map(json.dumps, tasks)))
        return {'heldout_task_ids': [t['task_id'] for t in tasks], 'source_sha256': 'source'}

    @asynccontextmanager
    async def server(*args, **kwargs):
        yield 'http://unused'

    async def episode(task, mode, attempt, directory, settings, **kwargs):
        assert settings.episode_timeout == 900
        if task['task_id'] == '0' and attempt == 1:
            return {'status': 'timeout' if failure != 'evaluation_error' else 'error',
                    'evaluation': {'score': None},
                    'cleanup_errors': ['worker still running'] if failure == 'cleanup' else []}
        return {'status': 'finished', 'evaluation': {'score': 1.0}}

    monkeypatch.setattr(baseline, 'prepare', prepare)
    monkeypatch.setattr(baseline, 'agent_server', server)
    monkeypatch.setattr(baseline, 'episode', episode)
    monkeypatch.setattr(baseline, 'create_model', lambda name: object())
    close = AsyncMock()
    monkeypatch.setattr(baseline, 'close_model', close)
    monkeypatch.setenv('WS_ARTIFACT_ROOT', '')
    args = SimpleNamespace(model=tmp_path / 'model', output=tmp_path / 'baseline', source=tmp_path / 'source', worker_python='unused')
    if failure == 'timeout':
        asyncio.run(baseline.main(args))
        manifest = json.loads((args.output / 'manifest.json').read_text())
        assert manifest['status'] == 'completed'
        assert len(manifest['episodes']) == 32
        timeout = next(r for r in manifest['episodes'] if r['status'] == 'timeout')
        assert timeout['score'] == 0 and timeout['passed'] is False
        summary = json.loads((args.output / 'summary.json').read_text())
        assert summary['mean_score'] == 31 / 32
    else:
        with pytest.raises(ExceptionGroup):
            asyncio.run(baseline.main(args))
        assert not (args.output / 'summary.json').exists()
    close.assert_awaited_once()
