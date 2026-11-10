import asyncio
import json

from gyms.workplace_assistant.in_process import Workplace


def test_native_tools_are_isolated_and_execution_log_drives_score():
    first, second = Workplace(), Workplace()
    args = {'time_min': '2023-11-23', 'time_max': '2023-11-30',
            'value_to_plot': 'total_visits', 'plot_type': 'line'}
    reference = [{'name': 'analytics_create_plot', 'arguments': json.dumps(args)}]
    tool = next(t for t in first.tools() if t.name == 'analytics_create_plot')
    assert not first.score(reference)
    asyncio.run(tool.ainvoke(args))
    assert first.score(reference)
    assert not second.score(reference)
    assert first.actions == reference


def test_worker_exhaustion_and_infrastructure_failures_keep_distinct_meanings(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gyms.workplace_assistant import in_process
    from opd.policy import is_episode_failure

    gym = SimpleNamespace(actions=[], tools=lambda: [], score=lambda _: False)
    monkeypatch.setattr(in_process, 'Workplace', lambda: gym)
    row = {'id': 1, 'environment_name': 'workplace_assistant', 'prompt': 'Task', 'ground_truth': []}
    for error, expected in [('GraphRecursionError: budget exhausted', True),
                            ('TimeoutError: run budget', True),
                            ('APIConnectionError: offline', False)]:
        workers = SimpleNamespace(records={'run': {'error': error}}, close=AsyncMock())
        monkeypatch.setattr(in_process, 'LocalWorkers', lambda *a, **kw: workers)
        agent = SimpleNamespace(ainvoke=AsyncMock(),
                                aget_state=AsyncMock(return_value=SimpleNamespace(values={'messages': []})))
        monkeypatch.setattr(in_process, 'create_agent', lambda *a, **kw: agent)
        result = asyncio.run(in_process.episode(row, object(), object()))
        assert result['error_type'] == 'WorkerError'
        assert result['worker_errors'] == [error]
        assert result['score'] == 0
        assert is_episode_failure(result) is expected
        workers.close.assert_awaited_once()
