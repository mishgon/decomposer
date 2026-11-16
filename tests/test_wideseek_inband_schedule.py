import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sft.wideseek import run


@pytest.mark.parametrize('agent, expected', [
    ('student', [('a', 1), ('a', 2), ('b', 1), ('b', 2)]),
    ('decomposer', [('a', 1), ('b', 1), ('a', 2), ('b', 2)]),
])
def test_student_attempts_start_together_with_bounded_concurrency(tmp_path, monkeypatch, agent, expected):
    args = SimpleNamespace(agent=agent, adaptive=False, n=2, success_threshold=.9,
                           target_successes=4, max_waves=None, mode='decomposer')
    tasks = [{'task_id': 'a'}, {'task_id': 'b'}]
    monkeypatch.setattr(run, 'prepare_collection', AsyncMock(return_value=(tasks, tmp_path)))
    monkeypatch.setattr(run, 'update_index', lambda *args: None)
    order, active = [], 0
    both_started = asyncio.Event()

    async def collect(tasks, jobs, root, settings, **kwargs):
        nonlocal active
        semaphore = asyncio.Semaphore(2)

        async def episode(task, attempt):
            nonlocal active
            async with semaphore:
                order.append((task, attempt))
                active += 1
                if active == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), 1)
                active -= 1

        await asyncio.gather(*(episode(*job) for job in jobs))

    monkeypatch.setattr(run, 'run_jobs', collect)
    asyncio.run(run.main(args))
    assert order == expected
