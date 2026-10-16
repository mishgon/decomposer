import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from gyms.synth.run import cancel_worker_run


@pytest.mark.parametrize("status", [404, 500])
def test_cancel_finished_worker_race(status):
    response = httpx.Response(status, request=httpx.Request("POST", "http://worker/cancel"))
    error = httpx.HTTPStatusError("cancel failed", request=response.request, response=response)
    client = SimpleNamespace(runs=SimpleNamespace(cancel=AsyncMock(side_effect=error)))
    if status == 404:
        asyncio.run(cancel_worker_run(client, "thread", "run"))
    else:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(cancel_worker_run(client, "thread", "run"))
    client.runs.cancel.assert_awaited_once_with("thread", "run", wait=True, action="interrupt")
