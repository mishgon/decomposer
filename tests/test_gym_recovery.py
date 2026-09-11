import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, AsyncMock
import httpx
from gyms.toolathlon_gym.episode import Episode
from gyms.toolathlon_gym.cancel import cancel_subagents


class Recovery(unittest.IsolatedAsyncioTestCase):
    def test_port_collision_retry(self):
        ep = object.__new__(Episode)
        ep.container = "owned-task"
        ep.command = Mock(side_effect=[CompletedProcess([], 1, "", "address already in use"),
                                      CompletedProcess([], 0), CompletedProcess([], 0)])
        ep.start_task_container("run", "image")
        self.assertEqual(ep.command.call_count, 3)
        ep.command.assert_any_call("rm", "-f", "owned-task", check=False)

    def test_other_start_failure_is_not_retried(self):
        ep = object.__new__(Episode)
        ep.command = Mock(return_value=CompletedProcess([], 1, "", "image missing"))
        with self.assertRaisesRegex(RuntimeError, "image missing"):
            ep.start_task_container("run", "image")
        self.assertEqual(ep.command.call_count, 1)

    async def test_cancel_missing_finished_run(self):
        client = Mock()
        error = httpx.HTTPStatusError("missing", request=httpx.Request("POST", "http://test"),
                                      response=httpx.Response(404))
        client.runs.cancel = AsyncMock(side_effect=error)
        client.runs.list = AsyncMock(return_value=[])
        await cancel_subagents(client, {"a": {"thread_id": "t", "run_id": "r"}})
        self.assertEqual(client.runs.list.await_count, 2)
        client.runs.list.return_value = [{"status": "running"}]
        with self.assertRaises(httpx.HTTPStatusError):
            await cancel_subagents(client, {"a": {"thread_id": "t", "run_id": "r"}})


if __name__ == "__main__":
    unittest.main()
