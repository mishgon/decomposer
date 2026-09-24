import unittest
import json
import tempfile
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock, AsyncMock, patch
import httpx
from gyms.toolathlon_gym.episode import Episode
from gyms.toolathlon_gym.cancel import cancel_subagents


class Recovery(unittest.IsolatedAsyncioTestCase):
    def test_podman_command_waits_five_minutes(self):
        ep = object.__new__(Episode)
        ep.engine = "podman"
        with patch("gyms.toolathlon_gym.episode.subprocess.run", return_value=CompletedProcess([], 0)) as run:
            ep.command("ps")
            self.assertEqual(run.call_args.kwargs["timeout"], 300)

    def test_repeated_cleanup_preserves_logs_and_accepts_absent_network(self):
        with tempfile.TemporaryDirectory() as directory:
            ep = object.__new__(Episode)
            ep.directory = Path(directory)
            ep.container, ep.pg, ep.network = "task", "pg", "network"
            saved = ep.directory / "task.log"
            saved.write_text("original failure evidence")
            ep.command = Mock(return_value=CompletedProcess([], 1, "", "not found"))
            ep.close()
            ep.close()
            self.assertEqual(saved.read_text(), "original failure evidence")
            self.assertEqual(json.loads((ep.directory / "cleanup.json").read_text())["errors"], [])
            def command(*args, **kwargs):
                return CompletedProcess([], 0 if args[:2] == ("network", "exists") else 1, "", "still present")
            ep.command = Mock(side_effect=command)
            ep.close()
            self.assertIn("Network cleanup not verified", (ep.directory / "cleanup.json").read_text())

    def test_cleanup_removes_only_owned_volumes_and_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            ep = object.__new__(Episode)
            ep.directory = Path(directory)
            ep.container, ep.pg, ep.network = "owned-task", "owned-pg", "owned-net"
            def command(*args, **kwargs):
                if args[0] == "inspect":
                    return CompletedProcess([], 0, json.dumps([{"Mounts": [
                        {"Type": "volume", "Name": "owned-db"},
                        {"Type": "bind", "Source": "/keep/traces"}]}]), "")
                if "exists" in args: return CompletedProcess([], 1, "", "")
                return CompletedProcess([], 0, "", "")
            ep.command = Mock(side_effect=command)
            ep.close()
            ep.command.assert_any_call("volume", "rm", "owned-db", check=False, timeout=30)
            self.assertFalse(json.loads((ep.directory / "cleanup.json").read_text())["errors"])
            self.assertFalse(any("prune" in c.args for c in ep.command.call_args_list))

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
        await cancel_subagents(client, {"a": {"subagent_id": "s", "run_id": "r"}}, {"s": {"thread_id": "t"}})
        self.assertEqual(client.runs.list.await_count, 2)
        client.runs.list.return_value = [{"status": "running"}]
        with self.assertRaises(httpx.HTTPStatusError):
            await cancel_subagents(client, {"a": {"subagent_id": "s", "run_id": "r"}}, {"s": {"thread_id": "t"}})

    async def test_responded_run_does_not_cancel_reused_worker(self):
        client = Mock()
        client.runs.cancel = AsyncMock()
        await cancel_subagents(client, {
            "old": {"subagent_id": "s", "run_id": "old", "response_sequence_number": 0},
            "new": {"subagent_id": "s", "run_id": "new"},
        }, {"s": {"thread_id": "t"}})
        client.runs.cancel.assert_awaited_once_with("t", "new", wait=True)


if __name__ == "__main__":
    unittest.main()
