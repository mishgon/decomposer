import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from gyms.toolathlon_gym.evaluate_sft import run


class Resume(unittest.IsolatedAsyncioTestCase):
    async def test_extend_and_switch_endpoint_without_repeating_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = {"config": {"repetitions": 3, "concurrency": 1}, "tasks": ["task"],
                        "episodes": {"task/rep-001": {"status": "completed", "passed": True}}}
            (output / "manifest.json").write_text(json.dumps(manifest))
            args = argparse.Namespace(output=output, resume=True, repetitions=5, concurrency=None,
                subagent_url="https://router/v1", subagent_model="Qwen/Qwen3.5-4B",
                subagent_host=None, min_free_gb=0, episode_timeout=10)
            call = AsyncMock(return_value={"status": "completed", "passed": False})
            with patch("gyms.toolathlon_gym.evaluate_sft.episode", call), patch(
                "gyms.toolathlon_gym.evaluate_sft.subprocess.check_output", side_effect=["revision", temporary]
            ), patch.dict("os.environ", {"VLLM_API_KEY": "test"}):
                await run(args)
            actual = json.loads((output / "manifest.json").read_text())
            self.assertEqual(call.await_count, 4)
            self.assertEqual(actual["episodes"]["task/rep-001"], manifest["episodes"]["task/rep-001"])
            self.assertEqual(actual["config"]["repetitions"], 5)
            self.assertEqual(actual["resume_history"][0]["previous_config"]["repetitions"], 3)
            self.assertEqual(actual["status"], "completed")
