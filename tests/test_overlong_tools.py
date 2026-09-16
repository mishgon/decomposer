import asyncio
import inspect
from pathlib import Path
import tempfile
import unittest

from langchain_core.tools import tool
from gyms.toolathlon_gym.subagents.webapp import threaded_local_tool


class OverlongAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_disk_operations_run_without_event_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            async def save(content: str) -> str:
                """Save content."""
                with self.assertRaises(RuntimeError):
                    asyncio.get_running_loop()
                path = Path(directory) / "saved"
                path.write_text(content)
                return path.read_text()

            wrapped = threaded_local_tool(save)
            self.assertEqual(inspect.signature(wrapped), inspect.signature(save))
            self.assertFalse(inspect.iscoroutinefunction(wrapped))
            self.assertEqual(await tool(wrapped).ainvoke({"content": "roundtrip"}), "roundtrip")

    async def test_real_await_is_rejected(self):
        async def changed():
            await asyncio.sleep(0)
        with self.assertRaisesRegex(RuntimeError, "unexpectedly awaited"):
            await asyncio.to_thread(threaded_local_tool(changed))


if __name__ == "__main__":
    unittest.main()
