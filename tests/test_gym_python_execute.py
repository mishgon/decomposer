import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

from gyms.toolathlon_gym.subagents.python_execute import run_command


class PythonExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_output_and_nonzero_exit(self):
        result = await run_command([sys.executable, "-c", "print('hello'); exit(3)"], "/tmp", 5)
        self.assertIn("hello", result)
        self.assertIn("Return code: 3", result)

    async def test_output_is_bounded(self):
        result = await run_command([sys.executable, "-c", "print('x'*1000000)"], "/tmp", 5)
        self.assertLess(len(result), 8500)
        self.assertIn("truncated", result)

    async def check_descendants(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "escaped"
            ready = Path(directory) / "ready"
            child = f"import time,pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).touch()"
            parent = (f"import subprocess,sys,time,pathlib; subprocess.Popen([sys.executable,'-c',{child!r}]); "
                      f"pathlib.Path({str(ready)!r}).touch(); "
                      + ("time.sleep(30)" if mode != "success" else "pass"))
            task = asyncio.create_task(run_command(
                [sys.executable, "-c", parent], directory, .3 if mode == "timeout" else 5))
            if mode == "cancel":
                for _ in range(100):
                    if ready.exists():
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(ready.exists())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                if mode == "timeout":
                    self.assertIn("TIMEOUT", result)
            await asyncio.sleep(1.2)
            self.assertFalse(marker.exists(), "Child survived tool completion")

    async def test_timeout_kills_descendants(self):
        await self.check_descendants("timeout")

    async def test_cancellation_kills_descendants(self):
        await self.check_descendants("cancel")

    async def test_success_kills_background_descendants(self):
        await self.check_descendants("success")


if __name__ == "__main__":
    unittest.main()
