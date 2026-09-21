import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

from scripts.watch_wideseek import display, subagent_counts


class WatchTest(unittest.TestCase):
    def test_peak_counts_only_successful_spawns_and_collected_reports(self):
        messages = []
        def call(name, result):
            cid = str(len(messages))
            messages.extend([{'type': 'ai', 'tool_calls': [{'id': cid, 'name': name}]},
                             {'type': 'tool', 'tool_call_id': cid, 'content': json.dumps(result)}])
        call('spawn_subagent', {'subagent_run_id': 'a'})
        call('spawn_subagent', {'subagent_run_id': 'b'})
        call('wait', 'Timed out')  # No collected reports: neither worker is removed.
        call('spawn_subagent', {'subagent_run_id': 'c'})
        call('wait', [{'subagent_run_id': 'a'}, {'subagent_run_id': 'b'}])
        call('spawn_subagent', {'subagent_run_id': 'd'})
        call('spawn_subagent', {'error': 'failed'})
        self.assertEqual(subagent_counts(messages), (4, 3))

    def test_single_and_legacy_runs(self):
        for modes in (["simple"], ["simple", "decomposer"]):
            with self.subTest(modes=modes), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                run = root / "test"
                run.mkdir()
                (run / "manifest.json").write_text(json.dumps({
                    "started_at": time.time(), "settings": {
                        "tasks": ["task"], "repetitions": 3, "modes": modes,
                        "concurrency": 2, "model": "agent", "judge": {"model": "judge"}}}))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    display(root)
                text = output.getvalue()
                self.assertIn("Setup: simple", text)
                self.assertIn("Mean native score: --", text)
                self.assertIn("0 = 0 returned an answer + 0 stopped early", text)
                self.assertIn("STOPPED / interrupted", text)
                self.assertEqual("Legacy combined run" in text, len(modes) > 1)


if __name__ == "__main__":
    unittest.main()
