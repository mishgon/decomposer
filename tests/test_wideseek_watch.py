import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

from evals.wideseek.watch import display, subagent_counts


class WatchTest(unittest.TestCase):
    def test_current_harness_agent_run_ids(self):
        messages = [
            {'type': 'ai', 'tool_calls': [{'id': '1', 'name': 'run'}]},
            {'type': 'tool', 'tool_call_id': '1', 'content': '{"agent_run_id": "a"}'},
            {'type': 'ai', 'tool_calls': [{'id': '2', 'name': 'wait'}]},
            {'type': 'tool', 'tool_call_id': '2', 'content': '[{"agent_run_id": "a"}]'},
            {'type': 'ai', 'tool_calls': [{'id': '3', 'name': 'run'}]},
            {'type': 'tool', 'tool_call_id': '3', 'content': '{"agent_run_id": "b"}'},
        ]
        self.assertEqual(subagent_counts(messages), (2, 1))

    def test_peak_counts_only_successful_spawns_and_collected_reports(self):
        messages = []
        def call(name, result):
            cid = str(len(messages))
            messages.extend([{'type': 'ai', 'tool_calls': [{'id': cid, 'name': name}]},
                             {'type': 'tool', 'tool_call_id': cid, 'content': json.dumps(result)}])
        call('new', {'subagent_id': 'worker'})
        call('run', {'subagent_run_id': 'a'})
        call('run', {'subagent_run_id': 'b'})
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

    def test_finds_collection_under_sft_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'sft/wideseek/runs/smoke'
            run.mkdir(parents=True)
            (run / 'manifest.json').write_text(json.dumps({
                'started_at': time.time(), 'settings': {'tasks': ['task'], 'repetitions': 1,
                'modes': ['decomposer'], 'concurrency': 2, 'model': 'teacher',
                'judge': {'model': 'judge'}}}))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                display(root)
            self.assertIn('WIDESEEK  smoke', output.getvalue())

    def test_adaptive_run_is_not_complete_after_one_pass(self):
        from sft.wideseek.scheduler import new_state, plan_next_wave
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "adaptive"
            run.mkdir()
            (run / 'manifest.json').write_text(json.dumps({
                'started_at': time.time(), 'settings': {'tasks': ['task'], 'repetitions': 1,
                'modes': ['decomposer'], 'concurrency': 2, 'model': 'teacher', 'judge': {'model': 'judge'}}}))
            row = {'task_id': 'task', 'attempt': 1, 'mode': 'decomposer', 'status': 'finished',
                   'started_at': time.time()-10, 'finished_at': time.time(), 'execution_directory': 'execution',
                   'evaluation': {'status': 'scored', 'score': .5}}
            path = run / 'decomposer/task/attempt-001'
            path.mkdir(parents=True)
            (path / 'result.json').write_text(json.dumps(row))
            (path / 'execution').mkdir()
            (path / 'execution/trace.json').write_text('{"messages": []}')
            state = new_state(['task'])
            plan_next_wave(state, [row])
            (run / 'scheduler.json').write_text(json.dumps(state))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                display(root)
            text = output.getvalue()
            self.assertNotIn('Status: completed', text)
            self.assertIn('coverage 0/1', text)
            self.assertIn('wave 1: 0/1 ended', text)
            self.assertIn('ETA current wave:', text)
            self.assertIn('later waves depend on scores', text)


if __name__ == "__main__":
    unittest.main()
