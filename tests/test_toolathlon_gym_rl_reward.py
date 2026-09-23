import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from gyms.toolathlon_gym.episode import Episode, native_reward, agent_output_error


class NativeRewardTests(unittest.TestCase):
    def output_exception(self, kind, path):
        return dict(stdout='', stderr='Traceback (most recent call last):\n'
                    f"{kind}: [Errno 21] Is a directory: '{path}'", returncode=1,
                    agent_workspace='/artifacts/data/workspace',
                    groundtruth_workspace='/groundtruth', native_result=None)

    def test_bad_agent_file_is_zero_and_classified(self):
        for kind in ('IsADirectoryError', 'NotADirectoryError', 'FileNotFoundError'):
            result = self.output_exception(kind, '/artifacts/data/workspace/result.xlsx')
            self.assertEqual(native_reward(result), 0)
            self.assertEqual(agent_output_error(result)['type'], kind)

    def test_environment_and_unknown_errors_still_raise(self):
        for kind, path in [('IsADirectoryError', '/groundtruth/result.xlsx'),
                           ('IsADirectoryError', '/artifacts/data/workspace/../groundtruth/result.xlsx'),
                           ('PermissionError', '/artifacts/data/workspace/result.xlsx'),
                           ('IsADirectoryError', '/artifacts/data/workspace-other/result.xlsx')]:
            with self.assertRaises(RuntimeError):
                native_reward(self.output_exception(kind, path))

    def test_nested_groundtruth_is_not_agent_output(self):
        result = self.output_exception('IsADirectoryError', '/artifacts/data/workspace/truth/result.xlsx')
        result['groundtruth_workspace'] = '/artifacts/data/workspace/truth'
        with self.assertRaises(RuntimeError):
            native_reward(result)

    def test_score_continues_after_bad_output_and_saves_real_error(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = object.__new__(Episode)
            episode.directory, episode.container = Path(directory), 'test'
            episode.runtime = {'task_config': {'agent_workspace': '/artifacts/data/workspace',
                'launch_time': None, 'evaluation': {'evaluation_command': 'python evaluator.py',
                                                   'groundtruth_workspace': '/groundtruth'}}}
            bad = self.output_exception('IsADirectoryError', '/artifacts/data/workspace/result.xlsx')
            episode.command = Mock(side_effect=[SimpleNamespace(**bad), SimpleNamespace(stdout='{}')])
            result = episode.score()
            self.assertEqual(result['reward'], 0)
            self.assertEqual(result['outcome'], 'invalid_agent_output')
            self.assertEqual(result['stderr'], bad['stderr'])
            episode.command = Mock(side_effect=[SimpleNamespace(stdout='PASS', stderr='', returncode=0),
                                                SimpleNamespace(stdout='{}')])
            self.assertEqual(episode.score()['reward'], 1)
            episode.command = Mock(side_effect=[SimpleNamespace(stdout='', stderr='Traceback (most recent call last):\nImportError: broken', returncode=1),
                                                SimpleNamespace(stdout='{}')])
            with self.assertRaises(RuntimeError):
                episode.score()
            saved = json.loads((episode.directory / 'evaluation.json').read_text())
            self.assertIsNone(saved['reward'])
            self.assertEqual(saved['outcome'], 'evaluator_error')

    def score(self, stdout="", stderr="", returncode=1, native_result=None):
        return native_reward(dict(stdout=stdout, stderr=stderr, returncode=returncode,
                                  native_result=native_result))

    def test_missing_agent_output_is_zero(self):
        self.assertEqual(self.score("FAIL: Agent output not found: /workspace/test.xlsx\n"), 0)

    def test_missing_groundtruth_is_not_zero(self):
        with self.assertRaises(RuntimeError):
            self.score("FAIL: Groundtruth not found: /groundtruth/test.xlsx\n")

    def test_crash_is_not_zero(self):
        with self.assertRaises(RuntimeError):
            self.score(stderr="Traceback (most recent call last): ...")

    def test_unknown_exit_is_not_zero(self):
        with self.assertRaises(RuntimeError):
            self.score("Something went wrong")

    def test_fraction(self):
        self.assertEqual(self.score(native_result={"passed": 2, "total": 4}), 0.5)

    def test_binary(self):
        self.assertEqual(self.score("=== RESULT: FAIL (2 errors) ==="), 0)
        self.assertEqual(self.score("=== RESULT: PASS ===", returncode=0), 1)

    def test_successful_sheet_lines_do_not_hide_binary_failure(self):
        self.assertEqual(self.score("PASS\nPASS\n=== RESULT: FAIL (2 errors) ==="), 0)

    def test_contradictory_counts_are_not_success(self):
        with self.assertRaises(RuntimeError):
            self.score("2/2 passed")
