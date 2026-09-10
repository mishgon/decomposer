import unittest

from gyms.toolathlon_gym.episode import native_reward


class NativeRewardTests(unittest.TestCase):
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
