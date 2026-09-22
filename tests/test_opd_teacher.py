import unittest

from training.opd.toolathlon_gym.teacher import aligned_logprobs


class TeacherTests(unittest.TestCase):
    def test_aligned_sampled_tokens(self):
        raw = {'choices': [{'prompt_logprobs': [None, {'2': {'logprob': -0.5}}]}]}
        self.assertEqual(aligned_logprobs(raw, [1, 2]), [0., -0.5])

    def test_fail_closed_on_missing_wrong_or_nan_scores(self):
        for rows in (None, [], [None, {'3': {'logprob': -1}}],
                     [None, {'2': {'logprob': float('nan')}}]):
            with self.assertRaises(ValueError):
                aligned_logprobs({'choices': [{'prompt_logprobs': rows}]}, [1, 2])
