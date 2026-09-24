import unittest
from unittest.mock import Mock

from opd.toolathlon_gym.teacher import aligned_logprobs


class TeacherTests(unittest.TestCase):
    def test_incremental_unicode_decoding(self):
        tokenizer = Mock()
        tokenizer.decode.return_value = '📧'
        raw = {'choices': [{'prompt_logprobs': [None,
            {'2': {'logprob': -.5, 'decoded_token': ''}},
            {'3': {'logprob': -.2, 'decoded_token': '📧'}}]}]}
        self.assertEqual(aligned_logprobs(raw, [1, 2, 3], tokenizer), [0., -.5, -.2])

    def test_token_meaning_is_checked_not_just_numeric_id(self):
        tokenizer = Mock()
        tokenizer.decode.return_value = ' student'
        raw = {'choices': [{'prompt_logprobs': [None, {
            '2': {'logprob': -.5, 'decoded_token': ' teacher'}}]}]}
        with self.assertRaisesRegex(ValueError, 'meaning mismatch'):
            aligned_logprobs(raw, [1, 2], tokenizer)
        raw['choices'][0]['prompt_logprobs'][1]['2']['decoded_token'] = ' student'
        self.assertEqual(aligned_logprobs(raw, [1, 2], tokenizer), [0., -.5])

    def test_aligned_sampled_tokens(self):
        raw = {'choices': [{'prompt_logprobs': [None, {'2': {'logprob': -0.5}}]}]}
        self.assertEqual(aligned_logprobs(raw, [1, 2]), [0., -0.5])

    def test_fail_closed_on_missing_wrong_or_nan_scores(self):
        for rows in (None, [], [None, {'3': {'logprob': -1}}],
                     [None, {'2': {'logprob': float('nan')}}]):
            with self.assertRaises(ValueError):
                aligned_logprobs({'choices': [{'prompt_logprobs': rows}]}, [1, 2])
