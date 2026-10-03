import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, Mock, patch

from opd.teacher import aligned_logprobs, score


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


class TeacherClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_client_scores_exact_ids_and_closes(self):
        model = Mock(model_name='teacher')
        model.http_async_client.aclose = AsyncMock()
        raw = {'choices': [{'prompt_logprobs': [None, {'2': {'logprob': -.5}}]}]}
        model.root_async_client.completions.create = AsyncMock(return_value=Mock(model_dump=lambda: raw))
        with tempfile.TemporaryDirectory() as directory, patch(
                'opd.teacher.create_model', return_value=model) as registry:
            output = Path(directory) / 'score.json'
            self.assertEqual(await score([1, 2], tokenizer=None, output=output), [0., -.5])
            registry.assert_called_once_with('qwen_3_8_flash_next_non_thinking')
            model.root_async_client.completions.create.assert_awaited_once_with(
                model='teacher', prompt=[1, 2], max_tokens=1, temperature=1.0, extra_body={'prompt_logprobs': 0})
            self.assertEqual(json.loads(output.read_text())['status'], 'completed')
        model.http_client.close.assert_called_once()
        model.http_async_client.aclose.assert_awaited_once()

    async def test_failed_calls_are_saved_without_exception_secrets(self):
        model = Mock(model_name='teacher')
        model.http_async_client.aclose = AsyncMock()
        model.root_async_client.completions.create = AsyncMock(side_effect=RuntimeError('secret'))
        with tempfile.TemporaryDirectory() as directory, patch(
                'opd.teacher.create_model', return_value=model):
            output = Path(directory) / 'score.json'
            with self.assertRaises(RuntimeError):
                await score([1, 2], tokenizer=None, output=output)
            saved = output.read_text()
            self.assertNotIn('secret', saved)
            self.assertEqual(json.loads(saved)['error']['type'], 'RuntimeError')
        model.http_async_client.aclose.assert_awaited_once()
