import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch

import pandas as pd

from external.wideseek_reward import table_reward
from gyms.wideseek.evaluate import evaluate


class MixedColumnTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_judge_requires_identical_request_and_logs_provenance(self):
        policy = MagicMock()
        response = MagicMock(content='Correct')
        response.model_dump.return_value = {'content': 'Correct'}
        policy.ainvoke = AsyncMock(return_value=response)
        task = {'question': 'Who?', 'answer': 'Alice', 'unique_columns': []}
        with tempfile.TemporaryDirectory() as folder, \
                patch('gyms.wideseek.evaluate.model', return_value=policy), \
                patch('gyms.wideseek.evaluate.close_model', new=AsyncMock()):
            root = Path(folder)
            await evaluate(task, r'\boxed{Alice}', root/'first', judge_model_id='judge')
            policy.ainvoke.reset_mock()
            result = await evaluate(task, r'\boxed{Alice}', root/'replay', judge_model_id='judge',
                                    cached_calls=root/'first/judge_calls')
            self.assertEqual(result['score'], 1)
            policy.ainvoke.assert_not_awaited()
            saved = json.loads(next((root/'replay/judge_calls').glob('*.json')).read_text())
            self.assertIn('replayed_from', saved)
            await evaluate(task, r'\boxed{Bob}', root/'changed', judge_model_id='judge',
                           cached_calls=root/'first/judge_calls')
            policy.ainvoke.assert_awaited_once()

    async def score(self, reference, prediction):
        async def align(responses, targets, judge):
            self.assertNotIn('', targets)
            return {value: value for value in responses if value in targets}

        async def cells(responses, targets, judge):
            return [int(response == target) for response, target in zip(responses, targets)]

        async def judge(messages):
            self.fail('These deterministic regression tests must not call a model')

        with patch.object(table_reward, 'primary_key_preprocess', align), \
                patch.object(table_reward, 'llm_judge_column', cells):
            return await table_reward.evaluate_markdown(
                prediction, {'answer': reference, 'unique_columns': ['id']}, judge)

    async def test_roman_numeral_keys_are_not_erased_or_deduplicated(self):
        reference = pd.DataFrame({'id': ['XLIX', '50', 'LI'], 'winner': ['A', 'B', 'C']})
        self.assertEqual(await self.score(reference.copy(), reference.copy()), (1.0, True))
        # One of three rows gives precision=1, recall=1/3, F1=1/2.
        self.assertEqual(await self.score(reference.copy(), reference.iloc[[1]].copy()), (.5, True))

    async def test_wrong_text_next_to_numeric_title_is_still_rejected(self):
        reference = pd.DataFrame({'id': ['a', 'b'], 'series': ['24', 'Lost']})
        prediction = pd.DataFrame({'id': ['a', 'b'], 'series': ['24', 'Wrong title']})
        self.assertEqual(await self.score(reference, prediction), (.75, True))

    async def test_mixed_cell_text_is_preserved_for_judging(self):
        for text in ['1968-05-09', '1,200', 'circa 1950', '15%', '', 'N/A']:
            with self.subTest(text=text):
                reference = pd.DataFrame({'id': ['a', 'b'], 'value': ['24', text]})
                prediction = pd.DataFrame({'id': ['a', 'b'], 'value': ['24.0', 'different']})
                self.assertEqual(await self.score(reference, prediction), (.75, True))

    async def test_numeric_normalization_and_whitespace_are_unchanged(self):
        reference = pd.DataFrame({'id': [' a ', 'b'], 'value': ['1', '2.50']})
        prediction = pd.DataFrame({'id': ['a', 'b'], 'value': ['1.0', ' 2.5 ']})
        self.assertEqual(await self.score(reference, prediction), (1.0, True))

    async def test_text_only_columns_are_unchanged(self):
        reference = pd.DataFrame({'id': ['a', 'b'], 'value': ['Lost', 'Bond']})
        self.assertEqual(await self.score(reference, reference.copy()), (1.0, True))
