import os
import unittest
from types import SimpleNamespace

from opd.policy import parse_tool_calls


class ToolCallTests(unittest.TestCase):
    def test_raw_policy_calls_parse_without_inference_model(self):
        text = 'Plan. <tool_call><function=run><parameter=subagent_id>s1</parameter>' \
               '<parameter=prompt>Compare A &amp; B</parameter><parameter=options>{"n": 2}</parameter>' \
               '</function></tool_call>'
        content, calls = parse_tool_calls(text)
        self.assertEqual(content, 'Plan.')
        self.assertEqual(calls[0]['name'], 'run')
        self.assertEqual(calls[0]['args'], {'subagent_id': 's1', 'prompt': 'Compare A & B', 'options': {'n': 2}})
        self.assertTrue(calls[0]['id'].startswith('call_'))

    def test_multiple_calls_and_plain_answers(self):
        self.assertEqual(parse_tool_calls('done'), ('done', []))
        _, calls = parse_tool_calls('<tool_call><function=wait></function></tool_call>' * 2)
        self.assertEqual([call['name'] for call in calls], ['wait', 'wait'])
        self.assertNotEqual(calls[0]['id'], calls[1]['id'])

    def test_malformed_call_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'Could not parse'):
            parse_tool_calls('<tool_call>broken')


class ToolFreeFinalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_can_finalize_after_tools_are_disabled(self):
        from unittest.mock import AsyncMock
        from langchain.agents import create_agent
        from langchain_core.messages import AIMessage
        from opd.policy import VerlChatModel
        tokens = SimpleNamespace(respond=AsyncMock(return_value=AIMessage(content='Finished')))
        agent = create_agent(VerlChatModel(tokens=tokens), tools=[])
        state = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Summarize the completed work.'}]})
        self.assertEqual(state['messages'][-1].content, 'Finished')
        self.assertEqual(tokens.respond.call_args.args[1], [])


@unittest.skipUnless(os.environ.get("OPD_TOKENIZER"), "Set OPD_TOKENIZER for real Qwen template tests")
class PolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from transformers import AutoTokenizer
        from opd.policy import PolicyTokens
        self.tokenizer = AutoTokenizer.from_pretrained(os.environ["OPD_TOKENIZER"])

        async def generate(ids, params):
            tokens = self.tokenizer.encode("done" + self.tokenizer.eos_token, add_special_tokens=False)
            return SimpleNamespace(token_ids=tokens, log_probs=[-0.1] * len(tokens),
                                   extra_fields={"min_global_steps": 0, "max_global_steps": 0})

        self.policy = PolicyTokens(self.tokenizer, generate, {}, prompt_budget=4096, response_budget=4096)

    async def test_feedback_preserves_actual_policy_prefix_and_loss_masks(self):
        messages = [{"role": "user", "content": "hi"}]
        await self.policy.respond(messages, [])
        prefix = self.policy.prompt_ids + self.policy.response_ids
        messages += [{"role": "assistant", "content": "done"}, {"role": "user", "content": "again"}]
        self.policy.observe(messages, [])
        expected_delta = self.tokenizer.encode("\n", add_special_tokens=False) + self.policy.render(
            [{"role": "user", "content": "again"}], add_generation_prompt=True)
        self.assertEqual(self.policy.prompt_ids + self.policy.response_ids, prefix + expected_delta)
        self.assertEqual(len(self.policy.mask), len(self.policy.response_ids))
        self.assertEqual(len(self.policy.logprobs), len(self.policy.response_ids))
        self.assertIn(0, self.policy.mask)
        self.assertIn(1, self.policy.mask)

    async def test_tool_observation_wrapper(self):
        messages = [{"role": "user", "content": "hi"}]
        await self.policy.respond(messages, [])
        messages += [{"role": "assistant", "content": "done"},
                     {"role": "tool", "content": "report", "tool_call_id": "c1"}]
        self.policy.observe(messages, [])
        self.assertEqual(self.policy.prompt_ids + self.policy.response_ids,
                         self.policy.render(messages, add_generation_prompt=True, tools=[]))

    async def test_prompt_overflow_is_explicit(self):
        self.policy.prompt_budget = 1
        with self.assertRaises(ValueError):
            await self.policy.respond([{"role": "user", "content": "hi"}], [])

    async def test_observation_overflow_preserves_generated_tokens(self):
        from opd.policy import RolloutBudgetExceeded
        messages = [{"role": "user", "content": "hi"}]
        await self.policy.respond(messages, [])
        original = list(self.policy.response_ids)
        self.policy.response_budget = len(original) + 1
        with self.assertRaises(RolloutBudgetExceeded):
            self.policy.observe(messages + [{"role": "assistant", "content": "done"},
                                           {"role": "user", "content": "again"}], [])
        self.assertEqual(original, self.policy.response_ids)


class EpisodeFailureTests(unittest.TestCase):
    def test_shutdown_violation_is_an_episode_failure_in_both_harnesses(self):
        from opd.policy import is_episode_failure
        message = 'Decomposer requested tools instead of completing graceful shutdown'
        for error in (repr(ValueError(message)), f'ValueError: {message}'):
            self.assertTrue(is_episode_failure({'error_type': 'ValueError', 'error': error}))

    def test_infrastructure_and_alignment_errors_remain_fatal(self):
        from opd.policy import is_episode_failure
        for kind, message in [('ValueError', 'Invalid policy token/logprob alignment'),
                              ('ValueError', 'Initial prompt exceeds training prompt budget'),
                              ('RuntimeError', 'Policy inference request timed out'),
                              ('APIConnectionError', 'Connection failed')]:
            self.assertFalse(is_episode_failure({'error_type': kind, 'error': message}))

    def test_existing_budget_failures_remain_episode_failures(self):
        from opd.policy import is_episode_failure
        for kind in ('RolloutBudgetExceeded', 'GraphRecursionError', 'TimeoutError'):
            self.assertTrue(is_episode_failure({'error_type': kind, 'error': kind + ': exhausted'}))


    def test_expected_worker_errors_are_preserved(self):
        from opd.policy import is_episode_failure
        for error in ('GraphRecursionError: exhausted', 'TimeoutError: run budget',
                      'ModelCallLimitExceededError: too many calls',
                      'ValueError: Agent requested tools instead of returning its report'):
            self.assertTrue(is_episode_failure({'error_type': 'WorkerError', 'worker_errors': [error]}))

    def test_worker_infrastructure_errors_override_other_budget_errors(self):
        from opd.policy import is_episode_failure
        for kind in ('WorkerError', 'TimeoutError', 'GraphRecursionError'):
            self.assertFalse(is_episode_failure({'error_type': kind, 'worker_errors': [
                'GraphRecursionError: exhausted', 'APIConnectionError: offline']}))
        self.assertFalse(is_episode_failure({'error_type': 'WorkerError'}))
