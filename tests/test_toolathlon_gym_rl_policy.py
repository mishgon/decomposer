import os
import unittest
from types import SimpleNamespace


@unittest.skipUnless(os.environ.get("RL_TOKENIZER"), "Set RL_TOKENIZER for real Qwen template tests")
class PolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from transformers import AutoTokenizer
        from training.toolathlon_gym.policy import PolicyTokens
        self.tokenizer = AutoTokenizer.from_pretrained(os.environ["RL_TOKENIZER"])

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
        from training.toolathlon_gym.policy import RolloutBudgetExceeded
        messages = [{"role": "user", "content": "hi"}]
        await self.policy.respond(messages, [])
        original = list(self.policy.response_ids)
        self.policy.response_budget = len(original) + 1
        with self.assertRaises(RolloutBudgetExceeded):
            self.policy.observe(messages + [{"role": "assistant", "content": "done"},
                                           {"role": "user", "content": "again"}], [])
        self.assertEqual(original, self.policy.response_ids)
