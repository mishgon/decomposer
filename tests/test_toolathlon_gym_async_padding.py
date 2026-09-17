"""The separated trainer needs padding helpers even when its model uses SDPA."""
import unittest

import torch
from tensordict import TensorDict
from verl.utils.attention_utils import pad_input, unpad_input
from verl.workers.utils.padding import left_right_2_no_padding


class AsyncPaddingTests(unittest.TestCase):
    def test_roundtrip_and_gradients(self):
        x = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2).requires_grad_()
        mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
        values, indices, lengths, maximum, _ = unpad_input(x, mask)
        result = pad_input(values, indices, 2, 3)
        self.assertTrue(torch.equal(result, x * mask.unsqueeze(-1)))
        self.assertEqual(lengths.tolist(), [0, 2, 4])
        self.assertEqual(maximum, 2)
        result.sum().backward()
        self.assertTrue(torch.equal(x.grad, mask.unsqueeze(-1).expand_as(x)))

    def test_actual_trainer_batch_conversion(self):
        batch = TensorDict({
            "input_ids": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]]),
            "position_ids": torch.tensor([[0, 0, 1, 2], [0, 1, 2, 0]]),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
        }, batch_size=[2])
        converted = left_right_2_no_padding(batch)
        self.assertEqual(converted["input_ids"].values().tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(converted["loss_mask"].tolist(), [[1, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
