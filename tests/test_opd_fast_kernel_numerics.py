"""Opt-in GPU check of the restored FLA operator against FP32 recurrence math."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('OPD_TEST_FAST_KERNELS') != '1',
                                reason='Requires the restored kernel bundle and an allocated GPU')


@pytest.mark.parametrize('length', [128, 1024])
def test_fla_matches_fp32_recurrence(length):
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    torch.manual_seed(42)
    shape = (1, length, 4, 128)
    values = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    values += [-torch.rand((1, length, 4), device='cuda'),
               torch.rand((1, length, 4), device='cuda', dtype=torch.bfloat16)]
    values = [value.requires_grad_() for value in values]
    probe = torch.randn(shape, device='cuda', dtype=torch.float32)
    with torch.autocast('cuda', enabled=False):
        reference, _ = torch_chunk_gated_delta_rule(*values, use_qk_l2norm_in_kernel=True)
        gradients = torch.autograd.grad((reference.float() * probe).sum(), values)
        actual, _ = chunk_gated_delta_rule(*values, use_qk_l2norm_in_kernel=True)
        actual_gradients = torch.autograd.grad((actual.float() * probe).sum(), values)
    relative_output_error = ((actual.float() - reference.float()).norm() / reference.float().norm()).item()
    assert relative_output_error < .02
    for expected, found in zip(gradients, actual_gradients, strict=True):
        expected, found = expected.float().flatten(), found.float().flatten()
        cosine = torch.nn.functional.cosine_similarity(expected, found, dim=0).item()
        relative_error = ((expected - found).norm() / expected.norm()).item()
        print({'length': length, 'output_relative_error': relative_output_error,
               'gradient_cosine': cosine, 'gradient_relative_error': relative_error})
        assert cosine > .999 and relative_error < .05
