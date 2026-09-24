from __future__ import annotations

import pytest
import torch

from opd.loss import OPDLossConfig, opd_token_loss, selected_logprobs


def _inputs(logprobs, behavior, teacher, weight=None, reward=None):
    n = len(logprobs)
    return (
        torch.tensor(logprobs, requires_grad=True),
        torch.tensor(behavior),
        torch.tensor(teacher),
        torch.tensor(weight if weight is not None else [1.0] * n),
        torch.tensor(reward if reward is not None else [0.0] * n),
    )


def test_gradient_raises_tokens_the_teacher_prefers() -> None:
    logprobs, behavior, teacher, weight, reward = _inputs([-1.0, -1.0], [-1.0, -1.0], [-0.5, -3.0])
    loss, metrics = opd_token_loss(logprobs, behavior, teacher, weight, reward, OPDLossConfig())
    loss.backward()
    # d loss / d logp = -A at ratio 1: push up where the teacher is more likely.
    assert logprobs.grad.tolist() == pytest.approx([-0.5, 2.0])
    assert metrics["reverse_kl"] == pytest.approx((-0.5 + 2.0) / 2)
    assert metrics["log_ratio_mean"] == pytest.approx(0.0)


def test_advantage_is_clamped() -> None:
    logprobs, behavior, teacher, weight, reward = _inputs([-1.0], [-1.0], [-40.0])
    loss, metrics = opd_token_loss(logprobs, behavior, teacher, weight, reward, OPDLossConfig(advantage_clamp=4.0))
    loss.backward()
    assert metrics["advantage"] == pytest.approx(-4.0)
    assert logprobs.grad.item() == pytest.approx(4.0)


def test_ratio_clip_stops_the_gradient_outside_the_trust_region() -> None:
    # The policy already moved far towards a positive-advantage token: no further push.
    logprobs, behavior, teacher, weight, reward = _inputs([-0.5], [-1.0], [0.0])
    loss, metrics = opd_token_loss(logprobs, behavior, teacher, weight, reward, OPDLossConfig(ratio_clip=0.2))
    loss.backward()
    assert logprobs.grad.item() == pytest.approx(0.0)
    assert metrics["clip_fraction"] == 1.0


def test_token_weight_and_reward_term() -> None:
    logprobs, behavior, teacher, weight, reward = _inputs([-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [0.25, 0.5], [1.0, 1.0])
    loss, _ = opd_token_loss(logprobs, behavior, teacher, weight, reward, OPDLossConfig(reward_coef=2.0))
    loss.backward()
    assert logprobs.grad.tolist() == pytest.approx([-0.5, -1.0])


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="1-D shape"):
        opd_token_loss(torch.zeros(2), torch.zeros(3), torch.zeros(2), torch.zeros(2), torch.zeros(2), OPDLossConfig())


def test_selected_logprobs_matches_log_softmax_and_backpropagates() -> None:
    torch.manual_seed(0)
    logits = torch.randn(7, 11, requires_grad=True)
    targets = torch.randint(0, 11, (7,))
    expected = torch.log_softmax(logits.detach(), dim=-1)[torch.arange(7), targets]
    chunked = selected_logprobs(logits, targets, chunk_size=3)
    assert torch.allclose(chunked, expected, atol=1e-6)
    chunked.sum().backward()
    reference = logits.detach().clone().requires_grad_(True)
    torch.log_softmax(reference, dim=-1)[torch.arange(7), targets].sum().backward()
    assert torch.allclose(logits.grad, reference.grad, atol=1e-6)


@pytest.mark.parametrize("config", [{"advantage_clamp": 0}, {"ratio_clip": 1.5}])
def test_invalid_config(config: dict) -> None:
    with pytest.raises(ValueError):
        OPDLossConfig(**config)
