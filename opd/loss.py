"""On-policy distillation loss: sampled-token reverse KL as a policy gradient.

For a student token y_t sampled from the behaviour policy b (the round's served
checkpoint), the per-token reverse-KL estimate is log b(y_t) - log T(y_t). Its
negative is the token's reward, so the advantage is

    A_t = clamp(log T(y_t) - log b(y_t), -c, c)        (detached)

and the student maximises A_t * log pi(y_t) through a PPO-clipped ratio
r_t = pi(y_t) / b(y_t). The ratio uses the behaviour log-probs vLLM reported while
sampling, so it also absorbs the vLLM/trainer numeric mismatch (truncated
importance sampling); `log_ratio_mean` at the start of a round measures that
mismatch. Same objective family as tau2-gym's `online_pg_opd_k1`
(scripts/online_opd/launch.py:126-157), where clamp 4 is the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class OPDLossConfig:
    advantage_clamp: float = 4.0
    ratio_clip: float = 0.2
    # Weight of a task-reward advantage added to the distillation advantage
    # (reward minus the task's mean reward in the round). 0 is pure distillation.
    reward_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.advantage_clamp <= 0:
            raise ValueError("advantage_clamp must be positive")
        if not 0 < self.ratio_clip < 1:
            raise ValueError("ratio_clip must be in (0, 1)")


def _chunk_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return logits.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def selected_logprobs(logits: torch.Tensor, targets: torch.Tensor, *, chunk_size: int = 2048) -> torch.Tensor:
    """log softmax(logits)[target] per row, in fp32, without holding [K, V] fp32 at once.

    Each chunk is checkpointed, so backward recomputes its fp32 log-sum-exp instead of
    keeping it: with a 248k vocabulary a single fp32 row block of 2048 tokens is 2 GB.
    """
    if logits.ndim != 2 or targets.shape != logits.shape[:1]:
        raise ValueError(f"expected logits [K, V] and targets [K], got {tuple(logits.shape)} / {tuple(targets.shape)}")
    pieces = []
    for start in range(0, logits.shape[0], chunk_size):
        piece = logits[start : start + chunk_size]
        target = targets[start : start + chunk_size]
        if piece.requires_grad:
            pieces.append(checkpoint(_chunk_logprobs, piece, target, use_reentrant=False))
        else:
            pieces.append(_chunk_logprobs(piece, target))
    return torch.cat(pieces) if pieces else logits.new_zeros(0, dtype=torch.float32)


def opd_token_loss(
    logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    token_weight: torch.Tensor,
    reward_advantage: torch.Tensor,
    config: OPDLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of per-token clipped policy-gradient losses, and diagnostics.

    All inputs are 1-D over the trained tokens. `token_weight` carries the episode
    normalisation built by samples.py, so the sum is already the sample's share of
    the round's mean-over-episodes, token-mean-within-episode objective.
    """
    shapes = {tensor.shape for tensor in (logprobs, behavior_logprobs, teacher_logprobs, token_weight, reward_advantage)}
    if len(shapes) != 1 or logprobs.ndim != 1:
        raise ValueError(f"all loss inputs must share one 1-D shape, got {shapes}")
    behavior = behavior_logprobs.float()
    advantage = (teacher_logprobs.float() - behavior).clamp(-config.advantage_clamp, config.advantage_clamp)
    if config.reward_coef:
        advantage = advantage + config.reward_coef * reward_advantage.float()
    advantage = advantage.detach()

    log_ratio = logprobs.float() - behavior
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - config.ratio_clip, 1 + config.ratio_clip) * advantage
    per_token = -torch.minimum(unclipped, clipped)
    loss = (per_token * token_weight.float()).sum()

    with torch.no_grad():
        count = max(int(logprobs.numel()), 1)
        metrics = {
            "tokens": float(logprobs.numel()),
            "reverse_kl": float((behavior - teacher_logprobs.float()).sum()) / count,
            "advantage": float(advantage.sum()) / count,
            "log_ratio_mean": float(log_ratio.sum()) / count,
            "clip_fraction": float(((ratio - 1).abs() > config.ratio_clip).float().sum()) / count,
        }
    return loss, metrics
