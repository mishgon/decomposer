"""Tests for NullJudge stub."""

from __future__ import annotations

from claw_eval.graders.llm_judge import NullJudge


def test_null_judge_returns_zero_without_api():
    judge = NullJudge()
    result = judge.evaluate("task", "conv", "actions", "rubric")
    assert result.score == 0.0
    assert judge.evaluate_actions("task", "artifacts", "rubric").score == 0.0
    assert judge.evaluate_visual("rubric", [], []).score == 0.0
