"""CPU lifecycle checks against pinned veRL; no model or container launch."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("verl")
from rl.toolathlon_gym import agent_loop as loop


@pytest.fixture
def episode(monkeypatch, tmp_path):
    monkeypatch.setenv("RL_ARTIFACTS", str(tmp_path))
    env = Mock(url="http://worker", runtime={"task_config": {"task_str": "task"}})
    def make_episode(task, directory, **kwargs):
        env.directory = directory
        env.start.side_effect = lambda: directory.mkdir(parents=True)
        return env
    monkeypatch.setattr(loop, "Episode", make_episode)
    env.score.return_value = {"reward": .4}
    tokens = SimpleNamespace(prompt_ids=[1], response_ids=[2], mask=[1], logprobs=[-.1],
                             turns=1, extra={}, calls=[])
    monkeypatch.setattr(loop, "PolicyTokens", lambda *args, **kwargs: tokens)
    monkeypatch.setattr(loop, "VerlChatModel", lambda **kwargs: object())
    monkeypatch.setattr(loop, "create_decomposer_agent", Mock())
    return env


def run():
    instance = SimpleNamespace(tokenizer=object(), server_manager=Mock(),
                               rollout_config=SimpleNamespace(prompt_length=4096, response_length=12288))
    return asyncio.run(loop.ToolathlonAgentLoop.run(instance, {}, extra_info={"task_id": "task"}))


def test_constructor_failure_still_saves_trace_and_cleans_up(monkeypatch, episode):
    monkeypatch.setattr(loop, "create_decomposer_agent", Mock(side_effect=ValueError("constructor")))
    with pytest.raises(ValueError, match="constructor"):
        run()
    episode.close.assert_called_once()
    assert json.loads((episode.directory / "trace.json").read_text())["stop_reason"] == "ValueError"


def test_partial_timeout_is_scored_after_worker_shutdown(monkeypatch, episode):
    capture = AsyncMock(return_value=({"agent_shutdown": [{"status": "interrupted"}]}, TimeoutError()))
    monkeypatch.setattr(loop, "invoke_and_capture", capture)
    result = run()
    assert result.reward_score == .4
    assert capture.call_args.args[-1] == episode.url
    episode.score.assert_called_once()
    episode.close.assert_called_once()


def test_active_worker_prevents_scoring(monkeypatch, episode):
    monkeypatch.setattr(loop, "invoke_and_capture", AsyncMock(return_value=(
        {"agent_shutdown_error": "still running"}, RuntimeError("shutdown"))))
    with pytest.raises(RuntimeError, match="still running"):
        run()
    episode.score.assert_not_called()
    episode.close.assert_called_once()


def test_cancelled_episode_is_not_scored(monkeypatch, episode):
    monkeypatch.setattr(loop, "invoke_and_capture", AsyncMock(return_value=({}, asyncio.CancelledError())))
    with pytest.raises(asyncio.CancelledError):
        run()
    episode.score.assert_not_called()
    episode.close.assert_called_once()
