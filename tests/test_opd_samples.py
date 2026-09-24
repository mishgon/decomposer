from __future__ import annotations

import json
from pathlib import Path

import pytest

from opd.samples import build_samples, episode_samples, read_rollouts, read_samples, write_samples


def _turn_item(prompt: list[int], generated: list[int]) -> dict:
    return {
        "type": "function_call",
        "name": "spawn_subagent",
        "prompt_token_ids": prompt,
        # The Gym vLLM server reports ids as strings before pydantic coerces them.
        "generation_token_ids": [str(token) for token in generated],
        "generation_log_probs": [-0.1 * (index + 1) for index in range(len(generated))],
    }


def _row(turns: list[tuple[list[int], list[int]]], *, reward: float = 1.0, task: tuple[str, str] = ("shop", "t1"),
         task_index: int = 0, rollout_index: int = 0) -> dict:
    output = []
    for prompt, generated in turns:
        output.append({"type": "message", "role": "assistant", "content": []})
        output.append(_turn_item(prompt, generated))
        output.append({"type": "function_call_output", "output": "report"})
    return {"_ng_task_index": task_index, "_ng_rollout_index": rollout_index, "domain": task[0], "task_id": task[1],
            "reward": reward, "response": {"output": output}}


def test_prefix_consistent_turns_merge_into_one_sample() -> None:
    row = _row([([1, 2, 3], [10, 11]), ([1, 2, 3, 10, 11, 50, 51], [12])])
    (sample,) = episode_samples(row, episode_id="e")
    assert sample.input_ids == [1, 2, 3, 10, 11, 50, 51, 12]
    assert sample.spans == [(3, 5), (7, 8)]
    assert sample.generated_positions() == [3, 4, 7]
    assert sample.behavior_logprobs == pytest.approx([-0.1, -0.2, -0.1])
    assert sample.merged_turns == 2


def test_rerendered_history_starts_a_new_sample() -> None:
    # The template re-rendered turn one's tool call (11 -> 99), so turn two is its own sample.
    row = _row([([1, 2, 3], [10, 11]), ([1, 2, 3, 10, 99, 50], [12, 13])])
    first, second = episode_samples(row, episode_id="e")
    assert first.input_ids == [1, 2, 3, 10, 11] and first.spans == [(3, 5)]
    assert second.input_ids == [1, 2, 3, 10, 99, 50, 12, 13] and second.spans == [(6, 8)]


def test_episode_weights_give_mean_over_episodes_of_token_means() -> None:
    rows = [
        ("a", _row([([1, 2], [10, 11]), ([1, 2, 9], [12])], task=("shop", "t1"), reward=1.0)),  # 2 samples, 3 tokens
        ("b", _row([([1, 2], [10, 11, 12, 13])], task=("shop", "t1"), reward=0.0)),  # 1 sample, 4 tokens
    ]
    samples, summary = build_samples(rows, max_length=100)
    assert summary["episodes"] == 2 and summary["samples"] == 3
    # Averaging per-sample losses over samples must weight each episode equally.
    per_episode: dict[str, float] = {}
    for sample in samples:
        per_episode[sample.episode_id] = per_episode.get(sample.episode_id, 0.0) + sample.token_weight * sample.num_generated
    assert {key: value / len(samples) for key, value in per_episode.items()} == pytest.approx({"a": 0.5, "b": 0.5})
    assert {sample.episode_id: sample.reward_advantage for sample in samples} == {"a": 0.5, "b": -0.5}


def test_overlong_samples_and_empty_episodes_are_dropped() -> None:
    rows = [
        ("long", _row([(list(range(50)), [1, 2])])),
        ("empty", {"_ng_task_index": 1, "_ng_rollout_index": 0, "domain": "shop", "task_id": "t2", "reward": 0.0,
                   "response": {"output": [{"type": "message"}]}}),
        ("ok", _row([([1, 2], [3])])),
    ]
    samples, summary = build_samples(rows, max_length=10)
    assert [sample.episode_id for sample in samples] == ["ok"]
    assert summary["dropped"] == {"over_max_length": 1, "episode_without_trainable_turns": 2}


def test_failed_rollouts_can_be_excluded() -> None:
    failed = {**_row([([1, 2], [3])]), "_ng_failure_class": "timeout"}
    samples, summary = build_samples([("f", failed), ("ok", _row([([1], [2])]))], max_length=10, include_failed=False)
    assert [sample.episode_id for sample in samples] == ["ok"]
    assert summary["dropped"]["failed_rollout"] == 1


def test_mismatched_logprobs_are_an_error() -> None:
    row = _row([([1], [2, 3])])
    row["response"]["output"][1]["generation_log_probs"] = [-0.1]
    with pytest.raises(ValueError, match="log-probs"):
        episode_samples(row, episode_id="e")


def test_round_trip_through_files(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps(_row([([1, 2], [3, 4])])) + "\n")
    samples, _ = build_samples(read_rollouts([rollouts]), max_length=10)
    write_samples(samples, tmp_path / "samples.jsonl")
    assert read_samples(tmp_path / "samples.jsonl") == samples
    assert samples[0].episode_id == "0:0:0"
