"""Turn Gym Decomposer rollouts into OPD training samples.

With `vllm_model.return_token_id_information` on, the Gym model server stamps the
last output item of every manager turn with that turn's exact `prompt_token_ids`,
`generation_token_ids` and `generation_log_probs`
(nemo_gym/responses_converter.py:433-445), and the Decomposer agent replays those
items verbatim into `rollouts.jsonl` (decomposer_agent/app.py:710-760). So every
manager turn is available as (what the student saw, what it sampled, the behaviour
log-probs) without re-tokenising anything.

A sample is one token sequence with one or more generated spans. Consecutive turns
merge when turn k+1's prompt starts with turn k's prompt + generation; otherwise the
chat template re-rendered history (tool-call arguments, reasoning stripping) and the
turn becomes its own sample. Only generated manager tokens are trained; subagent
reports, tool outputs and harness nudges are context.

    python -m training.opd.samples ROUND/rollouts/rollouts.jsonl --output ROUND/samples.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TOKEN_FIELDS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")


@dataclass
class Sample:
    sample_id: str
    episode_id: str
    domain: str
    task_id: str
    reward: float
    input_ids: list[int]
    # Half-open [start, end) index ranges of generated tokens in input_ids.
    spans: list[tuple[int, int]]
    # Behaviour log-probs of the generated tokens, concatenated in span order.
    behavior_logprobs: list[float]
    reward_advantage: float = 0.0
    token_weight: float = 0.0
    merged_turns: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_generated(self) -> int:
        return sum(end - start for start, end in self.spans)

    def generated_positions(self) -> list[int]:
        return [index for start, end in self.spans for index in range(start, end)]


@dataclass(frozen=True)
class Turn:
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    generation_log_probs: list[float]


def manager_turns(row: dict[str, Any]) -> list[Turn]:
    """The token-carrying manager turns of one rollout row, in order."""
    turns = []
    for item in (row.get("response") or {}).get("output") or []:
        if not all(name in item for name in TOKEN_FIELDS):
            continue
        prompt = [int(token) for token in item["prompt_token_ids"]]
        generated = [int(token) for token in item["generation_token_ids"]]
        logprobs = [float(value) for value in item["generation_log_probs"]]
        if len(generated) != len(logprobs):
            raise ValueError(
                f"turn has {len(generated)} generated tokens but {len(logprobs)} log-probs"
            )
        turns.append(Turn(prompt, generated, logprobs))
    return turns


def episode_samples(row: dict[str, Any], *, episode_id: str) -> list[Sample]:
    """Merge prefix-consistent turns of one episode into samples."""
    samples: list[Sample] = []
    current: Sample | None = None
    for turn in manager_turns(row):
        if not turn.generation_token_ids:
            continue
        start = len(turn.prompt_token_ids)
        span = (start, start + len(turn.generation_token_ids))
        if current is not None and turn.prompt_token_ids[: len(current.input_ids)] == current.input_ids:
            current.input_ids = turn.prompt_token_ids + turn.generation_token_ids
            current.spans.append(span)
            current.behavior_logprobs.extend(turn.generation_log_probs)
            current.merged_turns += 1
            continue
        current = Sample(
            sample_id=f"{episode_id}/{len(samples)}",
            episode_id=episode_id,
            domain=str(row.get("domain")),
            task_id=str(row.get("task_id")),
            reward=float(row.get("reward", 0.0)),
            input_ids=turn.prompt_token_ids + turn.generation_token_ids,
            spans=[span],
            behavior_logprobs=list(turn.generation_log_probs),
        )
        samples.append(current)
    return samples


def read_rollouts(paths: Iterable[Path]) -> Iterator[tuple[str, dict[str, Any]]]:
    """(episode id, row) pairs; the id is file position, task index and rollout index."""
    for file_index, path in enumerate(paths):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                yield f"{file_index}:{row['_ng_task_index']}:{row['_ng_rollout_index']}", row


def build_samples(
    rollouts: Iterable[tuple[str, dict[str, Any]]],
    *,
    max_length: int,
    include_failed: bool = True,
) -> tuple[list[Sample], dict[str, Any]]:
    """Samples with episode weights and reward advantages, plus a summary."""
    dropped: collections.Counter[str] = collections.Counter()
    episodes: dict[str, list[Sample]] = {}
    rewards_by_task: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    turns = 0
    for episode_id, row in rollouts:
        if row.get("_ng_failure_class") and not include_failed:
            dropped["failed_rollout"] += 1
            continue
        samples = episode_samples(row, episode_id=episode_id)
        turns += sum(sample.merged_turns for sample in samples)
        kept = []
        for sample in samples:
            if len(sample.input_ids) > max_length:
                dropped["over_max_length"] += 1
                continue
            kept.append(sample)
        if not kept:
            dropped["episode_without_trainable_turns"] += 1
            continue
        episodes[episode_id] = kept
        rewards_by_task[(kept[0].domain, kept[0].task_id)].append(kept[0].reward)

    all_samples = [sample for samples in episodes.values() for sample in samples]
    if not all_samples:
        raise ValueError(f"no trainable samples; dropped: {dict(dropped)}")
    # Mean over episodes of the token mean within each episode. Averaging the per-sample
    # loss over samples (what the trainer does) then recovers exactly that objective.
    scale = len(all_samples) / len(episodes)
    for samples in episodes.values():
        episode_tokens = sum(sample.num_generated for sample in samples)
        for sample in samples:
            sample.token_weight = scale / episode_tokens
            task_rewards = rewards_by_task[(sample.domain, sample.task_id)]
            sample.reward_advantage = sample.reward - sum(task_rewards) / len(task_rewards)

    generated = [sample.num_generated for sample in all_samples]
    lengths = [len(sample.input_ids) for sample in all_samples]
    summary = {
        "episodes": len(episodes),
        "turns": turns,
        "samples": len(all_samples),
        "merged_turn_fraction": 1 - len(all_samples) / max(turns, 1),
        "generated_tokens": sum(generated),
        "max_sample_length": max(lengths),
        "mean_sample_length": sum(lengths) / len(lengths),
        "mean_reward": sum(samples[0].reward for samples in episodes.values()) / len(episodes),
        "dropped": dict(dropped),
        "max_length": max_length,
    }
    return all_samples, summary


def write_samples(samples: Sequence[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample)) + "\n")
    tmp.replace(path)


def read_samples(path: Path) -> list[Sample]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            value["spans"] = [tuple(span) for span in value["spans"]]
            samples.append(Sample(**value))
    return samples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--exclude-failed", action="store_true")
    args = parser.parse_args(argv)
    samples, summary = build_samples(
        read_rollouts(args.rollouts), max_length=args.max_length, include_failed=not args.exclude_failed
    )
    write_samples(samples, args.output)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
