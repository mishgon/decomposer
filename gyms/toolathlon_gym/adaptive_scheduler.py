from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEDULER_SCHEMA_VERSION = 1
TERMINAL_EPISODE_STATUSES = frozenset({"completed", "failed"})


@dataclass(frozen=True)
class PartialScore:
    passed_checks: int
    total_checks: int
    source: str

    @property
    def fraction(self) -> float:
        return self.passed_checks / self.total_checks


@dataclass(frozen=True)
class LaunchOutcome:
    task: str
    strict_pass: bool
    partial_score: PartialScore | None

    def qualifies(self, threshold: float) -> bool:
        if self.strict_pass:
            return True
        return (
            self.partial_score is not None
            and self.partial_score.fraction > threshold
        )


def _numeric_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def extract_partial_score(evaluation: dict[str, Any]) -> PartialScore | None:
    """Extract native check counts without guessing from arbitrary log lines."""
    native = evaluation.get("native_result")
    if isinstance(native, dict):
        passed = _numeric_count(native.get("total_passed"))
        total = _numeric_count(native.get("total_checks"))
        if passed is not None and total and passed <= total:
            return PartialScore(passed, total, "native_total")

        passed = _numeric_count(native.get("passed"))
        total = _numeric_count(native.get("total"))
        if passed is not None and total and passed <= total:
            return PartialScore(passed, total, "native_total")

        for passed_key, failed_key in (("passed", "failed"), ("pass", "fail")):
            passed = _numeric_count(native.get(passed_key))
            failed = _numeric_count(native.get(failed_key))
            if passed is not None and failed is not None and passed + failed > 0:
                return PartialScore(passed, passed + failed, "native_pass_fail")

    stdout = evaluation.get("stdout")
    if not isinstance(stdout, str):
        return None

    fraction_patterns = (
        r"(?:Results:\s*)?(\d+)\s*/\s*(\d+)\s+passed",
        r"Passed\s+(\d+)\s*/\s*(\d+)\s+checks",
    )
    for pattern in fraction_patterns:
        matches = re.findall(pattern, stdout, flags=re.IGNORECASE)
        if matches:
            passed, total = map(int, matches[-1])
            if total > 0 and passed <= total:
                return PartialScore(passed, total, "stdout_fraction")

    pass_fail_patterns = (
        r"Passed\s*:?\s*(\d+)\s*(?:,|\n)\s*Failed\s*:?\s*(\d+)",
        r"(\d+)\s+passed\s*,\s*(\d+)\s+failed",
    )
    for pattern in pass_fail_patterns:
        matches = re.findall(pattern, stdout, flags=re.IGNORECASE)
        if matches:
            passed, failed = map(int, matches[-1])
            if passed + failed > 0:
                return PartialScore(passed, passed + failed, "stdout_pass_fail")

    # Some native evaluators only emit one line per check. Keep this fallback
    # deliberately narrow: bracketed check markers and standalone status lines
    # are unambiguous, while arbitrary occurrences of words like "error" are not.
    passed = len(re.findall(r"^\s*\[(?:PASS|OK)\]", stdout, re.MULTILINE))
    failed = len(re.findall(r"^\s*\[(?:FAIL|ERROR)\]", stdout, re.MULTILINE))
    if passed + failed > 0:
        return PartialScore(passed, passed + failed, "stdout_check_markers")

    passed = len(re.findall(r"^\s*PASS\s*$", stdout, re.MULTILINE))
    failed = len(re.findall(r"^\s*FAIL\s*$", stdout, re.MULTILINE))
    if passed + failed > 0:
        return PartialScore(passed, passed + failed, "stdout_status_lines")
    return None


def load_launch_outcome(task: str, evaluation_path: str | None) -> LaunchOutcome:
    if not evaluation_path:
        return LaunchOutcome(task, False, None)
    path = Path(evaluation_path)
    try:
        import json

        evaluation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LaunchOutcome(task, False, None)
    return LaunchOutcome(
        task=task,
        strict_pass=evaluation.get("pass") is True,
        partial_score=extract_partial_score(evaluation),
    )


def is_interrupted_attempt(attempt: dict[str, Any]) -> bool:
    error = attempt.get("error")
    return isinstance(error, dict) and error.get("interrupted") is True


def effective_attempts(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Return actual launches, excluding attempts stopped by batch migration."""
    return [
        attempt
        for attempt in episode.get("attempts", ())
        if not is_interrupted_attempt(attempt)
    ]


def task_launches(manifest: dict[str, Any], task: str) -> list[dict[str, Any]]:
    launches: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        if episode["task"] != task:
            continue
        launches.extend(effective_attempts(episode))
    return launches


def task_outcomes(manifest: dict[str, Any], task: str) -> list[LaunchOutcome]:
    return [
        load_launch_outcome(task, attempt.get("evaluation_path"))
        for attempt in task_launches(manifest, task)
    ]


def has_unscored_evaluation(manifest: dict[str, Any], task: str) -> bool:
    for attempt in task_launches(manifest, task):
        path = attempt.get("evaluation_path")
        if path and load_launch_outcome(task, path).partial_score is None:
            return True
    return False


def task_rank(
    manifest: dict[str, Any], task: str, threshold: float
) -> tuple[int, float, int]:
    outcomes = task_outcomes(manifest, task)
    qualifying = sum(outcome.qualifies(threshold) for outcome in outcomes)
    known = [
        outcome.partial_score.fraction
        for outcome in outcomes
        if outcome.partial_score is not None
    ]
    # Prefer keeping tasks with successes, then tasks which came closest. Unknown
    # evaluator formats rank below known partial scores but are never interpreted
    # as a fabricated zero-percent score.
    return qualifying, max(known, default=-1.0), len(outcomes)


def task_summary(
    manifest: dict[str, Any], task: str, threshold: float
) -> dict[str, Any]:
    qualifying, best_partial, launches = task_rank(manifest, task, threshold)
    return {
        "task": task,
        "launches": launches,
        "qualifying_traces": qualifying,
        "best_partial_score": None if best_partial < 0 else best_partial,
    }


def choose_bottom_tasks(
    manifest: dict[str, Any],
    tasks: Sequence[str],
    *,
    threshold: float,
    fraction: float,
) -> list[str]:
    if not 0 <= fraction < 1:
        raise ValueError("cull fraction must be in [0, 1)")
    count = min(len(tasks), math.floor(len(tasks) * fraction))
    zero_success = [
        task
        for task in tasks
        if task_rank(manifest, task, threshold)[0] == 0
        and not has_unscored_evaluation(manifest, task)
    ]
    return sorted(
        zero_success,
        key=lambda task: (task_rank(manifest, task, threshold)[1:], task),
    )[:count]


def append_episodes_to_target(
    manifest: dict[str, Any], tasks: Iterable[str], target_launches: int
) -> list[str]:
    """Append pending repetitions until each task can reach the launch target."""
    added: list[str] = []
    episodes = manifest["episodes"]
    for task in tasks:
        current_launches = len(task_launches(manifest, task))
        pending = sum(
            episode["task"] == task and episode["status"] not in TERMINAL_EPISODE_STATUSES
            for episode in episodes
        )
        needed = max(0, target_launches - current_launches - pending)
        next_repetition = 1 + max(
            (
                episode["repetition"]
                for episode in episodes
                if episode["task"] == task
            ),
            default=0,
        )
        for offset in range(needed):
            repetition = next_repetition + offset
            key = f"{task}::rep-{repetition:03d}"
            episodes.append(
                {
                    "key": key,
                    "task": task,
                    "repetition": repetition,
                    "status": "pending",
                    "score": None,
                    "artifact_path": None,
                    "evaluation_path": None,
                    "started_at": None,
                    "finished_at": None,
                    "duration_seconds": None,
                    "error": None,
                    "attempts": [],
                }
            )
            added.append(key)
    return added


def append_one_episode_per_task(
    manifest: dict[str, Any], tasks: Iterable[str]
) -> list[str]:
    added: list[str] = []
    episodes = manifest["episodes"]
    for task in tasks:
        next_repetition = 1 + max(
            (
                episode["repetition"]
                for episode in episodes
                if episode["task"] == task
            ),
            default=0,
        )
        key = f"{task}::rep-{next_repetition:03d}"
        episodes.append(
            {
                "key": key,
                "task": task,
                "repetition": next_repetition,
                "status": "pending",
                "score": None,
                "artifact_path": None,
                "evaluation_path": None,
                "started_at": None,
                "finished_at": None,
                "duration_seconds": None,
                "error": None,
                "attempts": [],
            }
        )
        added.append(key)
    return added


def new_scheduler_state(
    tasks: Sequence[str],
    *,
    threshold: float,
    cull_fraction: float,
    target_successes: int,
) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("success threshold must be in [0, 1]")
    if target_successes < 1:
        raise ValueError("target successes must be positive")
    return {
        "schema_version": SCHEDULER_SCHEMA_VERSION,
        "policy": "coverage_first",
        "phase": "coverage_first",
        "success_threshold": threshold,
        "cull_fraction": cull_fraction,
        "target_successes": target_successes,
        "max_zero_success_launches": 6,
        "original_tasks": list(tasks),
        "active_tasks": list(tasks),
        "culled_tasks": [],
        "rounds": [],
    }


def migrate_to_coverage_first(manifest: dict[str, Any]) -> int:
    """Discard only unstarted legacy queue entries and enable water-filling."""
    state = manifest["adaptive_scheduler"]
    if state.get("policy") == "coverage_first":
        return 0

    removed_keys = {
        episode["key"]
        for episode in manifest["episodes"]
        if episode["status"] == "pending" and not episode.get("attempts")
    }
    manifest["episodes"] = [
        episode
        for episode in manifest["episodes"]
        if episode["key"] not in removed_keys
    ]
    retained_keys = {episode["key"] for episode in manifest["episodes"]}
    for round_ in state.get("rounds", []):
        round_["episode_keys"] = [
            key for key in round_.get("episode_keys", []) if key in retained_keys
        ]
    state.update(
        policy="coverage_first",
        phase="coverage_first",
        max_zero_success_launches=6,
    )
    state.setdefault("migrations", []).append(
        {
            "from_policy": "three_plus_three",
            "removed_unstarted_episodes": len(removed_keys),
        }
    )
    return len(removed_keys)


def phase_is_terminal(manifest: dict[str, Any], keys: Sequence[str]) -> bool:
    by_key = {episode["key"]: episode for episode in manifest["episodes"]}
    return bool(keys) and all(
        key in by_key and by_key[key]["status"] in TERMINAL_EPISODE_STATUSES
        for key in keys
    )


def plan_next_wave(manifest: dict[str, Any]) -> list[str]:
    """Prioritize breadth, then water-fill retained tasks to the target count."""
    state = manifest["adaptive_scheduler"]
    threshold = state["success_threshold"]
    active = state["active_tasks"]
    rounds = state["rounds"]

    if rounds and rounds[-1]["episode_keys"] and not phase_is_terminal(
        manifest, rounds[-1]["episode_keys"]
    ):
        return []

    phase = state["phase"]
    if phase == "coverage_first":
        exhausted = [
            task
            for task in active
            if task_rank(manifest, task, threshold)[0] == 0
            and task_rank(manifest, task, threshold)[2]
            >= state["max_zero_success_launches"]
            and not has_unscored_evaluation(manifest, task)
        ]
        if exhausted:
            state["culled_tasks"].extend(
                {
                    **task_summary(manifest, task, threshold),
                    "after_launches": state["max_zero_success_launches"],
                    "reason": "zero_success_after_launch_limit",
                }
                for task in exhausted
            )
            state["active_tasks"] = [
                task for task in active if task not in exhausted
            ]
            active = state["active_tasks"]

        uncovered = [
            task for task in active if task_rank(manifest, task, threshold)[0] == 0
        ]
        if uncovered:
            added = append_one_episode_per_task(manifest, uncovered)
            rounds.append(
                {
                    "phase": phase,
                    "target_successes": 1,
                    "episode_keys": added,
                }
            )
            return added
        state["phase"] = "balance_successes"
        return plan_next_wave(manifest)

    if phase == "initial_three":
        added = append_episodes_to_target(manifest, active, 3)
        if added:
            rounds.append(
                {"phase": phase, "target_launches": 3, "episode_keys": added}
            )
        state["phase"] = "cull_after_three"
        return added or plan_next_wave(manifest)

    if phase == "cull_after_three":
        culled = choose_bottom_tasks(
            manifest,
            active,
            threshold=threshold,
            fraction=state["cull_fraction"],
        )
        state["culled_tasks"].extend(
            {
                **task_summary(manifest, task, threshold),
                "after_launches": 3,
                "reason": "bottom_fraction_after_three",
            }
            for task in culled
        )
        state["active_tasks"] = [task for task in active if task not in culled]
        state["phase"] = "six_total"
        return plan_next_wave(manifest)

    if phase == "six_total":
        added = append_episodes_to_target(manifest, active, 6)
        if added:
            rounds.append(
                {"phase": phase, "target_launches": 6, "episode_keys": added}
            )
        state["phase"] = "drop_unsolved_after_six"
        return added or plan_next_wave(manifest)

    if phase == "drop_unsolved_after_six":
        unsolved = [
            task
            for task in active
            if task_rank(manifest, task, threshold)[0] == 0
            and not has_unscored_evaluation(manifest, task)
        ]
        state["culled_tasks"].extend(
            {
                **task_summary(manifest, task, threshold),
                "after_launches": 6,
                "reason": "zero_success_after_six",
            }
            for task in unsolved
        )
        state["active_tasks"] = [task for task in active if task not in unsolved]
        state["phase"] = "balance_successes"
        return plan_next_wave(manifest)

    if phase == "balance_successes":
        successes = {
            task: task_rank(manifest, task, threshold)[0] for task in active
        }
        if not successes or min(successes.values()) >= state["target_successes"]:
            state["phase"] = "complete"
            return []
        next_target = min(successes.values()) + 1
        deficient = [task for task in active if successes[task] < next_target]
        added = append_one_episode_per_task(manifest, deficient)
        rounds.append(
            {
                "phase": phase,
                "target_successes": next_target,
                "episode_keys": added,
            }
        )
        return added

    if phase == "complete":
        return []
    raise ValueError(f"Unknown adaptive scheduler phase: {phase!r}")
