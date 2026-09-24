"""Metrics for a finished tau2 gym run (`gyms/tau2_gym/run.py`)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals.common import binary_pass_metrics, read_json, read_jsonl
from evals.tau2_gym.analyze_traces import analyse_rollout, summarise
from gyms.tau2_gym.run import validate_result

COMPLETION_MARKER = ".eval_done.json"


def compute(run_dir: Path) -> dict[str, Any]:
    """Pass metrics overall and per domain, plus decomposition statistics.

    The run must be complete. Its rollout grid is re-validated with the gym's own
    check, so the metrics cover exactly the tasks and repeats the run promised.
    """
    marker_path = run_dir / COMPLETION_MARKER
    if not marker_path.is_file():
        raise FileNotFoundError(f"{run_dir} has no {COMPLETION_MARKER}; the run is not complete")
    marker = read_json(marker_path)
    rollouts_path = run_dir / "rollouts.jsonl"
    rows_total = int(marker["dataset_rows"])
    expected_tasks = rows_total if marker.get("limit") is None else min(int(marker["limit"]), rows_total)
    integrity = validate_result(
        rollouts_path, expected_tasks=expected_tasks, num_repeats=int(marker["num_repeats"])
    )

    rows = read_jsonl(rollouts_path)
    rewards_by_task: dict[int, list[float]] = defaultdict(list)
    rewards_by_domain: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        reward = float(row.get("reward", 0.0))
        task = row["_ng_task_index"]
        rewards_by_task[task].append(reward)
        rewards_by_domain[str(row.get("domain"))][task].append(reward)

    records = [analyse_rollout(row) for row in rows]
    scoring = scoring_summary(rows)
    domains = sorted(rewards_by_domain)
    return {
        "gym": "tau2_gym",
        "run_dir": str(run_dir),
        "run_name": marker["run_name"],
        # Runs from before the experiment registry recorded neither.
        "experiment": (marker.get("experiment") or {}).get("name"),
        "pool": marker.get("pool"),
        "num_repeats": marker["num_repeats"],
        "metrics": binary_pass_metrics(rewards_by_task),
        "by_domain": {domain: binary_pass_metrics(rewards_by_domain[domain]) for domain in domains},
        "decomposition": summarise(records),
        "decomposition_by_domain": {
            domain: summarise([record for record in records if str(record["domain"]) == domain])
            for domain in domains
        },
        "scoring": scoring,
        "integrity": integrity,
    }


def scoring_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which scoring path graded the rollouts, and how often the Decomposer under-reported.

    `server_log_v1` rows were scored from the resources server's log of executed calls;
    rows without the tag predate it and were scored from the calls the Decomposer
    reported (`reported_replay_v0`). Rows without a breakdown never reached the verifier.
    """
    paths: Counter[str] = Counter()
    underreported: list[float] = []
    unreported_calls = 0
    for row in rows:
        breakdown = row.get("breakdown") or {}
        if not breakdown:
            paths["not_scored"] += 1
            continue
        paths[breakdown.get("scoring", "reported_replay_v0")] += 1
        missing = int(breakdown.get("unreported_calls") or 0)
        unreported_calls += missing
        if missing:
            underreported.append(float(row.get("reward", 0.0)))
    return {
        "paths": dict(sorted(paths.items())),
        "rollouts_with_unreported_calls": len(underreported),
        "unreported_calls": unreported_calls,
        "pass_at_1_of_rollouts_with_unreported_calls": (
            sum(1 for reward in underreported if reward == 1.0) / len(underreported)
            if underreported else None
        ),
    }
