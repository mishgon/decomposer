"""Metrics for a finished GAIA2 evaluation run (`gyms/gaia2/run.py`)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.common import binary_pass_metrics, read_json, read_jsonl
from gyms.gaia2.run import validate_result

COMPLETION_MARKER = ".eval_done.json"


def compute(run_dir: Path) -> dict[str, Any]:
    """Pass metrics for a complete evaluation, next to the runner's own summary.

    The rollouts are re-validated against the selected tasks, so the metrics
    cover exactly the scenarios and repeats the run promised. The runner's
    summary (`gym_metrics`) keeps the fixed-denominator score, judge decisions
    and exception counts.
    """
    marker_path = run_dir / COMPLETION_MARKER
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"{run_dir} has no {COMPLETION_MARKER}; the run is not complete"
        )
    marker = read_json(marker_path)
    if marker.get("state") != "complete":
        raise ValueError(
            f"{marker_path} is not a completed run (state {marker.get('state')!r})"
        )
    domain = marker["domain"]
    scenario_ids = marker["scenario_ids"]
    integrity = validate_result(
        run_dir,
        num_repeats=int(marker["num_repeats"]),
        limit=marker.get("limit"),
        scenario_count=len(scenario_ids),
        scenario_ids=scenario_ids,
    )
    rewards_by_task: dict[str, list[float]] = defaultdict(list)
    for row in read_jsonl(run_dir / "output.jsonl"):
        rewards_by_task[str(row.get("task_id"))].append(float(row.get("score") or 0.0))
    return {
        "gym": "gaia2",
        "run_dir": str(run_dir),
        "agent": marker.get("agent")
        or (
            "decomposer"
            if marker["harness"] == "decomposer"
            else "qwen_3_5_4b_thinking"
        ),
        "decomposer_model_id": marker.get(
            "decomposer_model_id", marker.get("model_id")
        ),
        "agent_model_id": marker.get("agent_model_id"),
        "domain": domain,
        "num_repeats": marker["num_repeats"],
        "limit": marker.get("limit"),
        "metrics": binary_pass_metrics(rewards_by_task),
        "gym_metrics": integrity,
    }


def compute_suite(run_dir: Path) -> dict[str, Any]:
    configuration = read_json(run_dir / "evaluation.json")
    domains = {domain: compute(run_dir / domain) for domain in configuration["domains"]}
    rewards_by_task = defaultdict(list)
    for domain in domains:
        for row in read_jsonl(run_dir / domain / "output.jsonl"):
            rewards_by_task[(domain, row["task_id"])].append(
                float(row.get("score") or 0.0)
            )
    return {
        "gym": "gaia2",
        "run_dir": str(run_dir),
        "agent": configuration["agent"],
        "domains": domains,
        "metrics": binary_pass_metrics(rewards_by_task),
        "errors": sum(
            sum(result["gym_metrics"]["exception_types"].values())
            for result in domains.values()
        ),
    }
