"""Decomposition and parallelism statistics for Decomposer rollouts.

`scripts/render_gym_trace.py` renders a trace for reading; nothing in the repo
measures the *shape* of a decomposition. This does.

The parallelism measure that matters is spawns per wait batch. `spawn_subagent`
returns immediately and `wait` blocks, so the subagents spawned between two
consecutive `wait` calls are exactly the ones that ran concurrently. A manager
that spawns one subagent and immediately waits is a sequential loop with extra
steps; a manager that spawns several before waiting is using the architecture.

Usage:
    python -m evals.tau2_gym.analyze_traces ROLLOUTS.jsonl [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def analyse_rollout(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response") or {}
    output = response.get("output") or []
    usage = response.get("usage") or {}

    # Map spawn call_id -> subagent_type_id, and run_id -> type, from the
    # call/output pairs (same reconstruction as render_gym_trace.py:71-82).
    calls: dict[str, tuple[str, Any]] = {}
    run_types: dict[str, str] = {}
    for item in output:
        if item.get("type") == "function_call":
            calls[item.get("call_id") or item.get("id")] = (
                item.get("name", "tool"),
                _json(item.get("arguments", {})),
            )
        elif item.get("type") == "function_call_output":
            name, arguments = calls.get(item.get("call_id"), ("", {}))
            result = _json(item.get("output"))
            if name == "spawn_subagent" and isinstance(result, dict):
                run_id = result.get("subagent_run_id")
                if run_id:
                    run_types[run_id] = (
                        arguments.get("subagent_type_id", "unknown")
                        if isinstance(arguments, dict)
                        else "unknown"
                    )

    spawn_batches: list[int] = []
    prompt_lengths: list[int] = []
    pending = 0
    n_spawn = n_wait = n_reports = 0
    spawn_order: list[str] = []
    report_order: list[str] = []

    for item in output:
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        if name == "spawn_subagent":
            n_spawn += 1
            pending += 1
            arguments = _json(item.get("arguments", {}))
            if isinstance(arguments, dict):
                prompt_lengths.append(len(str(arguments.get("prompt", ""))))
        elif name == "wait":
            n_wait += 1
            # Subagents outstanding at this wait ran concurrently.
            spawn_batches.append(pending)
            pending = 0

    if pending:
        spawn_batches.append(pending)

    for item in output:
        if item.get("type") != "function_call_output":
            continue
        name, _ = calls.get(item.get("call_id"), ("", {}))
        result = _json(item.get("output"))
        if name == "spawn_subagent" and isinstance(result, dict):
            run_id = result.get("subagent_run_id")
            if run_id:
                spawn_order.append(run_id)
        elif name == "wait" and isinstance(result, list):
            for report in result:
                if isinstance(report, dict) and report.get("subagent_run_id"):
                    n_reports += 1
                    report_order.append(report["subagent_run_id"])

    batches = [b for b in spawn_batches if b > 0]
    return {
        "task_index": row.get("_ng_task_index"),
        "rollout_index": row.get("_ng_rollout_index"),
        "domain": row.get("domain"),
        "task_id": row.get("task_id"),
        "reward": float(row.get("reward", 0.0)),
        "failure_class": row.get("_ng_failure_class"),
        "n_spawn": n_spawn,
        "n_wait": n_wait,
        "n_reports": n_reports,
        "spawn_batches": batches,
        "max_fanout": max(batches) if batches else 0,
        "mean_fanout": round(statistics.fmean(batches), 2) if batches else 0.0,
        "sequential": bool(batches) and max(batches) == 1,
        "reports_in_spawn_order": report_order == spawn_order[: len(report_order)],
        "subagent_types": dict(Counter(run_types.values())),
        "mean_prompt_chars": round(statistics.fmean(prompt_lengths), 1) if prompt_lengths else 0.0,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    def agg(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.fmean(values), 2),
            "min": min(values),
            "max": max(values),
        }

    fanouts = [r["max_fanout"] for r in records]
    return {
        "rollouts": len(records),
        "pass_rate": round(
            sum(1 for r in records if r["reward"] == 1.0) / len(records), 4
        )
        if records
        else 0.0,
        "spawns_per_rollout": agg([r["n_spawn"] for r in records]),
        "waits_per_rollout": agg([r["n_wait"] for r in records]),
        "max_fanout": agg(fanouts),
        # The headline number: how often the manager actually parallelised.
        "share_fully_sequential": round(
            sum(1 for r in records if r["sequential"]) / len(records), 4
        )
        if records
        else 0.0,
        "share_any_parallel": round(
            sum(1 for r in records if r["max_fanout"] > 1) / len(records), 4
        )
        if records
        else 0.0,
        "mean_prompt_chars": agg([r["mean_prompt_chars"] for r in records]),
        "failure_classes": dict(
            Counter(r["failure_class"] for r in records if r["failure_class"])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--json", dest="json_out", type=Path, default=None)
    args = parser.parse_args(argv)

    records = [
        analyse_rollout(json.loads(line))
        for line in args.rollouts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        print("no rollouts")
        return 1

    overall = summarise(records)
    by_domain = {
        domain: summarise([r for r in records if r["domain"] == domain])
        for domain in sorted({r["domain"] for r in records if r["domain"]})
    }

    print("== overall ==")
    print(json.dumps(overall, indent=2))
    print("\n== per rollout ==")
    header = f"{'domain':<22} {'task':<14} {'rw':>3} {'spawn':>6} {'wait':>5} {'fanout':>7} {'batches'}"
    print(header)
    print("-" * len(header))
    for record in records:
        print(
            f"{str(record['domain'])[:22]:<22} {str(record['task_id'])[:14]:<14} "
            f"{record['reward']:>3.0f} {record['n_spawn']:>6} {record['n_wait']:>5} "
            f"{record['max_fanout']:>7} {record['spawn_batches']}"
        )

    if by_domain:
        print("\n== per domain ==")
        for domain, stats in by_domain.items():
            print(
                f"{domain:<24} pass={stats['pass_rate']:<6} "
                f"spawns={stats['spawns_per_rollout']['mean']:<6} "
                f"max_fanout={stats['max_fanout']['mean']:<5} "
                f"parallel={stats['share_any_parallel']}"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {"overall": overall, "by_domain": by_domain, "rollouts": records},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
