"""Decomposition and parallelism statistics for Decomposer rollouts.

`scripts/render_gym_trace.py` renders a trace for reading; nothing in the repo
measures the *shape* of a decomposition. This does.

The manager creates subagents with `new` (or copies one with `fork`), starts work
on them with `run`, and collects responses with `wait`. `run` returns immediately
and `wait` blocks, so the runs started between two consecutive `wait` calls are
exactly the ones that ran concurrently: runs per wait batch is the parallelism
measure. A manager that runs one subagent and immediately waits is a sequential
loop with extra steps; one that starts several runs before waiting is using the
architecture. Subagents persist, so a run can also continue a subagent that ran
before (`reused_runs`), keeping its conversation.

Rollouts from the earlier core used `spawn_subagent`, a `new` and a `run` in one
call; they are counted that way, so old runs stay comparable on runs and fan-out.

Usage:
    python -m evals.tau2_gym.analyze_traces ROLLOUTS.jsonl [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

RUN_TOOLS = {"run", "spawn_subagent"}


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

    calls: dict[str, tuple[str, Any]] = {}
    for item in output:
        if item.get("type") == "function_call":
            calls[item.get("call_id") or item.get("id")] = (
                item.get("name", "tool"),
                _json(item.get("arguments", {})),
            )

    # subagent_id -> type, from `new` (and `fork`, which copies its source's type).
    subagent_types: dict[str, str] = {}
    n_new = n_fork = 0
    run_order: list[str] = []
    response_order: list[str] = []
    n_responses = 0
    for item in output:
        if item.get("type") != "function_call_output":
            continue
        name, arguments = calls.get(item.get("call_id"), ("", {}))
        arguments = arguments if isinstance(arguments, dict) else {}
        result = _json(item.get("output"))
        if name == "new" and isinstance(result, dict) and result.get("subagent_id"):
            n_new += 1
            subagent_types[result["subagent_id"]] = str(arguments.get("subagent_type_id", "unknown"))
        elif name == "fork" and isinstance(result, dict) and result.get("subagent_id"):
            n_fork += 1
            subagent_types[result["subagent_id"]] = subagent_types.get(
                str(arguments.get("subagent_id")), "unknown"
            )
        elif name == "spawn_subagent" and isinstance(result, dict) and result.get("subagent_run_id"):
            n_new += 1
            subagent_types[result["subagent_run_id"]] = str(arguments.get("subagent_type_id", "unknown"))
            run_order.append(result["subagent_run_id"])
        elif name == "run" and isinstance(result, dict) and result.get("subagent_run_id"):
            run_order.append(result["subagent_run_id"])
        elif name == "wait" and isinstance(result, list):
            for entry in result:
                if isinstance(entry, dict) and entry.get("subagent_run_id"):
                    n_responses += 1
                    response_order.append(entry["subagent_run_id"])

    run_batches: list[int] = []
    prompt_lengths: list[int] = []
    runs_per_subagent: Counter[str] = Counter()
    pending = 0
    n_run = n_wait = 0
    for item in output:
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        if name in RUN_TOOLS:
            n_run += 1
            pending += 1
            arguments = _json(item.get("arguments", {}))
            if isinstance(arguments, dict):
                prompt_lengths.append(len(str(arguments.get("prompt", ""))))
                if name == "run" and arguments.get("subagent_id"):
                    runs_per_subagent[str(arguments["subagent_id"])] += 1
        elif name == "wait":
            n_wait += 1
            # Runs started since the previous wait ran concurrently.
            run_batches.append(pending)
            pending = 0
    if pending:
        run_batches.append(pending)

    batches = [b for b in run_batches if b > 0]
    return {
        "task_index": row.get("_ng_task_index"),
        "rollout_index": row.get("_ng_rollout_index"),
        "domain": row.get("domain"),
        "task_id": row.get("task_id"),
        "reward": float(row.get("reward", 0.0)),
        "failure_class": row.get("_ng_failure_class"),
        "n_new": n_new,
        "n_fork": n_fork,
        "n_run": n_run,
        "n_wait": n_wait,
        "n_responses": n_responses,
        "reused_runs": sum(count - 1 for count in runs_per_subagent.values() if count > 1),
        "run_batches": batches,
        "max_fanout": max(batches) if batches else 0,
        "mean_fanout": round(statistics.fmean(batches), 2) if batches else 0.0,
        "sequential": bool(batches) and max(batches) == 1,
        "responses_in_run_order": response_order == run_order[: len(response_order)],
        "subagent_types": dict(Counter(subagent_types.values())),
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

    def share(predicate) -> float:
        return round(sum(1 for r in records if predicate(r)) / len(records), 4) if records else 0.0

    return {
        "rollouts": len(records),
        "pass_rate": share(lambda r: r["reward"] == 1.0),
        "subagents_per_rollout": agg([r["n_new"] + r["n_fork"] for r in records]),
        "forks_per_rollout": agg([r["n_fork"] for r in records]),
        "runs_per_rollout": agg([r["n_run"] for r in records]),
        "reused_runs_per_rollout": agg([r["reused_runs"] for r in records]),
        "waits_per_rollout": agg([r["n_wait"] for r in records]),
        "max_fanout": agg([r["max_fanout"] for r in records]),
        # The headline number: how often the manager actually parallelised.
        "share_fully_sequential": share(lambda r: r["sequential"]),
        "share_any_parallel": share(lambda r: r["max_fanout"] > 1),
        "share_any_fork": share(lambda r: r["n_fork"] > 0),
        "share_any_reuse": share(lambda r: r["reused_runs"] > 0),
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
    header = (
        f"{'domain':<22} {'task':<14} {'rw':>3} {'new':>4} {'fork':>5} {'run':>4} "
        f"{'wait':>5} {'fanout':>7} {'batches'}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        print(
            f"{str(record['domain'])[:22]:<22} {str(record['task_id'])[:14]:<14} "
            f"{record['reward']:>3.0f} {record['n_new']:>4} {record['n_fork']:>5} "
            f"{record['n_run']:>4} {record['n_wait']:>5} "
            f"{record['max_fanout']:>7} {record['run_batches']}"
        )

    if by_domain:
        print("\n== per domain ==")
        for domain, stats in by_domain.items():
            print(
                f"{domain:<24} pass={stats['pass_rate']:<6} "
                f"runs={stats['runs_per_rollout']['mean']:<6} "
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
