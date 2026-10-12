"""Observed GAIA2 usage, timings, outcomes, and worker concurrency."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from gyms.gaia2.prepare import RESULTS_ROOT

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_FIELD = re.compile(r"(\w+)=(\S+)")


@dataclass(frozen=True)
class Lifecycle:
    """One subagent run as the LangGraph server saw it."""

    run_id: str
    status: str
    created: float | None
    started: float | None
    ended: float | None
    exec_seconds: float | None
    wait_seconds: float | None


def _iso(value: str | None) -> float | None:
    if not value or value == "None":
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _ms(value: str | None) -> float | None:
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def parse_langgraph_log(path: Path) -> dict[str, Lifecycle]:
    """Map ``run_id`` to its lifecycle for every terminated subagent run."""
    out: dict[str, Lifecycle] = {}
    if not path.exists():
        return out
    with path.open(errors="ignore") as handle:
        for line in handle:
            if "Background run " not in line:
                continue
            clean = _ANSI.sub("", line)
            fields = dict(_FIELD.findall(clean))
            run_id = fields.get("run_id")
            if not run_id:
                continue
            out[run_id] = Lifecycle(
                run_id=run_id,
                status=clean.split("Background run ", 1)[1].split()[0],
                created=_iso(fields.get("run_created_at")),
                started=_iso(fields.get("run_started_at")),
                ended=_iso(fields.get("run_ended_at")),
                exec_seconds=_ms(fields.get("run_exec_ms")),
                wait_seconds=_ms(fields.get("run_wait_time_ms")),
            )
    return out


def temporal_parallelism(
    intervals: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    """Sweep the real subagent intervals for overlap.

    ``mean_concurrent`` is time-weighted over the span in which any subagent was
    running, so it is not diluted by the manager's own thinking time;
    ``busy_sum / busy_union`` measures overlap, not end-to-end speedup.
    """
    clean = [
        (lo, hi)
        for lo, hi in intervals
        if lo is not None and hi is not None and hi >= lo
    ]
    if not clean:
        return {
            "max_concurrent": 0,
            "mean_concurrent": 0.0,
            "busy_union_seconds": 0.0,
            "busy_sum_seconds": 0.0,
            "overlap_factor": 0.0,
            "measured_subagents": 0,
        }

    events: list[tuple[float, int]] = []
    for lo, hi in clean:
        events.append((lo, 1))
        events.append((hi, -1))
    events.sort()

    active = 0
    peak = 0
    union = 0.0
    weighted = 0.0
    previous = events[0][0]
    for stamp, delta in events:
        if active > 0:
            span = stamp - previous
            union += span
            weighted += span * active
        previous = stamp
        active += delta
        peak = max(peak, active)

    busy_sum = sum(hi - lo for lo, hi in clean)
    return {
        "max_concurrent": peak,
        "mean_concurrent": round(weighted / union, 3) if union else 0.0,
        "busy_union_seconds": round(union, 3),
        "busy_sum_seconds": round(busy_sum, 3),
        "overlap_factor": round(busy_sum / union, 3) if union else 0.0,
        "measured_subagents": len(clean),
    }


def _outcomes(run_dir: Path) -> dict[tuple[str, int | None], float | None]:
    """Per-rollout score keyed by (scenario, run number).

    ``output.jsonl`` rows are not in run order, so the repeat index has to come
    from ``metadata.run_number`` rather than from the position in the file;
    deriving it by counting silently mispairs scores with rollouts.
    """
    path = run_dir / "output.jsonl"
    if not path.exists():
        return {}
    out: dict[tuple[str, int | None], float | None] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        scenario = str(meta.get("scenario_id") or row.get("task_id"))
        run_number = meta.get("run_number")
        # ARE records no run number when each scenario runs once (--num-repeats
        # 1); the rollout records carry None then too, so the key still joins.
        key = (scenario, None if run_number is None else int(run_number))
        out[key] = (
            float(row["score"]) if isinstance(row.get("score"), (int, float)) else None
        )
    return out


def analyse(run_dir: Path) -> dict[str, Any]:
    lifecycles = parse_langgraph_log(
        run_dir / "logs" / "agent_server.log"
        if (run_dir / "logs" / "agent_server.log").exists()
        else run_dir / "logs" / "langgraph_subagent.log"
    )
    outcomes = _outcomes(run_dir)
    records = {
        key: {"scenario_id": key[0], "run_number": key[1], "score": score}
        for key, score in outcomes.items()
    }
    for path in sorted((run_dir / "decomposer_sidecars").glob("*.json")):
        data = json.loads(path.read_text())
        key = (data["scenario_id"], data.get("run_number"))
        turns = data.get("turns", [])
        managers = [turn["manager"] for turn in turns]
        # Manager state and usage are cumulative across turns.
        snapshots = [manager for manager in managers if "usage" in manager]
        usage = snapshots[-1]["usage"] if snapshots else None
        states = {
            state["agent_run_id"]: state
            for manager in managers
            for state in manager.get("subagent_states", [])
        }
        lives = [lifecycles[run_id] for run_id in states if run_id in lifecycles]
        intervals = [
            (life.started, life.ended)
            for life in lives
            if life.started is not None and life.ended is not None
        ]
        record = records.setdefault(
            key, {"scenario_id": key[0], "run_number": key[1], "score": None}
        )
        record.update(
            {
                "manager_usage": usage or None,
                "manager_usage_complete": bool(managers)
                and all(bool(m.get("usage")) for m in managers),
                "worker_usage": None,
                "manager_turn_seconds": sum(
                    m.get("timing", {}).get("elapsed_seconds", 0) for m in managers
                ),
                "manager_failures": [
                    m["failure"] for m in managers if m.get("failure")
                ],
                "subagent_statuses": dict(
                    Counter(s.get("status", "unknown") for s in states.values())
                ),
                "parallelism": temporal_parallelism(intervals),
                "env_tool_calls": len(data.get("tool_calls", [])),
                "env_tool_seconds": sum(
                    c.get("latency_seconds", 0) for c in data.get("tool_calls", [])
                ),
            }
        )
    for path in sorted((run_dir / "lite").glob("*.json")):
        data = json.loads(path.read_text())
        match = re.search(r"_run_(\d+)(?:_|\.)", path.name)
        run_number = data.get("run_number", int(match[1]) if match else None)
        key = (data["scenario_id"], run_number)
        if key in records:
            records[key]["run_duration_seconds"] = data.get("run_duration")
            records[key]["are_llm_usage"] = (
                data.get("per_agent_llm_usage_stats") or None
            )
            records[key]["validation_decision"] = data.get("validation_decision")
    rollouts = list(records.values())
    usage_totals = Counter()
    for record in rollouts:
        usage_totals.update(record.get("manager_usage") or {})
    scores = [r["score"] for r in rollouts if r["score"] is not None]
    return {
        "summary": {
            "rollouts": len(rollouts),
            "scored": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "passed": sum(score == 1 for score in scores),
            "manager_observed_usage": dict(usage_totals) or None,
            "manager_usage_complete_rollouts": sum(
                r.get("manager_usage_complete", False) for r in rollouts
            ),
            "worker_usage": None,
        },
        "rollouts": rollouts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "trace_stats")
    args = parser.parse_args()
    result = analyse(args.run_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.run_dir.name}.json").write_text(
        json.dumps(result["summary"], indent=2)
    )
    (args.out / f"{args.run_dir.name}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in result["rollouts"])
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
