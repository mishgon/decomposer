"""Extract one decomposer rollout as a chat transcript.

`trace_stats.py` measures rollouts in aggregate; this renders a single one for
reading. It reuses that module's LangGraph parser and outcome reader, and
mirrors its interval-restricted `(name, args)` matching -- but returns whole
environment records rather than only their results, because a chat view needs
each call's timestamp, latency and payload.

Manager messages carry no timestamps of their own. Every message is therefore
anchored to something that does: a spawn to the created time of the run it
returned, a wait to the last ended time among the runs it collected. Those are
*return* times, the same convention the timeline uses, so the two views share
one clock by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gyms.gaia2.trace_stats import (  # noqa: E402
    Lifecycle,
    _outcomes,
    parse_langgraph_log,
)

RESULT_CHARS = 800
PROMPT_CHARS = 2000


def _clip(text: str, limit: int) -> tuple[str, int]:
    """Return ``text`` clipped to ``limit`` characters plus its true length."""
    full = len(text)
    return (text if full <= limit else text[:limit]), full


def join_env_calls(
    subagents: list[dict[str, Any]],
    env_calls: list[dict[str, Any]],
    lifecycles: dict[str, Lifecycle],
) -> dict[str, list[dict[str, Any] | None]]:
    """Attribute each subagent's calls to the flat env record that ran them.

    Mirrors ``trace_stats.join_tool_calls``: a subagent record keeps only
    ``(name, args)`` and identical reads repeat across subagents, so candidates
    are restricted to the subagent's own LangGraph interval before being
    consumed in order.
    """
    buckets: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(env_calls):
        key = (record["tool"], json.dumps(record.get("arguments"), sort_keys=True, default=str))
        buckets.setdefault(key, []).append(index)

    taken: set[int] = set()
    out: dict[str, list[dict[str, Any] | None]] = {}
    for state in subagents:
        run_id = state["subagent_run_id"]
        life = lifecycles.get(run_id)
        low = life.started if life and life.started else -math.inf
        high = life.ended if life and life.ended else math.inf
        picked: list[dict[str, Any] | None] = []
        for call in state.get("tool_calls") or []:
            key = (call["name"], json.dumps(call.get("args"), sort_keys=True, default=str))
            candidates = [i for i in buckets.get(key, []) if i not in taken]
            inside = [i for i in candidates if low <= env_calls[i]["started_at"] <= high]
            chosen = inside[0] if inside else (candidates[0] if candidates else None)
            if chosen is None:
                picked.append(None)
                continue
            taken.add(chosen)
            picked.append(env_calls[chosen])
        out[run_id] = picked
    return out


def _run_ids(content: str) -> list[str]:
    """Run ids named by a tool result, or [] when it is an error string."""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    return [r["subagent_run_id"] for r in payload if isinstance(r, dict) and "subagent_run_id" in r]


def build(run_dir: Path, scenario: str, run: int) -> dict[str, Any]:
    sidecar = run_dir / "decomposer_sidecars" / f"{scenario}__run{run}.json"
    raw = json.loads(sidecar.read_text())
    turn = raw["turns"][0]
    manager = turn["manager"]
    messages = manager["trace"]["manager_messages"]
    states = manager["subagent_states"]
    env_calls = raw.get("tool_calls") or []

    lifecycles = parse_langgraph_log(run_dir / "logs" / "langgraph_subagent.log")
    known = [lifecycles[s["subagent_run_id"]] for s in states if s["subagent_run_id"] in lifecycles]
    if len(known) != len(states):
        raise SystemExit(f"run {run}: only {len(known)}/{len(states)} run ids found in the log")
    t0 = min(life.created for life in known)
    elapsed = float(manager["timing"]["elapsed_seconds"])

    order = sorted(states, key=lambda s: lifecycles[s["subagent_run_id"]].created)
    number = {s["subagent_run_id"]: i + 1 for i, s in enumerate(order)}
    joined = join_env_calls(states, env_calls, lifecycles)

    subagents = []
    for state in order:
        run_id = state["subagent_run_id"]
        life = lifecycles[run_id]
        calls = []
        for call, record in zip(state.get("tool_calls") or [], joined[run_id]):
            body = "" if record is None else json.dumps(record.get("result"), default=str)
            text, full = _clip(body, RESULT_CHARS)
            calls.append({
                "tool": call["name"],
                "args": call.get("args") or {},
                "at": None if record is None else round(record["started_at"] - t0, 2),
                "ms": None if record is None else round(1000 * record.get("latency_seconds", 0), 1),
                "ok": True if record is None else bool(record.get("ok", True)),
                "result": text,
                "bytes": full,
                "matched": record is not None,
            })
        report = (state.get("report") or {}).get("content") or ""
        subagents.append({
            "n": number[run_id],
            "runId": run_id,
            "status": state.get("status"),
            "created": round(life.created - t0, 2),
            "started": round(life.started - t0, 2),
            "ended": round(life.ended - t0, 2),
            "queueMs": round(1000 * (life.started - life.created)),
            "prompt": state.get("prompt") or "",
            "report": report,
            "calls": calls,
        })

    # --- the transcript, anchored to the lifecycles ---
    out_msgs: list[dict[str, Any]] = []
    last_t = 0.0
    for i, message in enumerate(messages):
        kind = message["type"]
        data = message["data"]
        usage = data.get("usage_metadata") or {}
        tokens = {"in": usage.get("input_tokens"), "out": usage.get("output_tokens")}
        if kind == "human":
            out_msgs.append({"role": "user", "kind": "task", "t": 0.0,
                             "text": data.get("content") or ""})
            continue
        if kind == "tool":
            continue  # folded into the ai message that called it
        calls = data.get("tool_calls") or []
        if not calls:  # the final answer
            out_msgs.append({"role": "manager", "kind": "answer", "t": round(elapsed, 2),
                             "text": data.get("content") or "", "tokens": tokens})
            continue
        call = calls[0]
        result = messages[i + 1]["data"].get("content") if i + 1 < len(messages) else ""
        ids = _run_ids(str(result))
        if call["name"] == "spawn_subagent":
            run_id = ids[0]
            t = round(lifecycles[run_id].created - t0, 2)
            last_t = t
            out_msgs.append({"role": "manager", "kind": "spawn", "t": t, "sub": number[run_id],
                             "text": call["args"].get("prompt") or "", "tokens": tokens})
        elif ids:
            t = round(max(lifecycles[r].ended for r in ids) - t0, 2)
            last_t = t
            out_msgs.append({"role": "manager", "kind": "wait", "t": t, "endedAt": t,
                             "collected": [number[r] for r in ids], "tokens": tokens})
        else:
            # A wait with nothing running returns an error and is not timed by
            # anything. Its position is known, its instant is not.
            out_msgs.append({"role": "manager", "kind": "empty", "t": None,
                             "text": str(result), "tokens": tokens})

    # A wait cannot return before it was issued, but the only clock on it is
    # when its subagent finished -- and a subagent can finish while the manager
    # is still generating an earlier call. Where that happened the anchor is
    # pulled forward to the previous act and the message is flagged: the wait
    # returned instantly because the work was already done.
    for i, m in enumerate(out_msgs):
        if m["t"] is None or i == 0:
            continue
        floor = max((z["t"] for z in out_msgs[:i] if z["t"] is not None), default=0.0)
        if m["t"] < floor:
            m["immediate"] = True
            m["t"] = floor

    # Untimed messages (empty waits) are spread across the gap between their
    # timed neighbours, wherever in the transcript they fall.
    i = 0
    while i < len(out_msgs):
        if out_msgs[i]["t"] is not None:
            i += 1
            continue
        j = i
        while j < len(out_msgs) and out_msgs[j]["t"] is None:
            j += 1
        start = out_msgs[i - 1]["t"] if i else 0.0
        stop = out_msgs[j]["t"] if j < len(out_msgs) else elapsed
        step = (stop - start) / (j - i + 1)
        for k in range(i, j):
            out_msgs[k]["t"] = round(start + step * (k - i + 1), 2)
            out_msgs[k]["derived"] = True
        i = j

    scores = _outcomes(run_dir)
    reward = scores.get((scenario, run))
    rationale = ""
    for line in (run_dir / "output.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        if str(meta.get("scenario_id")) == scenario and int(meta.get("run_number", -1)) == run:
            value = meta.get("rationale")
            # a passing rollout carries the *string* "None", not a null
            rationale = "" if value in (None, "None") else str(value)
            break

    return {
        "scenario": scenario,
        "run": run,
        "reward": reward,
        "rationale": rationale,
        "elapsed": round(elapsed, 2),
        "manager": raw["configuration"]["model_configuration"]["manager"]["served_name"],
        "worker": raw["configuration"]["model_configuration"]["subagent"]["served_name"],
        "usage": manager["usage"],
        "envTotal": len(env_calls),
        "messages": out_msgs,
        "subagents": subagents,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    traces = [build(args.run_dir, args.scenario, r) for r in args.runs]

    for trace in traces:
        # every spawn card must sit on its lane's created, every wait on the
        # last ended it collected -- the two views agree or the build fails
        lanes = {s["n"]: s for s in trace["subagents"]}
        for m in trace["messages"]:
            if m["kind"] == "spawn":
                assert m["t"] == lanes[m["sub"]]["created"], (trace["run"], m)
            if m["kind"] == "wait":
                assert m["endedAt"] == max(lanes[n]["ended"] for n in m["collected"]), (trace["run"], m)
        times = [m["t"] for m in trace["messages"]]
        assert times == sorted(times), (trace["run"], "transcript out of order", times)
        got = sum(len(s["calls"]) for s in trace["subagents"])
        assert got == trace["envTotal"], f"run {trace['run']}: {got} != {trace['envTotal']}"
        tin = sum(m.get("tokens", {}).get("in") or 0 for m in trace["messages"])
        tout = sum(m.get("tokens", {}).get("out") or 0 for m in trace["messages"])
        assert (tin, tout) == (trace["usage"]["input_tokens"], trace["usage"]["output_tokens"]), \
            f"run {trace['run']}: tokens {tin}/{tout} vs {trace['usage']}"
        unmatched = sum(1 for s in trace["subagents"] for c in s["calls"] if not c["matched"])
        print(f"run {trace['run']}: {len(trace['messages'])} messages, "
              f"{len(trace['subagents'])} subagents, {trace['envTotal']} env calls, "
              f"reward {trace['reward']}, {unmatched} unmatched calls")

    args.out.write_text(json.dumps(traces, default=str))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
