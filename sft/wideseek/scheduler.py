"""Coverage first, then bounded water-filling, matching Toolathlon SFT policy."""
import math


def qualifies(result, threshold):
    score = result["evaluation"].get("score")
    return (result["status"] == "finished" and not result.get("cleanup_errors")
            and result.get("trace_available", True)
            and result["evaluation"].get("status") == "scored"
            and type(score) in (int, float) and math.isfinite(score)
            and 0 <= score <= 1 and (score == 1. or score > threshold))


def new_state(task_ids, threshold=.9, target_successes=4):
    if not 0 <= threshold <= 1 or target_successes < 1:
        raise ValueError("Threshold must be in [0, 1] and target successes positive")
    return {"schema_version": 1, "policy": "coverage_first", "phase": "coverage_first",
            "task_ids": list(task_ids), "success_threshold": threshold,
            "target_successes": target_successes, "max_zero_success_launches": 6,
            "max_balance_launches": 24, "culled_tasks": [], "balance_start": {}, "waves": []}


def statistics(state, results):
    stats = {task: {"attempts": 0, "successes": 0, "next_attempt": 1} for task in state["task_ids"]}
    for result in results:
        row = stats[result["task_id"]]
        row["attempts"] += 1
        row["successes"] += qualifies(result, state["success_threshold"])
        row["next_attempt"] = max(row["next_attempt"], result["attempt"] + 1)
    state["task_statistics"] = stats
    state["covered_tasks"] = sum(row["successes"] > 0 for row in stats.values())
    return stats


def plan_next_wave(state, results, *, allow_new_wave=True):
    """Persist a wave before launching; resume only its unfinished attempts."""
    stats = statistics(state, results)
    ended = {(r["task_id"], r["attempt"]) for r in results}
    if state["waves"]:
        pending = [job for job in state["waves"][-1]["jobs"] if tuple(job) not in ended]
        if pending:
            return pending
    if state["phase"] == "complete":
        return []
    if state["phase"] == "coverage_first":
        culled = set(state["culled_tasks"])
        for task in state["task_ids"]:
            if (stats[task]["successes"] == 0 and stats[task]["attempts"] >= state["max_zero_success_launches"]
                    and task not in culled):
                state["culled_tasks"].append(task)
                culled.add(task)
        tasks = [t for t in state["task_ids"] if t not in culled and not stats[t]["successes"]]
        if not tasks:
            state["phase"] = "balance_successes"
            state["balance_start"] = {t: stats[t]["attempts"] for t in state["task_ids"]
                                      if t not in culled}
            return plan_next_wave(state, results, allow_new_wave=allow_new_wave)
        target = 1
    elif state["phase"] == "balance_successes":
        eligible = [t for t in state["balance_start"]
                    if stats[t]["attempts"] - state["balance_start"][t] < state["max_balance_launches"]]
        if not eligible or min(stats[t]["successes"] for t in eligible) >= state["target_successes"]:
            state["phase"] = "complete"
            return []
        target = min(stats[t]["successes"] for t in eligible) + 1
        tasks = [t for t in eligible if stats[t]["successes"] < target]
    else:
        raise ValueError(f"Unknown collection phase: {state['phase']}")
    if not allow_new_wave:
        return []
    jobs = [[task, stats[task]["next_attempt"]] for task in tasks]
    state["waves"].append({"phase": state["phase"], "target_successes": target, "jobs": jobs})
    return jobs
