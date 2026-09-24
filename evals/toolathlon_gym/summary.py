"""Fixed-sample metrics. Missing, interrupted and infrastructure failures count as failures."""

import argparse
import json
from math import comb
from pathlib import Path

from gyms.toolathlon_gym.scoring import extract_partial_score


def summarize(root, denominator=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    tasks, n = manifest["tasks"], manifest["repetitions"]
    denominator = len(tasks) if denominator is None else denominator
    if denominator < len(tasks) or not tasks or n < 1:
        raise ValueError("Need tasks, positive repetitions and a denominator >= selected tasks")
    attempts = {(e["task"], e["repetition"]): e for e in manifest["episodes"]}
    successes, partials, scored = {}, [], 0
    for task in tasks:
        successes[task] = 0
        for repetition in range(1, n + 1):
            episode = attempts.get((task, repetition))
            if episode is None:
                continue
            path = root / "evals" / task / episode["episode_id"] / "result.json"
            try:
                result = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            scored += 1
            successes[task] += int(episode["returncode"] == 0 and not episode.get("timed_out")
                                   and result.get("pass") is True)
            partial = extract_partial_score(result)
            if partial is not None:
                partials.append(partial.fraction)
    metrics = {}
    for k in (1, 3, 5):
        if k > n:
            continue
        metrics[f"pass@{k}"] = sum(1 - comb(n - s, k) / comb(n, k) for s in successes.values()) / denominator
        if k > 1:
            metrics[f"pass^{k}"] = sum(comb(s, k) / comb(n, k) for s in successes.values()) / denominator
    return {"tasks": len(tasks), "denominator": denominator, "attempts_per_task": n,
            "expected_attempts": len(tasks) * n, "scored_attempts": scored,
            "successful_attempts": sum(successes.values()), "successes_by_task": successes,
            "partial_scores_observed": len(partials),
            "mean_observed_partial_score": sum(partials) / len(partials) if partials else None,
            "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--denominator", type=int)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.denominator), indent=2))


if __name__ == "__main__":
    main()
