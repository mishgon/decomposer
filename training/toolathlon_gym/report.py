"""Compare complete, fixed reward panels; never mix them with training batches."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(episodes, expected, samples):
    groups = defaultdict(lambda: defaultdict(list))
    for trace, evaluation in episodes:
        source = trace.get("data_source")
        if source not in expected:
            continue
        versions = trace["weight_versions"]
        step = versions["min_global_steps"]
        if step != versions["max_global_steps"]:
            raise ValueError("Evaluation trajectory spans policy versions")
        task = trace["task_id"]
        if task not in expected[source]:
            raise ValueError("Unexpected task in evaluation panel")
        groups[step, source][task].append(evaluation)
    panels = []
    for (step, source), tasks in sorted(groups.items()):
        counts = {task: len(tasks[task]) for task in sorted(expected[source])}
        complete = all(count == samples for count in counts.values())
        panels.append({
            "step": step, "panel": source, "complete": complete, "counts": counts,
            "reward": mean(mean(e["reward"] for e in tasks[t]) for t in counts) if complete else None,
            "strict_pass_rate": mean(mean(float(e["pass"]) for e in tasks[t]) for t in counts)
                                if complete else None,
            "task_rewards": {t: [e["reward"] for e in tasks[t]] for t in counts},
        })
    baseline = {p["panel"]: p for p in panels if p["step"] == 0 and p["complete"]}
    for panel in panels:
        initial = baseline.get(panel["panel"])
        panel["reward_delta"] = (panel["reward"] - initial["reward"]
                                 if panel["complete"] and initial else None)
    return panels


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    frame = pd.read_parquet(args.data / "evaluation.parquet")
    expected = {source: {row["task_id"] for row in group.extra_info}
                for source, group in frame.groupby("data_source")}
    episodes = []
    for path in sorted((args.run / "episodes").glob("*/trace.json")):
        evaluation = path.with_name("evaluation.json")
        if evaluation.exists():
            result = json.loads(evaluation.read_text())
            if "reward" in result:
                episodes.append((json.loads(path.read_text()), result))
    report = summarize(episodes, expected, args.samples)
    print(json.dumps(report, indent=2))
    (args.run / "reward_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
