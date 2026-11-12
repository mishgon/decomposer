"""Summarize saved MASBench episodes by agent, axis and difficulty value."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(root):
    manifest = json.loads((root / "manifest.json").read_text())
    groups = defaultdict(list)
    for row in manifest["episodes"]:
        for axis, value in (("all", "all"), (row["axis"], "all"), (row["axis"], row["value"])):
            groups[(row["agent"], axis, value)].append(row)
    rows = []
    for (agent, axis, value), episodes in sorted(groups.items()):
        rows.append({
            "agent": agent, "axis": axis, "value": value, "episodes": len(episodes),
            "accuracy": sum(row["evaluation"]["passed"] for row in episodes) / len(episodes),
            "errors": sum(row["status"] == "error" for row in episodes),
            "timeouts": sum(row["status"] == "timeout" for row in episodes),
        })
    return {"run_status": manifest["status"], "completed_episodes": len(manifest["episodes"]),
            "planned_episodes": len(manifest["tasks"]) * len(manifest["agents"]) * manifest["repetitions"],
            "groups": rows}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.run), indent=2))
