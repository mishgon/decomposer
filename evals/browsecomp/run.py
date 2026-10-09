"""Evaluate a fixed task set and aggregate pass rates."""
import asyncio
from collections import defaultdict
import json
from math import comb

from gyms.browsecomp.run import REPO_ROOT, main as run_episodes, parser, save


def summarize(manifest):
    groups = defaultdict(list)
    for row in manifest["episodes"]:
        groups[row["task"]].append(row)
    n = manifest["repetitions"]
    successes = [sum(row["status"] == "finished" and row["evaluation"]["passed"]
                     for row in groups[task]) for task in manifest["tasks"]]
    result = {"tasks": len(successes), "scheduled_attempts": len(successes) * n,
              "ended_attempts": len(manifest["episodes"]),
              "unscored": sum(row["evaluation"]["score"] is None for row in manifest["episodes"]),
              "status": manifest["status"], "denominator": "selected tasks; missing and infra failures count as failures"}
    for k in sorted({1, min(3, n), n}):
        result[f"pass@{k}"] = sum(1 - comb(n - c, k) / comb(n, k) for c in successes) / len(successes)
        result[f"pass^{k}"] = sum(comb(c, k) / comb(n, k) for c in successes) / len(successes)
    return result


async def main(args):
    root = await run_episodes(args)
    summary = summarize(json.loads((root / "manifest.json").read_text()))
    save(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli = parser()
    cli.set_defaults(output_dir=REPO_ROOT / "artifacts/evals/browsecomp")
    asyncio.run(main(cli.parse_args()))
