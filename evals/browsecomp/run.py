"""Evaluate a fixed task set and aggregate pass rates."""
import asyncio
from collections import defaultdict
import json
from math import comb
from uuid import uuid4

from gyms.browsecomp.run import REPO_ROOT, main as run_episodes, parser, save
from gyms.browsecomp.evaluate import evaluate
from gyms.infrastructure import classify


def summarize(manifest):
    groups = defaultdict(list)
    for row in manifest["episodes"]:
        groups[row["task"]].append(row)
    n = manifest["repetitions"]
    successes = [sum(row["status"] == "finished" and not row.get("infrastructure_errors") and row["evaluation"]["passed"]
                     for row in groups[task]) for task in manifest["tasks"]]
    result = {"tasks": len(successes), "scheduled_attempts": len(successes) * n,
              "ended_attempts": len(manifest["episodes"]),
              "unscored": sum(row["evaluation"]["score"] is None for row in manifest["episodes"]),
              "infrastructure_errors": sum(bool(row.get("infrastructure_errors")) for row in manifest["episodes"]),
              "status": manifest["status"], "denominator": "selected tasks; missing and infra failures count as failures"}
    for k in sorted({1, min(3, n), n}):
        result[f"pass@{k}"] = sum(1 - comb(n - c, k) / comb(n, k) for c in successes) / len(successes)
        result[f"pass^{k}"] = sum(comb(c, k) / comb(n, k) for c in successes) / len(successes)
    return result


async def main(args):
    root = await run_episodes(args)
    manifest = json.loads((root / "manifest.json").read_text())
    # Resume reuses generated traces when only the judge failed.
    if args.resume:
        for entry in manifest["episodes"]:
            if entry.get("infrastructure_errors") or entry["evaluation"]["status"] != "evaluation_error":
                continue
            directory = root / entry["directory"]
            entry.setdefault("evaluation_attempts", []).append(entry["evaluation"])
            judge = directory / "judge.json"
            if judge.exists():
                judge.rename(directory / f"judge-{uuid4().hex[:8]}.json")
            trace = json.loads((directory / "trace.json").read_text())
            last = trace["messages"][-1] if trace["messages"] else {}
            last = last.get("data", last)
            output = last.get("content", "") if last.get("type") == "ai" and not last.get("tool_calls") else ""
            if not isinstance(output, str):
                output = "\n".join(block.get("text", "") for block in output if isinstance(block, dict))
            try:
                entry["evaluation"] = await asyncio.wait_for(evaluate(json.loads((directory / "task.json").read_text()), output, directory), 180)
            except Exception as error:
                entry["evaluation"] = {"status": "evaluation_error", "score": None, "passed": False, "error": repr(error)}
            save(directory / "result.json", entry)
            save(root / "manifest.json", manifest)
            if classify(entry["evaluation"].get("error")):
                break
        if len(manifest["episodes"]) == len(manifest["tasks"]) * manifest["repetitions"] and not any(
            e.get("infrastructure_errors") or e["evaluation"]["status"] == "evaluation_error" for e in manifest["episodes"]
        ):
            manifest["status"] = "completed"
        save(root / "manifest.json", manifest)
    summary = summarize(manifest)
    save(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli = parser()
    cli.set_defaults(output_dir=REPO_ROOT / "artifacts/evals/browsecomp")
    asyncio.run(main(cli.parse_args()))
