"""Re-score saved answers with another judge, without changing original results."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import time

from gyms.wideseek.evaluate import evaluate
from gyms.wideseek.runtime import GENERATION, save


async def main(args):
    data = args.data.read_bytes()
    tasks = {task["task_id"]: task for task in map(json.loads, data.splitlines())}
    manifest = json.loads((args.run / "manifest.json").read_text())
    if hashlib.sha256(data).hexdigest() != manifest["settings"]["data_sha256"]:
        raise ValueError("Dataset differs from the original run")
    paths = sorted(args.run.glob("*/*/attempt-???/result.json"))
    if not paths:
        raise ValueError("No completed attempt results found")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "manifest.json", {"source_run": str(args.run.resolve()),
        "source_manifest": manifest, "judge_model": args.judge_model,
        "generation": {**GENERATION, "temperature": 0., "presence_penalty": 0.},
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at": time.time(), "attempts": len(paths)})
    semaphore = asyncio.Semaphore(args.concurrency)

    async def score(path):
        original = json.loads(path.read_text())
        destination = args.output / path.parent.relative_to(args.run)
        async with semaphore:
            started = time.time()
            try:
                result = await asyncio.wait_for(evaluate(tasks[original["task_id"]], original["answer"],
                    destination, judge_model_id=args.judge_model), 600)
            except Exception as exc:
                result = {"status": "evaluation_error", "score": None,
                          "error": f"{type(exc).__name__}: {exc}"}
        row = {"task_id": original["task_id"], "mode": original["mode"], "attempt": original["attempt"],
               "agent_status": original["status"], "previous_evaluation": original["evaluation"],
               "evaluation": result, "source_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "judge_model": args.judge_model, "started_at": started, "finished_at": time.time()}
        save(destination / "result.json", row)
        print(json.dumps(row), flush=True)
        return row

    rows = await asyncio.gather(*(score(path) for path in paths))
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        selected = [r for r in rows if r["mode"] == mode]
        scores = [r["evaluation"]["score"] for r in selected if r["evaluation"]["score"] is not None]
        summary[mode] = {"attempts": len(selected), "scored": len(scores), "unscored": len(selected)-len(scores),
            "mean_native_score": sum(scores)/len(scores) if scores else None,
            "mean_native_score_infra_zero": sum(scores)/len(selected)}
    save(args.output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("Concurrency must be positive")
    asyncio.run(main(args))
