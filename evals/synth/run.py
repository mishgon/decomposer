"""Stock-policy baseline and checkpoint evaluation on deterministic file swaps."""
import argparse
import asyncio
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import time
from uuid import uuid4

from decomposer.agent_server import agent_server
from decomposer.models import create_model
from gyms.synth import ROOT
from gyms.synth.run import episode
from gyms.synth.task import make_task


def summarize(rows, n):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    successes = [sum(r["passed"] for r in group) for group in groups.values()]
    return {"attempts": len(rows), "tasks": len(groups),
            "mean_score": sum(r["score"] for r in rows) / len(rows),
            "pass@1": sum(successes) / len(rows),
            f"pass@{n}": sum(c > 0 for c in successes) / len(groups),
            f"pass^{n}": sum(c == n for c in successes) / len(groups),
            "correct_and_parallel": sum(r["passed"] and r["parallel"] for r in rows) / len(rows),
            "mean_copies": sum(r["copies"] for r in rows) / len(rows),
            "errors": sum(r["status"] != "finished" for r in rows)}


async def main(args):
    if min(args.tasks, args.repetitions, args.concurrency) < 1:
        raise ValueError("Tasks, repetitions and concurrency must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["SYNTH_ARTIFACT_ROOT"] = str(root)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = {"status": "running", "model": args.model, "split": args.split,
                "tasks": args.tasks, "repetitions": args.repetitions,
                "started_at": time.time(), "episodes": []}

    def save():
        temp = root / "manifest.tmp"
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(root / "manifest.json")

    save()
    policy = create_model(args.model)
    semaphore = asyncio.Semaphore(args.concurrency)
    async def run(task, attempt, worker_url):
        async with semaphore:
            result = await episode(task, policy, root / f"{task['task_id']}-r{attempt}", worker_url)
            manifest["episodes"].append({**result, "attempt": attempt})
            save()
            print(f"{len(manifest['episodes'])}/{args.tasks * args.repetitions}: "
                  f"{task['task_id']} score={result['score']} parallel={result['parallel']}", flush=True)
    try:
        async with agent_server(ROOT / "gyms/synth/langgraph.json", port=port, n_jobs_per_worker=64) as url:
            # TaskGroup cancels and joins episodes before stopping their shared server.
            async with asyncio.TaskGroup() as group:
                for index in range(args.tasks):
                    for attempt in range(1, args.repetitions + 1):
                        group.create_task(run(make_task(args.split, index), attempt, url))
        manifest["status"] = "completed"
        summary = summarize(manifest["episodes"], args.repetitions)
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        manifest["finished_at"] = time.time()
        save()
        policy.http_client.close()
        await policy.http_async_client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vllm/qwen_3_5_4b_non_thinking")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("-n", "--repetitions", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evals/synth" / uuid4().hex[:8])
    asyncio.run(main(parser.parse_args()))
