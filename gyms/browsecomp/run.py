"""Run isolated BrowseComp episodes and save their raw results."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from uuid import uuid4

import httpx
from decomposer.agent_server import agent_server, invoke_and_capture
from decomposer.usage import build_usage_summary
from decomposer.visualization import write_trace_html
from gyms.browsecomp import REPO_ROOT
from gyms.browsecomp.agents import DECOMPOSER_MODEL, RESEARCHER_MODEL
from gyms.browsecomp.evaluate import FINAL_FORMAT, evaluate


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


async def episode(args):
    """Runs in its own process so server addresses and log paths cannot collide."""
    root = args.output_dir.resolve()
    task = json.loads((root / "task.json").read_text())
    os.environ["BROWSECOMP_RETRIEVAL_URL"] = args.retrieval_url
    os.environ["BROWSECOMP_MODEL_LOG"] = str(root / "model_calls.jsonl")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    os.environ["BROWSECOMP_AGENT_URL"] = f"http://127.0.0.1:{port}"
    result = {"task": task["task_id"], "episode_id": root.name, "run_id": root.parent.parent.name,
              "repetition": args.repetition, "started_at": now(), "assistant_id": args.agent}
    state = {}
    error = None
    try:
        async with agent_server(Path(__file__).with_name("langgraph.json"), port=port):
            state, error = await invoke_and_capture(
                os.environ["BROWSECOMP_AGENT_URL"], args.agent,
                {"messages": [{"role": "user", "content": task["question"] + FINAL_FORMAT}]},
                timeout=args.timeout, config={"recursion_limit": 410},
            )
    except Exception as exc:
        error = exc
    result["agent_finished_at"] = now()
    result["status"] = "finished" if error is None else "timeout" if isinstance(error, TimeoutError) else "error"
    result["error"] = repr(error) if error else None
    trace = {**state, **result, "harness": "decomposer" if args.agent == "decomposer" else "react",
             "agent_error": result["error"], "finished_at": result["agent_finished_at"],
             "model": DECOMPOSER_MODEL if args.agent == "decomposer" else RESEARCHER_MODEL,
             "messages": state.get("messages", []), "agents": state.get("agents", {}),
             "agent_runs": state.get("agent_runs", {})}
    save(root / "trace.json", trace)
    write_trace_html(trace, root / "trace.html")
    usage = build_usage_summary(trace["messages"], trace["agent_runs"], trace["agents"])
    if args.agent == "researcher":
        usage["react"] = usage.pop("decomposer")
    save(root / "usage.json", usage)
    last = trace["messages"][-1] if trace["messages"] else {}
    last = last.get("data", last)
    output = last.get("content", "") if last.get("type") == "ai" and not last.get("tool_calls") else ""
    if not isinstance(output, str):
        output = "\n".join(block.get("text", "") for block in output if isinstance(block, dict))
    try:
        result["evaluation"] = await asyncio.wait_for(evaluate(task, output, root), 180)
    except Exception as exc:
        result["evaluation"] = {"status": "evaluation_error", "score": None, "passed": False, "error": repr(exc)}
    result["finished_at"] = now()
    save(root / "result.json", result)
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        raise error


async def preflight(url):
    """Require the correct retrieval corpus before scheduling model calls."""
    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
        response = await client.post(url.rstrip("/") + "/search", json={
            "query": "Albert Einstein", "limit": 1, "source": "browsecomp_plus"})
        response.raise_for_status()
        hits = response.json()["results"]
        if not hits:
            raise RuntimeError("BrowseComp retrieval preflight returned no documents")
        response = await client.post(url.rstrip("/") + "/fetch",
                                    json={"page_id": hits[0]["page_id"], "source": "browsecomp_plus"})
        response.raise_for_status()
        if response.json().get("error"):
            raise RuntimeError(f"BrowseComp fetch preflight failed: {response.json()['error']}")


async def main(args):
    if args.concurrency < 1 or args.repetitions < 1 or args.timeout <= 0:
        raise ValueError("Concurrency, repetitions and timeout must be positive")
    tasks = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if args.tasks:
        selected = set(args.tasks)
        if len(selected) != len(args.tasks) or not selected <= {t["task_id"] for t in tasks}:
            raise ValueError("Task IDs must be unique and present in the dataset")
        tasks = [t for t in tasks if t["task_id"] in selected]
    elif not args.all:
        raise ValueError("Choose --tasks or --all")
    if not tasks:
        raise ValueError("No tasks selected")
    await preflight(args.retrieval_url)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    root = args.output_dir.resolve() / run_id
    root.mkdir(parents=True)
    manifest = {"run_id": run_id, "status": "running", "started_at": now(),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
                "assistant_id": args.agent, "tasks": [t["task_id"] for t in tasks],
                "repetitions": args.repetitions, "episodes": [],
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    save(root / "manifest.json", manifest)
    print(f"Artifacts: {root}", flush=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def launch(task, repetition):
        async with semaphore:
            directory = root / "traces" / f"{task['task_id']}-r{repetition:03d}"
            directory.mkdir(parents=True)
            save(directory / "task.json", task)
            command = [sys.executable, "-m", "gyms.browsecomp.run", "--episode",
                       "--agent", args.agent, "--output-dir", str(directory),
                       "--retrieval-url", args.retrieval_url, "--timeout", str(args.timeout),
                       "--repetition", str(repetition)]
            env = {**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}"}
            with (directory / "runner.log").open("w") as log:
                process = await asyncio.create_subprocess_exec(*command, cwd=REPO_ROOT, env=env,
                                                               stdout=log, stderr=log)
                try:
                    code = await asyncio.wait_for(process.wait(), args.timeout + 360)
                except TimeoutError:
                    code = -1
                finally:
                    if process.returncode is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            await asyncio.wait_for(process.wait(), 120)
                        except TimeoutError:
                            process.kill()
                            await process.wait()
            entry = {"task": task["task_id"], "repetition": repetition,
                     "returncode": code, "directory": str(directory.relative_to(root))}
            if (directory / "result.json").exists():
                entry.update(json.loads((directory / "result.json").read_text()))
            else:
                entry.update(status="error", error="Episode exited without result.json",
                             evaluation={"status": "evaluation_error", "score": None, "passed": False})
            manifest["episodes"].append(entry)
            save(root / "manifest.json", manifest)
            print(f"{len(manifest['episodes'])}/{len(tasks) * args.repetitions} {task['task_id']}: {entry['status']}", flush=True)

    jobs = [asyncio.create_task(launch(task, rep)) for task in tasks for rep in range(1, args.repetitions + 1)]
    try:
        await asyncio.gather(*jobs)
        manifest["status"] = "completed"
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        manifest["finished_at"] = now()
        save(root / "manifest.json", manifest)
    return root


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--agent", choices=["researcher", "decomposer"], default="decomposer")
    select = cli.add_mutually_exclusive_group()
    select.add_argument("--tasks", nargs="+")
    select.add_argument("--all", action="store_true")
    cli.add_argument("--data", type=Path, default=REPO_ROOT / "artifacts/gyms/browsecomp/tasks.jsonl")
    cli.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/gyms/browsecomp")
    cli.add_argument("--retrieval-url", default="http://127.0.0.1:8000")
    cli.add_argument("--concurrency", type=int, default=2)
    cli.add_argument("-n", "--repetitions", type=int, default=1)
    cli.add_argument("--timeout", type=float, default=2700)
    cli.add_argument("--episode", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--repetition", type=int, default=1, help=argparse.SUPPRESS)
    return cli


if __name__ == "__main__":
    args = parser().parse_args()
    asyncio.run(episode(args) if args.episode else main(args))
