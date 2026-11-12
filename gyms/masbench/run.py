"""Run MASBench with a dedicated Agent Server for each episode."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from uuid import uuid4

import httpx
from decomposer.agent_server import agent_server
from gyms.agent_server import invoke_and_capture
from gyms.masbench import AXES, DATASET, REVISION, REPO_ROOT, task_input
from gyms.masbench.agents import AGENTS
from gyms.masbench.evaluate import evaluate, final_output
from gyms.usage import build_usage_summary


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


async def episode(args):
    root = args.output_dir.resolve()
    task = json.loads((root / "task.json").read_text())
    os.environ["MASBENCH_MODEL_LOG"] = str(root / "model_calls.jsonl")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    os.environ["MASBENCH_AGENT_URL"] = f"http://127.0.0.1:{port}"
    started = now()
    state, error = {}, None
    inputs = task_input(task)
    save(root / "input.json", inputs)
    try:
        async with agent_server(Path(__file__).with_name("langgraph.json"), port=port,
                                startup_timeout=120, n_jobs_per_worker=64):
            state, error = await invoke_and_capture(
                os.environ["MASBENCH_AGENT_URL"], args.agent, inputs,
                timeout=args.timeout, config={"recursion_limit": 410},
            )
    except Exception as exc:
        error = exc
    result = {
        "task_id": task["task_id"], "axis": task["axis"], "value": task["value"],
        "agent": args.agent, "repetition": args.repetition,
        "started_at": started, "finished_at": now(),
        "status": "finished" if error is None else "timeout" if isinstance(error, TimeoutError) else "error",
        "error": repr(error) if error is not None else None,
    }
    trace = {**state, **result, "model": AGENTS[args.agent]}
    save(root / "trace.json", trace)
    save(root / "usage.json", build_usage_summary(
        state.get("messages", []), state.get("agent_runs", {}), state.get("agents", {}),
    ))
    output = final_output(state.get("messages", []))
    (root / "answer.txt").write_text(output)
    result["evaluation"] = evaluate(output, task["ground_truth"])
    if error is not None:
        result["evaluation"].update(passed=False, score=0)
    save(root / "result.json", result)
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        raise error


async def preflight(agents):
    models = {os.environ["MASBENCH_BASE_URL"]: "Qwen/Qwen3.5-4B"}
    if "sft_manager" in agents:
        models[os.environ["MASBENCH_SFT_URL"]] = "decomposer-manager-qwen35-4b-sft-1.5.0"
    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
        for url, model in models.items():
            response = await client.get(url.rstrip("/") + "/models")
            response.raise_for_status()
            if model not in {row["id"] for row in response.json()["data"]}:
                raise ValueError(f"Model {model} is not served at {url}")


async def main(args):
    if args.concurrency < 1 or args.repetitions < 1 or args.timeout <= 0:
        raise ValueError("Concurrency, repetitions and timeout must be positive")
    tasks = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if not tasks or len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Tasks must be nonempty and have unique IDs")
    if args.smoke:
        tasks = [next(task for task in tasks if task["axis"] == axis) for axis in AXES]
    elif not args.all:
        raise ValueError("Choose --smoke or --all")
    agents = args.agents
    if len(set(agents)) != len(agents):
        raise ValueError("Agents must be unique")
    await preflight(agents)
    root = args.output_dir.resolve()
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "started_at": now(), "agents": agents,
        "dataset": DATASET, "dataset_revision": REVISION,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "tasks": [t["task_id"] for t in tasks], "repetitions": args.repetitions,
        "timeout_seconds": args.timeout, "concurrency": args.concurrency,
        "models": AGENTS, "base_url": os.environ["MASBENCH_BASE_URL"],
        "sft_url": os.environ["MASBENCH_SFT_URL"], "episodes": [],
        "packages": {name: version(name) for name in
                     ("langchain", "langgraph", "langgraph-sdk", "math-verify", "datasets")},
    }
    if args.resume:
        previous = json.loads((root / "manifest.json").read_text())
        if previous["status"] not in {"interrupted", "needs_attention"}:
            raise ValueError("Stop the previous runner before resuming")
        for key in ("agents", "dataset", "dataset_revision", "data_sha256", "tasks",
                    "repetitions", "timeout_seconds", "models", "base_url", "sft_url", "packages"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume must preserve {key}")
        previous.setdefault("resumptions", []).append({
            "started_at": now(), "code_revision": manifest["code_revision"],
            "previous_concurrency": previous["concurrency"], "concurrency": args.concurrency,
        })
        manifest = previous
        manifest.update(status="running", concurrency=args.concurrency)
        manifest.pop("finished_at", None)
    save(root / "manifest.json", manifest)
    semaphore = asyncio.Semaphore(args.concurrency)
    stop = asyncio.Event()
    consecutive_errors = 0

    async def launch(task, agent, repetition):
        nonlocal consecutive_errors
        async with semaphore:
            if stop.is_set():
                return
            directory = root / agent / f"{task['task_id']}-r{repetition:03d}"
            if directory.exists():
                result_path = directory / "result.json"
                if result_path.exists():
                    entry = json.loads(result_path.read_text())
                    if entry.get("error") != "CancelledError()":
                        entry["directory"] = str(directory.relative_to(root))
                        manifest["episodes"].append(entry)
                        save(root / "manifest.json", manifest)
                        return
                archived = root / "interrupted" / agent
                archived.mkdir(parents=True, exist_ok=True)
                directory.rename(archived / f"{directory.name}-{uuid4().hex[:8]}")
            directory.mkdir(parents=True)
            save(directory / "task.json", task)
            command = [sys.executable, "-m", "gyms.masbench.run", "--episode", "--agent", agent,
                       "--output-dir", str(directory), "--timeout", str(args.timeout),
                       "--repetition", str(repetition)]
            env = {**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}"}
            with (directory / "runner.log").open("w") as log:
                process = await asyncio.create_subprocess_exec(*command, cwd=REPO_ROOT, env=env,
                                                               stdout=log, stderr=log)
                try:
                    await asyncio.wait_for(process.wait(), args.timeout + 300)
                except TimeoutError:
                    pass
                finally:
                    if process.returncode is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            await asyncio.wait_for(process.wait(), 60)
                        except TimeoutError:
                            process.kill()
                            await process.wait()
            result_path = directory / "result.json"
            entry = json.loads(result_path.read_text()) if result_path.exists() else {
                "task_id": task["task_id"], "axis": task["axis"], "value": task["value"],
                "agent": agent, "repetition": repetition, "status": "error",
                "error": f"Episode exited without result.json: {process.returncode}",
                "evaluation": {"passed": False, "score": 0},
            }
            entry["directory"] = str(directory.relative_to(root))
            manifest["episodes"].append(entry)
            consecutive_errors = consecutive_errors + 1 if entry["status"] == "error" else 0
            if consecutive_errors >= 3:
                stop.set()
            save(root / "manifest.json", manifest)
            print(f"{len(manifest['episodes'])}/{len(tasks) * len(agents) * args.repetitions} "
                  f"{agent} {task['task_id']}: {entry['status']} score={entry['evaluation']['score']}", flush=True)

    completed = {(row["task_id"], row["agent"], row["repetition"]) for row in manifest["episodes"]}
    jobs = [asyncio.create_task(launch(task, agent, rep))
            for task in tasks for rep in range(1, args.repetitions + 1) for agent in agents
            if (task["task_id"], agent, rep) not in completed]
    try:
        await asyncio.gather(*jobs)
        manifest["status"] = "needs_attention" if stop.is_set() or any(
            entry["status"] == "error" for entry in manifest["episodes"]
        ) else "completed"
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        manifest["finished_at"] = now()
        save(root / "manifest.json", manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(f"Run {manifest['status']}; inspect {root}")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    select = cli.add_mutually_exclusive_group()
    select.add_argument("--smoke", action="store_true", help="First task in each axis")
    select.add_argument("--all", action="store_true")
    cli.add_argument("--agents", choices=list(AGENTS), nargs="+", default=list(AGENTS))
    cli.add_argument("--data", type=Path)
    cli.add_argument("--output-dir", type=Path, required=True, help="New run directory (must not exist)")
    cli.add_argument("--concurrency", type=int, default=12)
    cli.add_argument("--resume", action="store_true", help="Resume a stopped run in --output-dir")
    cli.add_argument("--repetitions", type=int, default=1)
    cli.add_argument("--timeout", type=float, default=1800)
    cli.add_argument("--episode", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--agent", choices=list(AGENTS), help=argparse.SUPPRESS)
    cli.add_argument("--repetition", type=int, default=1, help=argparse.SUPPRESS)
    return cli


if __name__ == "__main__":
    args = parser().parse_args()
    asyncio.run(episode(args) if args.episode else main(args))
