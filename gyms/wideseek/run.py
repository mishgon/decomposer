"""Run WideSeek episodes and save raw trajectories and native scores."""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from uuid import uuid4
from datetime import datetime, timezone

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk import get_client
from langchain_core.messages import message_to_dict
from decomposer.visualization import write_trace_html
from decomposer.usage import build_usage_summary

from gyms.wideseek import REPO_ROOT, agents
from gyms.wideseek.evaluate import evaluate
from gyms.wideseek.metrics import subagent_counts
from gyms.wideseek.prepare import agent_input
from gyms.wideseek.runtime import (BudgetExceeded, close_model, init_budget, model, model_metadata, save)


def usage(path):
    """Provider usage and wall time, with judge costs separate from agent costs."""
    roles = {}
    rows = [json.loads(file.read_text()) for file in (path / "judge_calls").glob("*.json")]
    log = path / "model_calls.jsonl"
    if log.exists():
        calls = {}
        for line in log.read_text().splitlines():
            record = json.loads(line)
            calls.setdefault(record["call_id"], {}).update(record)
        rows.extend(calls.values())
    for row in rows:
        role = row.get("role", "judge")
        totals = roles.setdefault(role, dict(calls=0, errors=0, unfinished=0, input_tokens=0,
            output_tokens=0, reasoning_tokens=0, call_seconds=0., missing_usage=0))
        totals["calls"] += 1
        totals["errors"] += int("error" in row)
        totals["unfinished"] += int(row.get("status") == "started")
        totals["call_seconds"] += (row.get("duration_seconds", 0) if "call_id" in row else
            row.get("finished_at", row["started_at"]) - row["started_at"])
        responses = row.get("response", [])
        if isinstance(responses, dict):
            responses = [responses]
        responses = [r.get("data", r) for r in responses]
        totals["missing_usage"] += int(not any(r.get("usage_metadata") for r in responses))
        for response in responses:
            meta = response.get("usage_metadata") or {}
            for field in ("input_tokens", "output_tokens"):
                totals[field] += meta.get(field, 0)
            totals["reasoning_tokens"] += (meta.get("output_token_details") or {}).get("reasoning", 0)
    return roles


async def cleanup_workers(client, state, path):
    """Discover orphan runs, stop workers, then archive and release each thread."""
    async def list_runs(thread_id):
        runs = []
        while True:
            page = await asyncio.wait_for(client.runs.list(thread_id, limit=100, offset=len(runs)), 30)
            runs.extend(page)
            if len(page) < 100:
                return runs

    errors = []
    thread_ids = dict.fromkeys(s["thread_id"] for s in state.get("agents", {}).values())
    for thread_id in thread_ids:
        try:
            # A runs.create response can be lost before its ID reaches parent state.
            runs = await list_runs(thread_id)
            for run in runs:
                if run["status"] in {"pending", "running"}:
                    await asyncio.wait_for(client.runs.cancel(thread_id, run["run_id"], wait=True), 30)
            runs = await list_runs(thread_id)
            if any(r["status"] in {"pending", "running"} for r in runs):
                raise RuntimeError("Worker still active after cancellation")
            worker_state = await asyncio.wait_for(client.threads.get_state(thread_id), 30)
            save(path / "subagents" / f"{thread_id}.json", {**worker_state, "runs": runs})
            # Never discard remote checkpoints unless the archive was saved successfully.
            await asyncio.wait_for(client.threads.delete(thread_id), 30)
        except Exception as exc:
            errors.append(f"{thread_id}: {type(exc).__name__}: {exc}")
    return errors


async def episode(task, mode, attempt, root, args):
    attempt_path = root / mode / task["task_id"] / f"attempt-{attempt:03d}"
    if (attempt_path / "result.json").exists():
        return
    # Never reuse a crashed execution's path: surviving workers retain that context.
    path = attempt_path / f"execution-{uuid4().hex}"
    init_budget(path, args.model_calls, args.output_tokens)
    checkpoint = InMemorySaver()
    client = get_client(url=args.worker_url)
    policy = model(agents.AGENT_MODELS[args.agent])
    if args.agent == "decomposer":
        agent = agents.decomposer(policy, checkpoint, args.worker_url)
    else:
        agent = agents.react(policy, checkpoint)
    config = {"recursion_limit": 410, "configurable": {"thread_id": uuid4().hex}}
    result = {"task_id": task["task_id"], "task": task["task_id"],
              "episode_id": path.name, "run_id": root.name,
              "repetition": attempt, "mode": mode, "attempt": attempt,
              "execution_directory": path.name, "started_at": time.time(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()}
    state = {}
    try:
        state = await asyncio.wait_for(agent.ainvoke(agent_input(task), config=config,
                               context={"directory": str(path.resolve())}), args.timeout)
        result["status"] = "finished"
    except Exception as exc:
        result["status"] = ("timeout" if isinstance(exc, TimeoutError) else
                            "budget_exceeded" if isinstance(exc, BudgetExceeded) else
                            "context_exceeded" if type(exc).__name__ == "OpenAIContextOverflowError" else "error")
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        snapshot = await agent.aget_state(config)
        state = state or dict(snapshot.values)
        if errors := await cleanup_workers(client, state, path):
            result["cleanup_errors"] = errors
        await close_model(policy)
        result["agent_finished_at"] = time.time()
        messages = state.get("messages", [])
        trace = {**state, "episode_id": result["episode_id"], "run_id": root.name,
                 "thread_id": config["configurable"]["thread_id"],
                 "task": task["task_id"], "repetition": attempt, "attempt": attempt,
                 "harness": args.agent, "assistant_id": args.agent,
                 "model": agents.AGENT_MODELS[args.agent],
                 "started_at": datetime.fromtimestamp(result["started_at"], timezone.utc).isoformat(),
                 "finished_at": datetime.fromtimestamp(result["agent_finished_at"], timezone.utc).isoformat(),
                 "agent_error": result.get("error"),
                 "agents": state.get("agents", {}), "agent_runs": state.get("agent_runs", {}),
                 "decomposer_agent_runs": state.get("decomposer_agent_runs", []),
                 "messages": [message_to_dict(m) for m in messages]}
        save(path / "trace.json", trace)
        write_trace_html(trace, path / "trace.html")
        result["subagent_statistics"] = subagent_counts(trace["messages"])
    answer = messages[-1].content if messages and messages[-1].type == "ai" and not messages[-1].tool_calls else ""
    result["answer"] = answer
    try:
        result["evaluation"] = await asyncio.wait_for(evaluate(task, answer, path,
            judge_model_id=agents.JUDGE_MODEL), timeout=600)
    except Exception as exc:
        result["evaluation"] = {"status": "evaluation_error", "score": None,
                                "error": f"{type(exc).__name__}: {exc}"}
    result["finished_at"] = time.time()
    result["usage"] = usage(path)
    trace_usage = build_usage_summary(trace["messages"], trace["agent_runs"], trace["agents"])
    if args.agent == "react":
        trace_usage["react"] = trace_usage.pop("decomposer")
    save(path / "usage.json", trace_usage)
    result["trace_path"] = str((path / "trace.json").relative_to(root))
    save(attempt_path / "result.json", result)
    print(json.dumps({k: result[k] for k in ("mode", "task_id", "attempt", "status", "evaluation")}), flush=True)


async def prepare_run(args):
    """Validate services and create or resume a raw run, without scheduling tasks."""
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    root = args.output.resolve()
    raw = args.data.read_bytes()
    tasks = [json.loads(line) for line in raw.splitlines()][:args.limit]
    if not tasks:
        raise ValueError("Choose a nonempty task set")
    if args.mode not in {"simple", "decomposer"}:
        raise ValueError("Each run must use one setup: simple or decomposer")
    modes = [args.mode]
    async with httpx.AsyncClient(timeout=30, trust_env=False) as http:
        response = await http.get(os.environ.get("WS_SEARCH_URL", "http://127.0.0.1:18080") + "/health")
        response.raise_for_status()
        retrieval = response.json()
        if "decomposer" in modes:
            response = await http.get(args.worker_url + "/ok")
            response.raise_for_status()
    profiles = {}
    for role, profile in (("agent", agents.AGENT_MODELS[args.agent]),
                          ("subagent", agents.RESEARCHER_MODEL), ("judge", agents.JUDGE_MODEL)):
        policy = model(profile)
        try:
            profiles[role] = {"profile": profile, **model_metadata(policy)}
        finally:
            await close_model(policy)
    settings = {"tasks": [t["task_id"] for t in tasks], "data_sha256": hashlib.sha256(raw).hexdigest(),
                "agent": args.agent, "modes": modes, "repetitions": args.n, "concurrency": args.concurrency,
                "model": profiles["agent"]["model_name"], "model_profiles": profiles,
                "generation": profiles["agent"], "recursion_limit": 410,
                "retrieval": retrieval,
                "model_calls": args.model_calls, "output_tokens": args.output_tokens,
                "timeout": args.timeout,
                "judge": {"model": profiles["judge"]["model_name"], "profile": agents.JUDGE_MODEL,
                          "temperature": 0., "thinking": False, "paper_comparable": False}}
    sources = {}
    for base in (REPO_ROOT / "gyms/wideseek", REPO_ROOT / "src/decomposer"):
        for folder, directories, files in os.walk(base):
            directories[:] = [name for name in directories if name not in {".venv", "__pycache__"}]
            for name in files:
                path = Path(folder) / name
                if path.suffix in {".py", ".sh", ".json", ".txt"}:
                    sources[str(path.relative_to(REPO_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.resume:
        previous = json.loads((root / "manifest.json").read_text())
        if previous["settings"] != settings:
            raise ValueError("Resume settings differ from the saved run")
        if previous["source_sha256"] != sources:
            if not args.allow_source_change:
                raise ValueError("Resume source code differs; use --allow-source-change for a Gym-only migration")
            original_core = {p: h for p, h in previous["source_sha256"].items() if p.startswith("src/decomposer/")}
            current_core = {p: h for p, h in sources.items() if p.startswith("src/decomposer/")}
            if original_core != current_core:
                raise ValueError("Cannot migrate a run between different Decomposer harness versions")
            previous.setdefault("source_history", []).append({
                "previous_source_sha256": previous["source_sha256"], "changed_at": time.time(),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()})
            previous["source_sha256"] = sources
            save(root / "manifest.json", previous)
    else:
        root.mkdir(parents=True, exist_ok=False)
        save(root / "manifest.json", {"run_id": root.name, "status": "running",
             "harness": args.agent, "assistant_id": args.agent,
             "tasks": settings["tasks"], "repetitions": args.n, "episodes": [],
             "settings": settings, "started_at": time.time(),
             "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
             "packages": {p: importlib.metadata.version(p) for p in
                 ("langchain", "langchain-openai", "langgraph", "langgraph-api", "httpx", "pandas")},
             "source_sha256": sources})
        (root / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO_ROOT))
    return tasks, root


async def run_jobs(tasks, jobs, root, args):
    """Run explicit (task ID, attempt number) pairs at bounded concurrency."""
    by_id = {task["task_id"]: task for task in tasks}
    semaphore = asyncio.Semaphore(args.concurrency)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    save(manifest_path, manifest)

    async def bounded(task_id, attempt):
        async with semaphore:
            await episode(by_id[task_id], args.mode, attempt, root, args)
            result_path = root / args.mode / task_id / f"attempt-{attempt:03d}" / "result.json"
            result = json.loads(result_path.read_text())
            episodes = manifest.setdefault("episodes", [])
            episodes[:] = [e for e in episodes if (e["task_id"], e["attempt"]) != (task_id, attempt)]
            episodes.append({key: result[key] for key in (
                "episode_id", "task_id", "task", "attempt", "repetition", "status",
                "started_at", "finished_at", "trace_path", "evaluation")})
            save(manifest_path, manifest)

    try:
        await asyncio.gather(*(bounded(task, attempt) for task, attempt in jobs))
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    else:
        manifest["status"] = "completed"
    finally:
        save(manifest_path, manifest)


async def main(args):
    tasks, root = await prepare_run(args)
    await run_jobs(tasks, ((t["task_id"], n) for n in range(1, args.n + 1) for t in tasks), root, args)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "artifacts/gyms/wideseek/data/width/tasks.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", choices=agents.AGENT_MODELS, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("-n", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--model-calls", type=int, default=None, help="Optional shared call cap; default uncapped")
    parser.add_argument("--output-tokens", type=int, default=None, help="Optional shared token cap; default uncapped")
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--worker-url", default="http://127.0.0.1:18081")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-source-change", action="store_true",
                        help="Record an explicit Gym-source migration on resume; core/model/data must match")
    return parser


def cli(run=main, argv=None, *, parser=None):
    parser = parser or create_parser()
    args = parser.parse_args(argv)
    args.mode = "simple" if args.agent == "react" else "decomposer"
    if args.allow_source_change and not args.resume:
        parser.error("--allow-source-change requires --resume")
    if min(v for v in (args.limit, args.n, args.concurrency, args.model_calls, args.output_tokens, args.timeout) if v is not None) < 1:
        parser.error("Counts and budgets must be positive")
    # Lock outside the run directory so first-launch mkdir remains exclusive.
    root = args.output.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args))


if __name__ == "__main__":
    cli()
