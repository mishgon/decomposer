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

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk import get_client

from decomposer.core import create_decomposer_agent
from gyms.wideseek.evaluate import evaluate
from gyms.wideseek.metrics import subagent_counts
from gyms.wideseek.prepare import agent_input
from gyms.wideseek.runtime import (BudgetExceeded, Context, DEFAULT_MODEL, DEFAULT_SUBAGENT,
    DEFAULT_TEACHER, MODEL_PROFILES, ModelLog, close_model, init_budget, model, model_metadata, save)


def usage(path):
    """Provider usage and wall time, with judge costs separate from agent costs."""
    roles = {}
    for folder in ("model_calls", "judge_calls"):
        for file in (path / folder).glob("*.json"):
            row = json.loads(file.read_text())
            role = row.get("role", "judge")
            totals = roles.setdefault(role, dict(calls=0, errors=0, input_tokens=0,
                output_tokens=0, reasoning_tokens=0, call_seconds=0., missing_usage=0))
            totals["calls"] += 1
            totals["errors"] += int("error" in row)
            totals["call_seconds"] += row.get("finished_at", row["started_at"]) - row["started_at"]
            responses = row.get("responses", [row["response"]] if "response" in row else [])
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
    thread_ids = dict.fromkeys(s["thread_id"] for s in state.get("subagents", {}).values())
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
    policy = model(getattr(args, "model", DEFAULT_MODEL))
    subagent = getattr(args, "subagent_model", DEFAULT_MODEL)
    if mode == "decomposer":
        agent = create_decomposer_agent(decomposer_model=policy,
            subagent_types=[{"subagent_type_id": subagent,
              "description": "Qwen3.5-4B unlooped researcher with offline Wiki-2018 search and access tools.",
              "assistant_id": subagent, "url": args.worker_url}],
            checkpointer=checkpoint, middleware=[ModelLog("decomposer")],
            context_schema=Context, subagent_recursion_limit=410)
    else:
        # Construct the same researcher graph with a checkpoint for interrupted traces.
        from langchain.agents import create_agent
        from gyms.wideseek.worker import search, access, SYSTEM_PROMPT
        agent = create_agent(policy, tools=[search, access], system_prompt=SYSTEM_PROMPT,
            context_schema=Context, middleware=[ModelLog("researcher")], checkpointer=checkpoint)
    config = {"recursion_limit": 410, "configurable": {"thread_id": uuid4().hex}}
    result = {"task_id": task["task_id"], "mode": mode, "attempt": attempt,
              "execution_directory": path.name, "started_at": time.time(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
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
        trace = {**state, "messages": [m.model_dump(mode="json") for m in messages]}
        save(path / "trace.json", trace)
        result["subagent_statistics"] = subagent_counts(trace["messages"])
    answer = messages[-1].content if messages and messages[-1].type == "ai" and not messages[-1].tool_calls else ""
    result["answer"] = answer
    try:
        result["evaluation"] = await asyncio.wait_for(evaluate(task, answer, path,
            judge_model_id=getattr(args, "judge_model", DEFAULT_TEACHER)), timeout=600)
    except Exception as exc:
        result["evaluation"] = {"status": "evaluation_error", "score": None,
                                "error": f"{type(exc).__name__}: {exc}"}
    result["finished_at"] = time.time()
    result["usage"] = usage(path)
    save(attempt_path / "result.json", result)
    print(json.dumps({k: result[k] for k in ("mode", "task_id", "attempt", "status", "evaluation")}), flush=True)


async def prepare_run(args):
    """Validate services and create or resume a raw run, without scheduling tasks."""
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    root = args.output.resolve()
    os.environ.setdefault("WS_ARTIFACT_ROOT", str(Path("artifacts").resolve()))
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
    for role, profile in (("agent", args.model), ("subagent", args.subagent_model), ("judge", args.judge_model)):
        policy = model(profile)
        try:
            profiles[role] = {"profile": profile, **model_metadata(policy)}
        finally:
            await close_model(policy)
    settings = {"tasks": [t["task_id"] for t in tasks], "data_sha256": hashlib.sha256(raw).hexdigest(),
                "modes": modes, "repetitions": args.n, "concurrency": args.concurrency,
                "model": profiles["agent"]["model_name"], "model_profiles": profiles,
                "generation": profiles["agent"], "recursion_limit": 410,
                "retrieval": retrieval,
                "model_calls": args.model_calls, "output_tokens": args.output_tokens,
                "timeout": args.timeout,
                "judge": {"model": profiles["judge"]["model_name"], "profile": args.judge_model,
                          "temperature": 0., "thinking": False, "paper_comparable": False}}
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for base in (Path("gyms/wideseek"), Path("src/decomposer"))
               for p in base.rglob("*") if p.is_file() and p.suffix in {".py", ".sh", ".json", ".txt"}}
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
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
            previous["source_sha256"] = sources
            save(root / "manifest.json", previous)
    else:
        root.mkdir(parents=True, exist_ok=False)
        save(root / "manifest.json", {"settings": settings, "started_at": time.time(),
             "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             "packages": {p: importlib.metadata.version(p) for p in
                 ("langchain", "langchain-openai", "langgraph", "langgraph-api", "httpx", "pandas")},
             "source_sha256": sources})
        (root / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    return tasks, root


async def run_jobs(tasks, jobs, root, args):
    """Run explicit (task ID, attempt number) pairs at bounded concurrency."""
    by_id = {task["task_id"]: task for task in tasks}
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(task_id, attempt):
        async with semaphore:
            await episode(by_id[task_id], args.mode, attempt, root, args)

    await asyncio.gather(*(bounded(task, attempt) for task, attempt in jobs))


async def main(args):
    tasks, root = await prepare_run(args)
    await run_jobs(tasks, ((t["task_id"], n) for n in range(1, args.n + 1) for t in tasks), root, args)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/gyms/wideseek/data/width/tasks.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    harness = parser.add_mutually_exclusive_group(required=True)
    harness.add_argument("--harness", choices=["react", "decomposer"])
    harness.add_argument("--mode", choices=["simple", "decomposer"], help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("-n", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--model-calls", type=int, default=None, help="Optional shared call cap; default uncapped")
    parser.add_argument("--output-tokens", type=int, default=None, help="Optional shared token cap; default uncapped")
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--worker-url", default="http://127.0.0.1:18081")
    parser.add_argument("--model", choices=MODEL_PROFILES, default=DEFAULT_MODEL)
    parser.add_argument("--subagent-model", choices=(DEFAULT_MODEL, DEFAULT_SUBAGENT), default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", choices=MODEL_PROFILES, default=DEFAULT_TEACHER)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-source-change", action="store_true",
                        help="Record an explicit Gym-source migration on resume; core/model/data must match")
    return parser


def cli(run=main, argv=None, *, parser=None):
    parser = parser or create_parser()
    args = parser.parse_args(argv)
    if args.harness is not None:
        args.mode = "simple" if args.harness == "react" else "decomposer"
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
