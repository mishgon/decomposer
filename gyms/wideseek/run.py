"""Collect simple/decomposer trajectories and native width scores under one resumable run."""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import signal
import subprocess
import time
from uuid import uuid4

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk import get_client

from decomposer.core import create_decomposer_agent
from gyms.wideseek.evaluate import evaluate
from gyms.wideseek.prepare import agent_input
from gyms.wideseek.runtime import BudgetExceeded, Context, GENERATION, ModelLog, init_budget, model, save


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


async def episode(task, mode, attempt, root, args):
    attempt_path = root / mode / task["task_id"] / f"attempt-{attempt:03d}"
    if (attempt_path / "result.json").exists():
        return
    # Never reuse a crashed execution's path: surviving workers retain that context.
    path = attempt_path / f"execution-{uuid4().hex}"
    init_budget(path, args.model_calls, args.output_tokens)
    checkpoint = InMemorySaver()
    client = get_client(url=args.worker_url)
    policy = model()
    if mode == "decomposer":
        agent = create_decomposer_agent(policy,
            [{"subagent_type_id": "researcher", "description": "Qwen3.5-4B researcher with offline Wiki-2018 search and access tools.",
              "assistant_id": "researcher", "url": args.worker_url}],
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
              "execution_directory": path.name, "started_at": time.time()}
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
        for run in state.get("subagent_runs", {}).values():
            try:
                live = await asyncio.wait_for(client.runs.get(run["thread_id"], run["run_id"]), 30)
                if live["status"] in {"pending", "running"}:
                    await asyncio.wait_for(client.runs.cancel(run["thread_id"], run["run_id"], wait=True), 30)
                worker_state = await asyncio.wait_for(client.threads.get_state(run["thread_id"]), 30)
                save(path / "subagents" / f"{run['run_id']}.json", worker_state)
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(str(exc))
        await policy.http_async_client.aclose()
        result["agent_finished_at"] = time.time()
        messages = state.get("messages", [])
        save(path / "trace.json", {**state, "messages": [m.model_dump(mode="json") for m in messages]})
    answer = messages[-1].content if messages and messages[-1].type == "ai" and not messages[-1].tool_calls else ""
    result["answer"] = answer
    try:
        result["evaluation"] = await asyncio.wait_for(evaluate(task, answer, path), timeout=600)
    except Exception as exc:
        result["evaluation"] = {"status": "evaluation_error", "score": None,
                                "error": f"{type(exc).__name__}: {exc}"}
    result["finished_at"] = time.time()
    result["usage"] = usage(path)
    save(attempt_path / "result.json", result)
    print(json.dumps({k: result[k] for k in ("mode", "task_id", "attempt", "status", "evaluation")}), flush=True)


async def main(args):
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    root = args.output.resolve()
    os.environ.setdefault("WS_ARTIFACT_ROOT", str(root.parent))
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
    settings = {"tasks": [t["task_id"] for t in tasks], "data_sha256": hashlib.sha256(raw).hexdigest(),
                "modes": modes, "repetitions": args.n, "concurrency": args.concurrency,
                "model": os.environ.get("WS_MODEL", "Qwen/Qwen3.5-4B"),
                "generation": GENERATION, "recursion_limit": 410,
                "model_url": os.environ["LLM_PROXY_URL"], "retrieval": retrieval,
                "model_calls": args.model_calls, "output_tokens": args.output_tokens,
                "timeout": args.timeout,
                "judge": {"model": os.environ.get("WS_JUDGE_MODEL") or os.environ.get("WS_MODEL", "Qwen/Qwen3.5-4B"),
                          "temperature": 0., "thinking": False, "paper_comparable": False}}
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for base in (Path("gyms/wideseek"), Path("src/decomposer"))
               for p in base.rglob("*") if p.is_file() and p.suffix in {".py", ".sh", ".json", ".txt"}}
    if args.resume:
        previous = json.loads((root / "manifest.json").read_text())
        if previous["settings"] != settings or previous["source_sha256"] != sources:
            raise ValueError("Resume settings or source code differ from the saved run")
    else:
        root.mkdir(parents=True, exist_ok=False)
        save(root / "manifest.json", {"settings": settings, "started_at": time.time(),
             "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             "packages": {p: importlib.metadata.version(p) for p in
                 ("langchain", "langchain-openai", "langgraph", "langgraph-api", "httpx", "pandas")},
             "source_sha256": sources})
        (root / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(task, mode, attempt):
        async with semaphore:
            await episode(task, mode, attempt, root, args)

    await asyncio.gather(*(bounded(t, args.mode, n) for n in range(1, args.n + 1) for t in tasks))
    for mode in modes:
        rows = [json.loads(p.read_text()) for p in (root / mode).glob("*/attempt-???/result.json")]
        scores = [r["evaluation"]["score"] for r in rows if r["evaluation"]["score"] is not None]
        seconds = [r["agent_finished_at"] - r["started_at"] for r in rows]
        tokens = {field: sum(role[field] for r in rows for name, role in r["usage"].items() if name != "judge")
                  for field in ("input_tokens", "output_tokens")}
        save(root / f"{mode}-summary.json", {"attempts": len(rows), "scored": len(scores),
             "unscored": len(rows) - len(scores), "mean_native_score": sum(scores)/len(scores) if scores else None,
             "mean_native_score_infra_zero": sum(scores)/len(rows) if rows else None,
             "metrics": sorted({r["evaluation"].get("metric", "unscored") for r in rows}),
             "mean_agent_seconds": statistics.mean(seconds) if seconds else None,
             "median_agent_seconds": statistics.median(seconds) if seconds else None,
             "agent_tokens": tokens,
             "normal_finishes": sum(r["status"] == "finished" for r in rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/gyms/wideseek/data/width/tasks.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["simple", "decomposer"], required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("-n", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--model-calls", type=int, default=None, help="Optional shared call cap; default uncapped")
    parser.add_argument("--output-tokens", type=int, default=None, help="Optional shared token cap; default uncapped")
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--worker-url", default="http://127.0.0.1:18081")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(v for v in (args.limit, args.n, args.concurrency, args.model_calls, args.output_tokens, args.timeout) if v is not None) < 1:
        parser.error("Counts and budgets must be positive")
    # Lock outside the run directory so first-launch mkdir remains exclusive.
    root = args.output.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main(args))
