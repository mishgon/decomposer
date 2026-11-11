"""Run isolated BrowseComp episodes and save their raw results."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import hashlib
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
from gyms.usage import build_usage_summary
from gyms.infrastructure import failures, classify
from decomposer.visualization import write_trace_html
from gyms.browsecomp import REPO_ROOT
from gyms.browsecomp.agents import AGENTS, RESEARCHER_MODEL, WORKER_BUDGET_SECONDS, WORKER_GRACE_SECONDS
from gyms.browsecomp.evaluate import FINAL_FORMAT, evaluate
from gyms.wideseek.runtime import init_budget
from gyms.wideseek.assets import CORPUS_REVISION


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


async def episode(args, *, server_config=None, final_format=FINAL_FORMAT, evaluator=evaluate):
    """Runs in its own process so server addresses and log paths cannot collide."""
    model_id, harness = AGENTS[args.agent]
    root = args.output_dir.resolve()
    task = json.loads((root / "task.json").read_text())
    os.environ["WS_SEARCH_URL"] = args.retrieval_url.rstrip("/")
    runtime_directory = root / "runtime"
    init_budget(runtime_directory)
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
        async with agent_server(server_config or Path(__file__).with_name("langgraph.json"), port=port):
            state, error = await invoke_and_capture(
                os.environ["BROWSECOMP_AGENT_URL"], args.agent,
                {"messages": [{"role": "user", "content": task["question"] + final_format}]},
                timeout=args.timeout, config={"recursion_limit": 410},
                context={"directory": str(runtime_directory)},
            )
    except Exception as exc:
        error = exc
    result["agent_finished_at"] = now()
    result["status"] = "finished" if error is None else "timeout" if isinstance(error, TimeoutError) else "error"
    result["error"] = repr(error) if error else None
    trace = {**state, **result, "harness": harness,
             "agent_error": result["error"], "finished_at": result["agent_finished_at"],
             "model": model_id,
             "messages": state.get("messages", []), "agents": state.get("agents", {}),
             "agent_runs": state.get("agent_runs", {})}
    save(root / "trace.json", trace)
    write_trace_html(trace, root / "trace.html")
    usage = build_usage_summary(trace["messages"], trace["agent_runs"], trace["agents"])
    if harness == "react":
        usage["react"] = usage.pop("decomposer")
    save(root / "usage.json", usage)
    last = trace["messages"][-1] if trace["messages"] else {}
    last = last.get("data", last)
    output = last.get("content", "") if last.get("type") == "ai" and not last.get("tool_calls") else ""
    if not isinstance(output, str):
        output = "\n".join(block.get("text", "") for block in output if isinstance(block, dict))
    try:
        result["evaluation"] = await asyncio.wait_for(evaluator(task, output, root), 180)
    except Exception as exc:
        result["evaluation"] = {"status": "evaluation_error", "score": None, "passed": False, "error": repr(exc)}
    result["finished_at"] = now()
    save(root / "result.json", result)
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        raise error


async def preflight(url):
    """Require the same pinned Wiki-2018 corpus used by WideSeek-R1."""
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        response = await client.get(url.rstrip("/") + "/health")
        response.raise_for_status()
        if response.json().get("corpus_revision") != CORPUS_REVISION:
            raise ValueError("Expected the pinned WideSeek-R1 Wiki-2018 corpus")
        response = await client.post(url.rstrip("/") + "/retrieve",
                                    json={"queries": ["Albert Einstein"], "topk": 1})
        response.raise_for_status()
        hits = response.json()["result"][0]
        if not hits:
            raise RuntimeError("Wiki-2018 preflight returned no documents")
        response = await client.post(url.rstrip("/") + "/access", json={"urls": [hits[0]["url"]]})
        response.raise_for_status()
        if not response.json()["result"][0]:
            raise RuntimeError("Wiki-2018 preflight could not access its search result")


async def main(args, *, episode_module="gyms.browsecomp.run", preflight_check=None):
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
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    provenance = {"revision": revision, "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  "backend": args.retrieval_url, "corpus_revision": CORPUS_REVISION, "model": AGENTS[args.agent][0],
                  "worker_model": RESEARCHER_MODEL if AGENTS[args.agent][1] == "decomposer" else None,
                  "worker_budget_seconds": WORKER_BUDGET_SECONDS if AGENTS[args.agent][1] == "decomposer" else None,
                  "worker_grace_seconds": WORKER_GRACE_SECONDS if AGENTS[args.agent][1] == "decomposer" else None}
    await (preflight_check or preflight)(args.retrieval_url)
    if args.resume:
        root = args.resume.resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["status"] not in {"interrupted", "needs_attention"}:
            raise ValueError("Resume requires an interrupted run; stop its runner first")
        if (manifest.get("provenance") != provenance
                or manifest.get("episode_module", "gyms.browsecomp.run") != episode_module
                or manifest["tasks"] != [t["task_id"] for t in tasks]
                or manifest["repetitions"] != args.repetitions
                or manifest["assistant_id"] != args.agent
                or any(manifest["config"][key] != config[key]
                       for key in ("data", "retrieval_url", "timeout"))):
            raise ValueError("Resume must keep the same tasks, repetitions, agent, data, retrieval and timeout")
        manifest.setdefault("resumptions", []).append({
            "started_at": now(), "concurrency": args.concurrency,
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        })
        manifest["status"] = "running"
        manifest.pop("finished_at", None)
        manifest["config"]["concurrency"] = args.concurrency
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        root = args.output_dir.resolve() / run_id
        root.mkdir(parents=True)
        manifest = {"run_id": run_id, "status": "running", "started_at": now(),
                    "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
                    "episode_module": episode_module,
                    "assistant_id": args.agent, "tasks": [t["task_id"] for t in tasks],
                    "repetitions": args.repetitions, "episodes": [],
                    "config": config, "provenance": provenance}
    save(root / "manifest.json", manifest)
    print(f"Artifacts: {root}", flush=True)
    retry = [row for row in manifest["episodes"] if row.get("infrastructure_errors")]
    if retry:
        manifest.setdefault("archived_attempts", []).extend(retry)
        manifest["episodes"] = [row for row in manifest["episodes"] if not row.get("infrastructure_errors")]
        save(root / "manifest.json", manifest)
    completed = {(row["task"], row["repetition"]) for row in manifest["episodes"]}
    stop = asyncio.Event()
    consecutive_errors = 0
    semaphore = asyncio.Semaphore(args.concurrency)

    async def launch(task, repetition):
        nonlocal consecutive_errors
        async with semaphore:
            if stop.is_set():
                return
            directory = root / "traces" / f"{task['task_id']}-r{repetition:03d}"
            if directory.exists():
                archived = root / "interrupted"
                archived.mkdir(exist_ok=True)
                destination = archived / f"{directory.name}-{uuid4().hex[:8]}"
                directory.rename(destination)
                for old in manifest.get("archived_attempts", []):
                    if old.get("directory") == str(directory.relative_to(root)):
                        old["directory"] = str(destination.relative_to(root))
                save(root / "manifest.json", manifest)
            directory.mkdir(parents=True)
            save(directory / "task.json", task)
            command = [sys.executable, "-m", episode_module, "--episode",
                       "--agent", args.agent, "--output-dir", str(directory),
                       "--retrieval-url", args.retrieval_url, "--timeout", str(args.timeout),
                       "--repetition", str(repetition)]
            env = {**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}"}
            with (directory / "runner.log").open("w") as log:
                process = await asyncio.create_subprocess_exec(*command, cwd=REPO_ROOT, env=env,
                                                               stdout=log, stderr=log)
                try:
                    deadline = asyncio.get_running_loop().time() + args.timeout + 360
                    while True:
                        try:
                            code = await asyncio.wait_for(process.wait(), 5)
                            break
                        except TimeoutError:
                            detected = await asyncio.to_thread(failures, directory)
                            if any(e["kind"] == "credentials_or_quota" for e in detected):
                                stop.set()
                            if detected or stop.is_set():
                                code = -1
                                break
                            if asyncio.get_running_loop().time() >= deadline:
                                raise
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
            detected = await asyncio.to_thread(failures, directory)
            if not (directory / "result.json").exists():
                detected.append({"kind": "service_unavailable", "source": "runner", "error": "Episode exited without result.json"})
            if stop.is_set() and entry["status"] != "finished" and not detected:
                detected.append({"kind": "interrupted_by_queue", "source": "runner", "error": "Queue stopped after infrastructure failure"})
            if detected:
                entry["infrastructure_errors"] = detected
            judge_failure = classify(entry.get("evaluation", {}).get("error"))
            consecutive_errors = consecutive_errors + 1 if detected or judge_failure else 0
            if consecutive_errors >= 3 or judge_failure == "credentials_or_quota" or any(
                e["kind"] == "credentials_or_quota" for e in detected
            ):
                stop.set()
            manifest["episodes"].append(entry)
            save(root / "manifest.json", manifest)
            print(f"{len(manifest['episodes'])}/{len(tasks) * args.repetitions} {task['task_id']}: {entry['status']}", flush=True)

    jobs = [asyncio.create_task(launch(task, rep)) for task in tasks for rep in range(1, args.repetitions + 1)
            if (task["task_id"], rep) not in completed]
    try:
        await asyncio.gather(*jobs)
        manifest["status"] = "needs_attention" if stop.is_set() or any(
            e.get("infrastructure_errors") or e.get("evaluation", {}).get("status") == "evaluation_error"
            for e in manifest["episodes"]
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
    return root


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--agent", choices=list(AGENTS), default="decomposer")
    select = cli.add_mutually_exclusive_group()
    select.add_argument("--tasks", nargs="+")
    select.add_argument("--all", action="store_true")
    cli.add_argument("--data", type=Path, default=REPO_ROOT / "artifacts/gyms/browsecomp/tasks.jsonl")
    cli.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/gyms/browsecomp")
    cli.add_argument("--retrieval-url", default="http://127.0.0.1:18080")
    cli.add_argument("--resume", type=Path, help="Continue an interrupted run directory, keeping completed episodes")
    cli.add_argument("--concurrency", type=int, default=2)
    cli.add_argument("-n", "--repetitions", type=int, default=1)
    cli.add_argument("--timeout", type=float, default=2700)
    cli.add_argument("--episode", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--repetition", type=int, default=1, help=argparse.SUPPRESS)
    return cli


if __name__ == "__main__":
    args = parser().parse_args()
    asyncio.run(episode(args) if args.episode else main(args))
