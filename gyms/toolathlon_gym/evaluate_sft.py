"""Resumable local SFT evaluation on Gym; no training and no OpenRouter."""
import argparse
import asyncio
import json
import subprocess
import time
from pathlib import Path

from langchain_core.messages import message_to_dict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph_sdk import get_client
from decomposer.chat_vllm import ChatVLLM
from decomposer.core import create_decomposer_agent
from gyms.toolathlon_gym.cancel import cancel_subagents
from gyms.toolathlon_gym.episode import Episode


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str))
    temporary.replace(path)


def metrics(manifest):
    n = manifest["config"]["repetitions"]
    tasks = {}
    for task in manifest["tasks"]:
        rows = [manifest["episodes"].get(f"{task}/rep-{rep:03d}", {}) for rep in range(1, n + 1)]
        passed = sum(r.get("passed") is True for r in rows)
        failed = sum(r.get("passed") is False for r in rows)
        tasks[task] = {"passed": passed, "failed": failed, "unknown_or_infrastructure": n - passed - failed}
    count = len(tasks)
    values = list(tasks.values())
    return {"tasks": tasks, "denominator_tasks": count,
            "all_attempts_scored": all(v["unknown_or_infrastructure"] == 0 for v in values),
            "pass@1_lower": sum(v["passed"] for v in values) / (count * n),
            "pass@1_upper": sum(v["passed"] + v["unknown_or_infrastructure"] for v in values) / (count * n),
            f"pass@{n}_lower": sum(v["passed"] > 0 for v in values) / count,
            f"pass@{n}_upper": sum(v["passed"] + v["unknown_or_infrastructure"] > 0 for v in values) / count,
            f"pass^{n}_lower": sum(v["passed"] == n for v in values) / count,
            f"pass^{n}_upper": sum(v["failed"] == 0 for v in values) / count}


async def episode(args, task, repetition, directory):
    env = Episode(task, directory, subagent_port=args.subagent_port, image=args.image)
    started = time.time()
    state, result = {}, {"task": task, "repetition": repetition, "started_at": started}
    startup = asyncio.create_task(asyncio.to_thread(env.start))
    try:
        try:
            await asyncio.shield(startup)
        except asyncio.CancelledError:
            await startup
            raise
        result["setup_seconds"] = time.time() - started
        model = ChatVLLM(model=args.model, base_url=args.url, api_key="EMPTY",
                        temperature=0.7, top_p=0.8, presence_penalty=1.5,
                        timeout=600, max_retries=2, preserve_reasoning=False,
                        parse_qwen_xml_tool_calls=True, disable_streaming=True,
                        extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                                    "chat_template_kwargs": {"enable_thinking": False}})
        agent = create_decomposer_agent(model, [{"subagent_type_id": "configured_non_thinking",
            "description": "Qwen3.5-4B non-thinking agent with all task tools.",
            "assistant_id": "configured_non_thinking", "url": env.url}],
            subagent_recursion_limit=410, checkpointer=InMemorySaver())
        config = {"recursion_limit": 410, "configurable": {"thread_id": directory.name}}
        try:
            state = await asyncio.wait_for(agent.ainvoke({"messages": [{"role": "user",
                "content": env.runtime["task_config"]["task_str"]}]}, config=config), args.agent_timeout)
            result["stop_reason"] = "finished"
        except (TimeoutError, GraphRecursionError) as error:
            result["stop_reason"] = type(error).__name__
        finally:
            state = dict((await agent.aget_state(config)).values)
        await cancel_subagents(get_client(url=env.url), state.get("subagent_runs", {}))
        evaluation = await asyncio.to_thread(env.score, require_partial=False)
        result.update(status="completed", passed=result["stop_reason"] == "finished" and evaluation["pass"],
                      partial_score=evaluation["reward"])
    except Exception as error:
        result.update(status="infrastructure_error", error=repr(error), passed=None)
    finally:
        if directory.exists():
            save(directory / "trace.json", {"task": task, "messages": [message_to_dict(m)
                 for m in state.get("messages", [])], "subagent_runs": state.get("subagent_runs", {})})
            await asyncio.to_thread(env.close)
        result["elapsed_seconds"] = time.time() - started
    return result


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        for key, value in manifest["config"].items():
            if key != "output": setattr(args, key, value)
    else:
        if manifest_path.exists(): raise ValueError("Run exists; use --resume")
        pool = Path(__file__).resolve().parents[2] / "external/toolathlon_gym/tasks/finalpool"
        tasks = sorted(p.name for p in pool.iterdir() if (p / "task_config.json").exists())
        if args.tasks:
            if not set(args.tasks) <= set(tasks): raise ValueError("Unknown task")
            tasks = sorted(set(args.tasks))
        args.image = subprocess.check_output(["podman", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
        manifest = {"config": {k: v for k, v in vars(args).items() if k != "resume"},
                    "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "started_at": time.time(), "tasks": tasks, "episodes": {}}
    manifest["status"] = "running"
    save(manifest_path, manifest)
    save(args.output / "metrics.json", metrics(manifest))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def work(task, repetition):
        key = f"{task}/rep-{repetition:03d}"
        if manifest["episodes"].get(key, {}).get("status") == "completed": return
        async with semaphore:
            parent = args.output / "episodes" / key
            parent.mkdir(parents=True, exist_ok=True)
            directory = parent / f"attempt-{len(list(parent.glob('attempt-*'))) + 1:03d}"
            manifest["episodes"][key] = {"status": "running", "artifact_path": str(directory)}
            save(manifest_path, manifest)
            try:
                result = await asyncio.wait_for(episode(args, task, repetition, directory), args.episode_timeout)
            except TimeoutError:
                result = {"status": "infrastructure_error", "error": "total episode deadline", "passed": None}
            result["artifact_path"] = str(directory)
            if directory.exists(): save(directory / "result.json", result)
            save(parent / "result.json", result)
            manifest["episodes"][key] = result
            save(manifest_path, manifest)
            save(args.output / "metrics.json", metrics(manifest))
            print(json.dumps({"episode": key, **result}), flush=True)
    await asyncio.gather(*(work(task, rep) for rep in range(1, args.repetitions + 1) for task in manifest["tasks"]))
    manifest["status"] = "completed" if all(e["status"] == "completed" for e in manifest["episodes"].values()) else "completed_with_infrastructure_errors"
    manifest["finished_at"] = time.time()
    save(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="localhost/decomposer-toolathlon-rl:status-filter-fix")
    parser.add_argument("--model", default="decomposer-4b-sft")
    parser.add_argument("--url", default="http://127.0.0.1:8026/v1")
    parser.add_argument("--subagent-port", type=int, default=8025)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--repetitions", "-n", type=int, default=3)
    parser.add_argument("--agent-timeout", type=float, default=2700)
    parser.add_argument("--episode-timeout", type=float, default=3300)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.repetitions < 1: parser.error("Counts must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
