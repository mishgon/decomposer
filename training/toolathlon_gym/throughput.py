"""One frozen RL-rollout batch with native scoring and host/inference telemetry."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from types import SimpleNamespace
import urllib.request

from transformers import AutoTokenizer
from training.toolathlon_gym.agent_loop import ToolathlonAgentLoop
from training.toolathlon_gym.smoke import FrozenServer


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def telemetry(url, gpu):
    result = {"at": time.time(), "load": Path("/proc/loadavg").read_text(),
              "cpu": Path("/proc/stat").read_text().splitlines()[0],
              "diskstats": Path("/proc/diskstats").read_text(),
              "meminfo": Path("/proc/meminfo").read_text()}
    for name, command in {
        "gpu": ["nvidia-smi", "-i", str(gpu), "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
        "containers": ["podman", "ps", "-q"],
    }.items():
        try:
            result[name] = subprocess.check_output(command, text=True, timeout=5).strip()
        except Exception as error:
            result[name + "_error"] = repr(error)
    try:
        with urllib.request.urlopen(url.removesuffix("/v1") + "/metrics", timeout=5) as response:
            result["vllm"] = response.read().decode()
    except Exception as error:
        result["vllm_error"] = repr(error)
    return result


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["RL_ARTIFACTS"] = str(args.output.resolve())
    os.environ["RL_EPISODE_TIMEOUT"] = str(args.timeout)
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2 * args.concurrency + 8))
    task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    if args.all_tasks:
        candidates = [p.parent.name for p in Path("external/toolathlon_gym/tasks/finalpool").glob("*/task_config.json")]
        repetitions = 1
        pool_bytes = json.dumps(sorted(candidates)).encode()
    else:
        pool_bytes = args.pool.read_bytes()
        candidates = [t["task_id"] for t in json.loads(pool_bytes)["tasks"]]
        repetitions = 5
    if args.concurrency > len(candidates) * repetitions:
        raise ValueError("Not enough tasks for requested concurrency")
    names = sorted(candidates, key=lambda t: hashlib.sha256(f"42:{t}".encode()).hexdigest())[:args.concurrency // repetitions]
    started = time.time()
    status = {"status": "starting", "concurrency": args.concurrency, "tasks": names, "repetitions": repetitions,
              "started_at": started, "rows": [], "model": args.model,
              "subagent_url": os.environ["SUBAGENT_URL"], "episode_timeout": args.timeout,
              "pool_sha256": hashlib.sha256(pool_bytes).hexdigest(), "all_tasks": args.all_tasks,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "scope": "Frozen SFT policy, exact RL AgentLoop/token budgets; no optimizer or Ray scheduling"}
    save(args.output / "status.json", status)
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"])
    server = FrozenServer(args.url, args.model, args.output)
    stopping = asyncio.Event()

    async def sample():
        while not stopping.is_set():
            row = await asyncio.to_thread(telemetry, args.url, args.gpu)
            with (args.output / "telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            try: await asyncio.wait_for(stopping.wait(), 10)
            except TimeoutError: pass

    async def episode(name, repetition):
        begin = time.time()
        loop = object.__new__(ToolathlonAgentLoop)
        loop.tokenizer, loop.server_manager = tokenizer, server
        loop.rollout_config = SimpleNamespace(prompt_length=4096, response_length=12288)
        row = {"task": name, "repetition": repetition, "started_at": begin}
        try:
            out = await asyncio.wait_for(loop.run({"temperature": .7, "top_p": .8, "top_k": 20,
                "presence_penalty": 1.5}, extra_info={"task_id": name, "split": "throughput"},
                data_source="toolathlon_gym/throughput"), args.timeout + 600)
            row.update(status="scored", reward=out.reward_score, policy_tokens=sum(out.response_mask),
                       turns=out.num_turns, **out.extra_fields)
        except asyncio.CancelledError:
            row.update(status="interrupted")
            raise
        except Exception as error:
            row.update(status="infrastructure_error", error=repr(error))
        finally:
            row["seconds"] = time.time() - begin
            status["rows"].append(row)
            save(args.output / "status.json", status)
            print(json.dumps(row, default=str), flush=True)

    sensor = asyncio.create_task(sample())
    status["status"] = "running"
    save(args.output / "status.json", status)
    try:
        cases = [(name, rep) for rep in range(1, repetitions + 1) for name in names]
        if args.limit is not None:
            cases = cases[:args.limit]
        await asyncio.gather(*(episode(name, rep) for name, rep in cases))
        status["status"] = "complete"
    except asyncio.CancelledError:
        status["status"] = "interrupted"
    finally:
        stopping.set()
        await sensor
        await server.client.close()
        status["finished_at"] = time.time()
        save(args.output / "status.json", status)
        scored = [r for r in status["rows"] if r["status"] == "scored"]
        finished = [r for r in scored if r.get("stop_reason") == "finished"]
        elapsed = time.time() - started
        anchors = [r["seconds"] for r in scored if r["task"] in names[:8]]
        summary = {"status": status["status"], "concurrency": args.concurrency, "seconds": elapsed,
            "scored": len(scored), "natural_finishes": len(finished),
            "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in status["rows"]),
            "scored_per_hour": len(scored) * 3600 / elapsed,
            "finished_per_hour": len(finished) * 3600 / elapsed,
            "mean_reward": statistics.mean(r["reward"] for r in scored) if scored else None,
            "anchor_median_seconds": statistics.median(anchors) if anchors else None}
        save(args.output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=[40, 60, 80, 120, 200, 300, 400, 500], required=True)
    parser.add_argument("--all-tasks", action="store_true", help="Use distinct tasks from full Gym, one attempt each")
    parser.add_argument("--pool", type=Path, default=Path("training/toolathlon_gym/rl_task_pool.json"))
    parser.add_argument("--url", default="http://127.0.0.1:8026/v1")
    parser.add_argument("--model", default="decomposer-4b-sft")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--limit", type=int, help="Smoke-test only: limit total episodes")
    asyncio.run(run(parser.parse_args()))
