"""Fixed task execution with raw artifacts; no sampling or retry policy."""

import concurrent.futures
import json
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def select_tasks(root, task=None, tasks=None, all_tasks=False):
    if sum((bool(task), bool(tasks), all_tasks)) != 1:
        raise ValueError("Choose one task, --tasks, or --all")
    available = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    selected = available if all_tasks else ([task] if task else tasks)
    if len(set(selected)) != len(selected) or any(t not in available for t in selected):
        raise ValueError("Tasks must be unique names from the Gym task pool")
    return selected


def execute(command, directory, timeout, stop_event):
    """Supervise one process, preserving stdout/stderr even on cancellation."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runner.stdout.log").open("wb") as stdout, (directory / "runner.stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        started = time.monotonic()
        try:
            while True:
                try:
                    return process.wait(timeout=0.5), False
                except subprocess.TimeoutExpired:
                    if stop_event.is_set():
                        raise KeyboardInterrupt("Execution interrupted")
                    if time.monotonic() - started >= timeout:
                        return -1, True
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def episode_command(args, task, run_id, repetition, root):
    from . import run as gym
    values = vars(args) | {
        "task": task, "run_id": run_id,
        "episode_id": f"{run_id}-{task}-r{repetition:03d}",
        "repetition": repetition, "attempt": 1, "reuse_vllm": True,
        "artifacts_dir": root / "traces", "evals_dir": root / "evals",
        "container_lock_file": root / "container.lock",
    }
    command = [sys.executable, str(Path(gym.__file__)), task]
    for name, value in values.items():
        if name in {"task", "tasks", "all", "concurrency", "repetitions", "output_dir"} or value is None or value is False:
            continue
        command.append("--" + name.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    return command


def run(args):
    from . import run as gym
    if args.concurrency < 1 or args.repetitions < 1 or args.episode_timeout <= 0:
        raise ValueError("Concurrency, repetitions and timeout must be positive")
    tasks = select_tasks(gym.TOOLATHLON_ROOT / "tasks/finalpool", args.task, args.tasks, args.all)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    root = args.output_dir.resolve() / run_id
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": run_id, "status": "running", "harness": args.harness,
                "tasks": tasks, "repetitions": args.repetitions, "episodes": [],
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    def save():
        temporary = root / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(root / "manifest.json")
    save()
    stop = threading.Event()
    server = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    futures = {}
    try:
        if not args.subagent_base_url:
            server = gym.start_vllm(model=args.subagent_model, served_model_name=args.subagent_api_model,
                port=args.subagent_port, gpu=args.subagent_gpu, max_model_len=args.vllm_max_model_len,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization, timeout=args.vllm_startup_timeout,
                log_path=root / "vllm.log", reuse=args.reuse_vllm, data_parallel_size=args.vllm_data_parallel_size)
        for task in tasks:
            for repetition in range(1, args.repetitions + 1):
                command = episode_command(args, task, run_id, repetition, root)
                future = executor.submit(execute, command, root / "logs" / task / str(repetition), args.episode_timeout, stop)
                futures[future] = (task, repetition)
        for future in concurrent.futures.as_completed(futures):
            task, repetition = futures[future]
            try:
                code, timed_out = future.result()
                result = {"returncode": code, "timed_out": timed_out}
            except Exception as error:
                result = {"returncode": -1, "error": repr(error)}
            result.update(task=task, repetition=repetition,
                          episode_id=f"{run_id}-{task}-r{repetition:03d}")
            manifest["episodes"].append(result)
            save()
            print(f"{len(manifest['episodes'])}/{len(futures)} {task}: exit {result['returncode']}", flush=True)
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
        gym.stop_vllm(server)
        save()
    print(f"Raw results: {root}", flush=True)
    return root
