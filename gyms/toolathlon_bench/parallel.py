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
        raise ValueError("Tasks must be unique names from the task pool")
    return selected


def lanes(tasks, repetitions, conflict_groups):
    """Group episodes that share external state; each group runs sequentially.

    Repetitions of one task reuse its services, and tasks in a conflict group
    reset the same services.
    """
    lane_of = {task: task for task in tasks}
    for group in conflict_groups:
        members = [task for task in tasks if task in group]
        for task in members:
            lane_of[task] = members[0]
    grouped = {}
    for task in tasks:
        grouped.setdefault(lane_of[task], []).extend(
            (task, repetition) for repetition in range(1, repetitions + 1)
        )
    return list(grouped.values())


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
        "repetition": repetition, "attempt": 1,
        "artifacts_dir": root / "traces", "evals_dir": root / "evals",
        "container_lock_file": root / "container.lock",
    }
    command = [sys.executable, str(Path(gym.__file__)), task]
    for name, value in values.items():
        if name in {"task", "tasks", "all", "concurrency", "repetitions", "output_dir", "harness"} or value is None or value is False:
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
    manifest = {"run_id": run_id, "status": "running", "harness": "decomposer" if args.agent == "decomposer" else "react",
                "assistant_id": args.agent,
                "tasks": tasks, "repetitions": args.repetitions, "episodes": [],
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    def save():
        temporary = root / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(root / "manifest.json")
    save()
    # Kind clusters exhaust the host's inotify limits when created concurrently.
    conflict_groups = [
        *json.loads((gym.TOOLATHLON_ROOT / "tasks/finalpool/task_conflict.json").read_text())["conflict_groups"],
        list(gym.K8S_TASK_CLEANUP_COMMANDS),
    ]
    total = len(tasks) * args.repetitions
    saved = threading.Lock()
    stop = threading.Event()

    def run_lane(lane):
        for task, repetition in lane:
            if stop.is_set():
                return
            command = episode_command(args, task, run_id, repetition, root)
            try:
                code, timed_out = execute(command, root / "logs" / task / str(repetition), args.episode_timeout, stop)
                result = {"returncode": code, "timed_out": timed_out}
            except Exception as error:
                result = {"returncode": -1, "error": repr(error)}
            result.update(task=task, repetition=repetition,
                          episode_id=f"{run_id}-{task}-r{repetition:03d}")
            with saved:
                manifest["episodes"].append(result)
                save()
                print(f"{len(manifest['episodes'])}/{total} {task}: exit {result['returncode']}", flush=True)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        futures = [executor.submit(run_lane, lane) for lane in lanes(tasks, args.repetitions, conflict_groups)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
        save()
    print(f"Raw results: {root}", flush=True)
    return root
