"""Collect off-policy WideSeek trajectories and index successful traces."""
import json
import time

from gyms.wideseek.run import cli, create_parser, describe_run, run_jobs
from gyms.wideseek.runtime import save
from sft.wideseek.scheduler import new_state, plan_next_wave, qualifies, statistics
from sft.sequence_limit import COLLECTION_SEQUENCE_LENGTH, StudentSequenceLimit


def completed_results(root, mode):
    results = []
    for path in sorted((root / mode).glob("*/attempt-*/result.json")):
        result = json.loads(path.read_text())
        execution = path.parent / result["execution_directory"]
        result["trace_available"] = (execution / "trace.json").is_file()
        result["trace"] = str((execution / "trace.json").relative_to(root))
        result["subagent_traces"] = [str(p.relative_to(root))
                                    for p in sorted((execution / "subagents").glob("*.json"))]
        results.append(result)
    return results


def successful_traces(root, mode, threshold):
    return [{"task_id": r["task_id"], "attempt": r["attempt"], "score": r["evaluation"]["score"],
             "trace": r["trace"], "subagent_traces": r["subagent_traces"]}
            for r in completed_results(root, mode) if qualifies(r, threshold)]


def update_index(root, mode, threshold, tasks, adaptive):
    rows = successful_traces(root, mode, threshold)
    index = root / "successful-traces.jsonl"
    temporary = index.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(index)
    save(root / "collection.json", {
        "success_threshold": threshold, "success_rule": "normal finish and score > threshold (or 1.0)",
        "tasks": len(tasks),
        "successful_traces": len(rows), "covered_tasks": len({r["task_id"] for r in rows}),
        "trace_index": index.name, "schedule": "coverage_first" if adaptive else "fixed attempts per task"})


async def prepare_collection(args):
    tasks, root, manifest = await describe_run(args)
    manifest['settings']['collection_sequence_length'] = COLLECTION_SEQUENCE_LENGTH
    path = root / "manifest.json"
    if args.resume:
        previous = json.loads(path.read_text())
        previous_settings = dict(previous["settings"])
        previous_settings["concurrency"] = args.concurrency
        if previous_settings != manifest["settings"]:
            raise ValueError("Resume settings or task data differ; use a new run directory")
        previous["settings"] = previous_settings
        save(path, previous)
    else:
        root.mkdir(parents=True, exist_ok=False)
        save(path, manifest)
    return tasks, root


async def main(args):
    from transformers import AutoTokenizer
    middleware = [StudentSequenceLimit(AutoTokenizer.from_pretrained(args.student_tokenizer, local_files_only=True))]
    if not 0 <= args.success_threshold <= 1 or args.target_successes < 1:
        raise ValueError("Threshold must be in [0, 1] and target successes positive")
    if args.adaptive and args.n != 1:
        raise ValueError("Adaptive collection launches one attempt per task per wave; use -n 1")
    if args.max_waves is not None and args.max_waves < 1:
        raise ValueError("Wave limit must be positive")
    tasks, root = await prepare_collection(args)
    if not args.adaptive:
        await run_jobs(tasks, ((t["task_id"], n) for n in range(1, args.n + 1) for t in tasks), root, args, middleware=middleware)
        update_index(root, args.mode, args.success_threshold, tasks, False)
        return
    path = root / "scheduler.json"
    expected = new_state([t["task_id"] for t in tasks], args.success_threshold, args.target_successes)
    state = json.loads(path.read_text()) if path.exists() else expected
    for key in ("schema_version", "policy", "task_ids", "success_threshold", "target_successes",
                "max_zero_success_launches", "max_balance_launches"):
        if state[key] != expected[key]:
            raise ValueError(f"Cannot change adaptive {key} on resume")
    state["status"] = "running"
    waves = 0
    while True:
        results = completed_results(root, args.mode)
        statistics(state, results)
        update_index(root, args.mode, args.success_threshold, tasks, True)
        if args.max_waves is not None and waves >= args.max_waves:
            plan_next_wave(state, results, allow_new_wave=False)
            state["status"] = "completed" if state["phase"] == "complete" else "wave_limit"
            if state["status"] == "completed":
                state["finished_at"] = time.time()
            save(path, state)
            return
        jobs = plan_next_wave(state, results)
        if not jobs:
            state.update(status="completed", finished_at=time.time())
            save(path, state)
            return
        save(path, state)  # Durable before any launch; completed results are never repeated.
        await run_jobs(tasks, jobs, root, args, middleware=middleware)
        waves += 1


def create_collection_parser():
    parser = create_parser()
    parser.description = __doc__
    parser.set_defaults(n=1)
    parser.add_argument("--student-tokenizer", required=True, help="Local student tokenizer used by SFT")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--success-threshold", type=float, default=.9)
    parser.add_argument("--adaptive", action="store_true", help="Prioritize coverage, then balance successful traces")
    parser.add_argument("--target-successes", type=int, default=4, help="Use 1 for coverage only; default 4")
    parser.add_argument("--max-waves", type=int, help="Stop after this many waves in this invocation (resumable)")
    return parser


if __name__ == "__main__":
    cli(main, parser=create_collection_parser())
