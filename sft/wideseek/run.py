"""Collect off-policy WideSeek trajectories and index successful traces."""
import json

from gyms.wideseek.run import cli, create_parser, main as run_episodes
from gyms.wideseek.runtime import DEFAULT_SUBAGENT, DEFAULT_TEACHER, save


def successful_traces(root, mode, threshold):
    rows = []
    for path in sorted((root / mode).glob("*/attempt-???/result.json")):
        result = json.loads(path.read_text())
        score = result["evaluation"].get("score")
        if (result["status"] != "finished" or result.get("cleanup_errors")
                or score is None or score < threshold):
            continue
        execution = path.parent / result["execution_directory"]
        if not (execution / "trace.json").is_file():
            continue
        rows.append({"task_id": result["task_id"], "attempt": result["attempt"],
                     "score": score, "trace": str((execution / "trace.json").relative_to(root)),
                     "subagent_traces": [str(p.relative_to(root))
                                         for p in sorted((execution / "subagents").glob("*.json"))]})
    return rows


async def main(args):
    if not 0 < args.success_threshold <= 1:
        raise ValueError("Success threshold must be in (0, 1]")
    await run_episodes(args)
    root = args.output.resolve()
    rows = successful_traces(root, args.mode, args.success_threshold)
    # Failed attempts and complete model/tool/judge logs stay in the raw run.
    index = root / "successful-traces.jsonl"
    temporary = index.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(index)
    manifest = json.loads((root / "manifest.json").read_text())
    save(root / "collection.json", {
        "success_threshold": args.success_threshold, "tasks": len(manifest["settings"]["tasks"]),
        "successful_traces": len(rows), "covered_tasks": len({r["task_id"] for r in rows}),
        "trace_index": index.name, "schedule": "fixed attempts per task"})


if __name__ == "__main__":
    parser = create_parser()
    parser.description = __doc__
    parser.set_defaults(model=DEFAULT_TEACHER, subagent_model=DEFAULT_SUBAGENT, n=1)
    parser.add_argument("--success-threshold", type=float, default=.9)
    cli(main, parser=parser)
