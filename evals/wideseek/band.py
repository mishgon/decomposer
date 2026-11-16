"""RL band of finished WideSeek or DeepDive runs: the tasks an agent solves on some attempts but not all.

    python -m evals.wideseek.band <run_dir> [<run_dir> ...] [--undecided <tasks.jsonl> --data <pool.jsonl>]

Writes ``band.json`` into the last run directory, as ``evals/tau2_gym/band.py`` does for tau2
(branch rl-tau2): ``in_band`` lists task IDs and ``passes`` holds ``[task_id, passes, attempts]``
for every task, counted over all the given runs. A pass is an attempt the collector counts as a
success (``gyms.wideseek.metrics.qualifies``); a wrong answer, timeout, context overflow or runaway
turn is a failed attempt. Judge, infrastructure and unknown errors never count. A task needs at
least the largest repetition count requested for it in the supplied runs before it is classified.
``--rerun`` writes underfilled tasks from ``--data`` for replacement attempts in a new run.

Runs in stages: after a first run, ``--undecided`` writes the rows of ``--data`` whose attempts so
far all passed or all failed; a further run on that file gives those tasks more attempts, and the
band of both runs together counts all of them.
"""
import argparse
import json
import math
from pathlib import Path

from gyms.wideseek.metrics import qualifies
from gyms.wideseek.runtime import save

BAND_FILENAME = "band.json"


def resolved(result, threshold):
    """Return a measured pass/fail, or None when the outcome is unknown."""
    if (any(result.get(key) for key in ('cleanup_errors', 'capture_error', 'trace_error', 'export_error'))
            or not result.get('trace_available', True)):
        return None
    evaluation = result.get('evaluation', {})
    if evaluation.get('status') in {'evaluation_error', 'infrastructure_error'}:
        return None
    status = result.get('status')
    if status == 'finished':
        score = evaluation.get('score')
        if (evaluation.get('status') == 'scored' and type(score) in (int, float)
                and math.isfinite(score) and 0 <= score <= 1):
            return qualifies(result, threshold)
        return None
    if status in {'timeout', 'budget_exceeded', 'context_exceeded', 'invalid_tool_call'}:
        return False
    error = str(result.get('error', ''))
    if status == 'error' and (
        error.startswith(('GraphRecursionError:', 'OpenAIContextOverflowError:'))
        or any(name in error for name in ('ContextWindowExceededError', 'context_length_exceeded',
                                         "model's maximum context length"))
        or error in {
            'ValueError: Decomposer requested tools instead of completing graceful shutdown',
            'ValueError: Agent requested tools instead of returning its report',
            'RuntimeError: vLLM exhausted max_completion_tokens before completing its response.',
        }
    ):
        return False
    return None


def qualifies_band(passes, attempts, max_pass_rate=None):
    return (passes / attempts <= max_pass_rate if max_pass_rate is not None
            else 0 < passes < attempts)


def compute(run_dirs, threshold, max_pass_rate=None):
    passes, repetitions, required, unresolved = {}, [], {}, []
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"{run_dir} is {manifest['status']}, not completed")
        repetitions.append(manifest["repetitions"])
        mode, = manifest["settings"]["modes"]
        for task in manifest["settings"]["tasks"]:
            required[task] = max(required.get(task, 0), manifest["repetitions"])
            passes.setdefault(task, (0, 0))
            for attempt in range(1, manifest["repetitions"] + 1):
                path = run_dir / mode / task / f"attempt-{attempt:03d}" / "result.json"
                result = json.loads(path.read_text()) if path.exists() else {}
                outcome = resolved(result, threshold)
                if outcome is None:
                    unresolved.append({"task_id": task, "result": str(path),
                        "status": result.get('status', 'missing'), "error": result.get('error'),
                        "evaluation": result.get('evaluation')})
                    continue
                count, attempts = passes[task]
                passes[task] = (count + outcome, attempts + 1)
    eligible = {task: counts for task, counts in passes.items() if counts[1] >= required[task]}
    rerun = sorted(set(passes) - eligible.keys())
    in_band = sorted(task for task, (count, attempts) in eligible.items()
                     if qualifies_band(count, attempts, max_pass_rate))
    return {
        "run_dirs": [str(run_dir) for run_dir in run_dirs],
        "success_threshold": threshold,
        "max_pass_rate": max_pass_rate,
        "repetitions": repetitions,
        "tasks": len(passes),
        "never_solved": sum(count == 0 for count, _ in eligible.values()),
        "always_solved": sum(count == attempts for count, attempts in eligible.values()),
        "rerun": rerun,
        "missing_attempts": {task: required[task] - passes[task][1] for task in rerun},
        "required_attempts": required,
        "unresolved": unresolved,
        "in_band_count": len(in_band),
        "in_band": in_band,
        "passes": [[task, *passes[task]] for task in sorted(passes)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Runs drawn from the same task pool")
    parser.add_argument("--success-threshold", type=float, default=.9,
                        help="The collector's; QA scores are 0 or 1, so only table tasks depend on it")
    parser.add_argument("--max-pass-rate", type=float,
                        help="Select pass rates up to this inclusive bound, including zero")
    parser.add_argument("--undecided", type=Path, help="Write here the --data rows whose attempts all agree")
    parser.add_argument("--rerun", type=Path, help="Write here the --data rows needing replacement attempts")
    parser.add_argument("--data", type=Path, help="The task pool the runs drew from")
    args = parser.parse_args()
    if args.max_pass_rate is not None and not 0 <= args.max_pass_rate <= 1:
        parser.error('--max-pass-rate must be between 0 and 1')
    if (args.undecided or args.rerun) and not args.data:
        parser.error("--undecided and --rerun need --data")
    band = compute(args.run_dirs, args.success_threshold, args.max_pass_rate)
    save(args.run_dirs[-1] / BAND_FILENAME, band)
    undecided = {task for task, count, attempts in band["passes"] if count in (0, attempts)} | set(band['rerun'])
    for output, selected in ((args.undecided, undecided), (args.rerun, set(band['rerun']))):
        if output:
            rows = {json.loads(row)["task_id"]: row for row in args.data.read_text().splitlines() if row.strip()}
            if selected - rows.keys():
                raise ValueError(f"Task pool is missing requested tasks: {sorted(selected - rows.keys())}")
            output.write_text("".join(row + "\n" for task, row in rows.items() if task in selected))
    print(f"{band['tasks']} tasks over runs of {band['repetitions']} attempts: in band {band['in_band_count']}, "
          f"never solved {band['never_solved']}, always solved {band['always_solved']}, "
          f"need reruns {len(band['rerun'])}")


if __name__ == "__main__":
    main()
