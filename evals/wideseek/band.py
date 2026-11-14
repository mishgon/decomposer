"""RL band of finished WideSeek or DeepDive runs: the tasks an agent solves on some attempts but not all.

    python -m evals.wideseek.band <run_dir> [<run_dir> ...] [--undecided <tasks.jsonl> --data <pool.jsonl>]

Writes ``band.json`` into the last run directory, as ``evals/tau2_gym/band.py`` does for tau2
(branch rl-tau2): ``in_band`` lists task IDs and ``passes`` holds ``[task_id, passes, attempts]``
for every task, counted over all the given runs. A pass is an attempt the collector counts as a
success (``gyms.wideseek.metrics.qualifies``); a wrong answer, timeout, context overflow or runaway
turn is a failed attempt. The collector sets aside attempts that a service outage failed and runs
them again, so those never count.

Runs in stages: after a first run, ``--undecided`` writes the rows of ``--data`` whose attempts so
far all passed or all failed; a further run on that file gives those tasks more attempts, and the
band of both runs together counts all of them.
"""
import argparse
import json
from pathlib import Path

from gyms.wideseek.metrics import qualifies
from gyms.wideseek.runtime import save

BAND_FILENAME = "band.json"


def compute(run_dirs, threshold):
    passes, repetitions = {}, []
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"{run_dir} is {manifest['status']}, not completed")
        repetitions.append(manifest["repetitions"])
        for path in sorted((run_dir / manifest["settings"]["modes"][0]).glob("*/attempt-*/result.json")):
            result = json.loads(path.read_text())
            count, attempts = passes.get(result["task_id"], (0, 0))
            passes[result["task_id"]] = (count + qualifies(result, threshold), attempts + 1)
    in_band = sorted(task for task, (count, attempts) in passes.items() if 0 < count < attempts)
    return {
        "run_dirs": [str(run_dir) for run_dir in run_dirs],
        "success_threshold": threshold,
        "repetitions": repetitions,
        "tasks": len(passes),
        "never_solved": sum(count == 0 for count, _ in passes.values()),
        "always_solved": sum(count == attempts for count, attempts in passes.values()),
        "in_band_count": len(in_band),
        "in_band": in_band,
        "passes": [[task, *passes[task]] for task in sorted(passes)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Runs drawn from the same task pool")
    parser.add_argument("--success-threshold", type=float, default=.9,
                        help="The collector's; QA scores are 0 or 1, so only table tasks depend on it")
    parser.add_argument("--undecided", type=Path, help="Write here the --data rows whose attempts all agree")
    parser.add_argument("--data", type=Path, help="The task pool the runs drew from")
    args = parser.parse_args()
    if args.undecided and not args.data:
        parser.error("--undecided needs --data")
    band = compute(args.run_dirs, args.success_threshold)
    save(args.run_dirs[-1] / BAND_FILENAME, band)
    if args.undecided:
        undecided = {task for task, count, attempts in band["passes"] if count in (0, attempts)}
        rows = [row for row in args.data.read_text().splitlines() if json.loads(row)["task_id"] in undecided]
        args.undecided.write_text("".join(row + "\n" for row in rows))
    print(f"{band['tasks']} tasks over runs of {band['repetitions']} attempts: in band {band['in_band_count']}, "
          f"never solved {band['never_solved']}, always solved {band['always_solved']}")


if __name__ == "__main__":
    main()
