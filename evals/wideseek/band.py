"""RL band of a finished WideSeek or DeepDive run: the tasks an agent solves on some attempts but not all.

    python -m evals.wideseek.band <run_dir>

Writes ``<run_dir>/band.json``, as ``evals/tau2_gym/band.py`` does for tau2 (branch rl-tau2):
``in_band`` lists task IDs and ``passes`` holds ``[task_id, passes, attempts]`` for every task.
A pass is an attempt the collector counts as a success (``gyms.wideseek.metrics.qualifies``); a
wrong answer, timeout, context overflow or runaway turn is a failed attempt. The collector sets
aside attempts that a service outage failed and runs them again, so those never count.
"""
import argparse
import json
from pathlib import Path

from gyms.wideseek.metrics import qualifies
from gyms.wideseek.runtime import save

BAND_FILENAME = "band.json"


def compute(run_dir, threshold):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError(f"{run_dir} is {manifest['status']}, not completed")
    passes = {}
    for path in sorted((run_dir / manifest["settings"]["modes"][0]).glob("*/attempt-*/result.json")):
        result = json.loads(path.read_text())
        count, attempts = passes.get(result["task_id"], (0, 0))
        passes[result["task_id"]] = (count + qualifies(result, threshold), attempts + 1)
    in_band = sorted(task for task, (count, attempts) in passes.items() if 0 < count < attempts)
    return {
        "run_dir": str(run_dir),
        "success_threshold": threshold,
        "num_repeats": manifest["repetitions"],
        "tasks": len(passes),
        "never_solved": sum(count == 0 for count, _ in passes.values()),
        "always_solved": sum(count == attempts for count, attempts in passes.values()),
        "in_band_count": len(in_band),
        "in_band": in_band,
        "passes": [[task, *passes[task]] for task in sorted(passes)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--success-threshold", type=float, default=.9,
                        help="The collector's; QA scores are 0 or 1, so only table tasks depend on it")
    args = parser.parse_args()
    band = compute(args.run_dir, args.success_threshold)
    save(args.run_dir / BAND_FILENAME, band)
    print(f"{band['tasks']} tasks x {band['num_repeats']}: in band {band['in_band_count']}, "
          f"never solved {band['never_solved']}, always solved {band['always_solved']}")


if __name__ == "__main__":
    main()
