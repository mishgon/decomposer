"""Evaluate on the tau2 gym in one command: run the rollouts, then compute metrics.

Takes every flag of `gyms/tau2_gym/run.py` (experiment, pool, all or some tasks,
repeats, checkpoint, ...). A run that is already complete is not repeated unless
`--force` or `--resume` is given. Metrics go to `<run_dir>/eval_metrics.json`.

    python -m evals.tau2_gym.run --experiment qwen35_4b_student_checkpoint \\
        --manager-checkpoint <dir> --pool decomposer_eval_v1 --tasks-per-domain 5 --num-repeats 3
    python -m evals.tau2_gym.run --metrics-only <run_dir> [--no-write]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.common import METRICS_FILENAME, write_json  # noqa: E402
from evals.tau2_gym.metrics import COMPLETION_MARKER, compute  # noqa: E402
from gyms.tau2_gym import run as runner  # noqa: E402


def report(run_dir: Path, *, write: bool) -> dict[str, Any]:
    result = compute(run_dir)
    if write:
        write_json(run_dir / METRICS_FILENAME, result)
    summary = {
        "run_dir": str(run_dir),
        "metrics": result["metrics"],
        "pass_at_1_by_domain": {
            domain: round(metrics["pass_at_1"], 4) for domain, metrics in result["by_domain"].items()
        },
        "written": str(run_dir / METRICS_FILENAME) if write else None,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    options_parser = argparse.ArgumentParser(add_help=False)
    options_parser.add_argument("--metrics-only", type=Path, metavar="RUN_DIR")
    options_parser.add_argument("--no-write", action="store_true")
    options, rest = options_parser.parse_known_args(arguments)
    if options.metrics_only is not None:
        if rest:
            raise SystemExit(f"--metrics-only takes no run arguments, got {rest}")
        report(options.metrics_only, write=not options.no_write)
        return 0

    parser = runner.build_parser()
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    # Parsed above; listed here so --help shows them.
    parser.add_argument("--metrics-only", type=Path, metavar="RUN_DIR",
                        help="only compute metrics for a finished run directory")
    parser.add_argument("--no-write", action="store_true", help=f"print metrics without writing {METRICS_FILENAME}")
    args = parser.parse_args(rest)
    _, output_dir = runner.resolve_output_dir(args)
    if args.dry:
        return runner.execute(args)
    if (output_dir / COMPLETION_MARKER).is_file() and not (args.force or args.resume):
        print(f"[evals.tau2_gym] {output_dir} is complete; computing metrics only", flush=True)
    else:
        code = runner.execute(args)
        if code:
            return code
    report(output_dir, write=not options.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
