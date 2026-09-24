"""Evaluate on GAIA2 in one command: run the evaluation, then compute metrics.

Takes every flag of `gyms/gaia2/run.py` except `--purpose trace-generation` (trace
generation stays in the gym). The runner skips a run that is already complete, so
repeating the command only recomputes metrics. Metrics go to
`<run_dir>/eval_metrics.json`; a full held-out execution run (`--partition test`,
three repeats, no limit) also gets its baseline `comparison.json`.

    python -m evals.gaia2.run --experiment <name> --partition test --num-repeats 3
    python -m evals.gaia2.run --metrics-only <run_dir> [--no-write]
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

from evals.common import METRICS_FILENAME, read_json, write_json  # noqa: E402
from evals.gaia2 import comparison  # noqa: E402
from evals.gaia2.metrics import COMPLETION_MARKER, compute  # noqa: E402
from gyms.gaia2 import run as runner  # noqa: E402
from gyms.gaia2.experiments import get_domain_spec, get_experiment  # noqa: E402

COMPARISON_FILENAME = "comparison.json"


def report(
    run_dir: Path,
    *,
    write: bool,
    baselines: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = compute(run_dir)
    marker = read_json(run_dir / COMPLETION_MARKER)
    compared = None
    if comparison.applies(marker):
        compared = comparison.build_heldout_comparison(
            get_experiment(marker["experiment"]), run_dir, baselines=baselines
        )
    if write:
        write_json(run_dir / METRICS_FILENAME, result)
        if compared is not None:
            write_json(run_dir / COMPARISON_FILENAME, compared)
    summary = {
        "run_dir": str(run_dir),
        "metrics": result["metrics"],
        "fixed_denominator_score": result["gym_metrics"]["fixed_denominator_score"],
        "comparison": str(run_dir / COMPARISON_FILENAME) if compared is not None and write else None,
        "written": str(run_dir / METRICS_FILENAME) if write else None,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return result


def output_dir(args: argparse.Namespace) -> Path:
    """The directory `gyms/gaia2/run.py` writes for these evaluation arguments."""
    spec = get_domain_spec(args.domain)
    experiment = runner.select_prompt_profile(get_experiment(args.experiment), args.prompt_profile)
    return runner.selected_output_dir(
        experiment,
        args,
        domain=spec.name,
        prompt_profile=args.prompt_profile,
        ports=runner.Gaia2PortLayout(args.port_offset),
    )


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

    if "-h" in rest or "--help" in rest:
        print(__doc__)
    args = runner.build_parser().parse_args(rest)
    if args.purpose != "evaluation":
        raise SystemExit("evals.gaia2.run evaluates; run trace generation with gyms/gaia2/run.py")
    baselines = None
    prospective = {
        "domain": get_domain_spec(args.domain).name,
        "partition": args.partition,
        "limit": args.limit,
        "num_repeats": args.num_repeats,
    }
    if not args.dry and comparison.applies(prospective):
        # Fail before hours of rollouts if a baseline the comparison needs is missing.
        baselines = comparison.collect_baseline_summaries()
    code = runner.main(rest)
    if code or args.dry:
        return code
    report(output_dir(args), write=not options.no_write, baselines=baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
