"""Evaluate on Workplace Assistant in one command: run the experiment, then compute metrics.

Takes every flag of `gyms/workplace_assistant/run.py`; `--purpose` defaults to
`evaluation`. The runner skips a run that is already complete, so repeating the
command only recomputes metrics. Metrics go to `<run_dir>/eval_metrics.json`; the
Qwen3.6 teacher run also gets its checksum-pinned `comparison.json`.

    python -m evals.workplace_assistant.run --experiment qwen35-4b-base-non-thinking \\
        --split validation --num-repeats 3
    python -m evals.workplace_assistant.run --metrics-only <run_dir> [--no-write]
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
from evals.workplace_assistant import comparison  # noqa: E402
from evals.workplace_assistant.metrics import COMPLETION_MARKER, compute  # noqa: E402
from gyms.workplace_assistant import run as runner  # noqa: E402
from gyms.workplace_assistant.experiments import get_experiment  # noqa: E402

COMPARISON_FILENAME = "comparison.json"


def report(run_dir: Path, *, write: bool) -> dict[str, Any]:
    result = compute(run_dir)
    compared = None
    if comparison.applies(read_json(run_dir / COMPLETION_MARKER)):
        compared = comparison.build_teacher_comparison(run_dir)
    if write:
        write_json(run_dir / METRICS_FILENAME, result)
        if compared is not None:
            write_json(run_dir / COMPARISON_FILENAME, compared)
    summary = {
        "run_dir": str(run_dir),
        "metrics": result["metrics"],
        "comparison": (
            compared["delta_candidate_minus_baseline"] if compared is not None else None
        ),
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

    if "-h" in rest or "--help" in rest:
        print(__doc__)
    if not any(argument == "--purpose" or argument.startswith("--purpose=") for argument in rest):
        rest = ["--purpose", "evaluation", *rest]
    args = runner.build_parser().parse_args(rest)
    code = runner.main(rest)
    if code or args.dry:
        return code
    report(runner.selected_output_dir(get_experiment(args.experiment), args), write=not options.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
