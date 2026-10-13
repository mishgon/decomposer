from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.common import METRICS_FILENAME, write_json  # noqa: E402
from evals.gaia2.metrics import compute, compute_suite  # noqa: E402
from gyms.gaia2 import run as runner  # noqa: E402


def report(
    run_dir: Path,
    *,
    write: bool,
) -> dict[str, Any]:
    result = (
        compute(run_dir)
        if (run_dir / ".eval_done.json").is_file()
        else compute_suite(run_dir)
    )
    if write:
        write_json(run_dir / METRICS_FILENAME, result)
    summary = {
        "run_dir": str(run_dir),
        "metrics": result["metrics"],
        "domains": result.get("domains"),
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

    parser = runner.build_parser(evaluation=True)
    args = parser.parse_args(rest)
    if len(set(args.domains)) != len(args.domains):
        parser.error("--domains must not contain duplicates")
    if args.task and len(args.domains) != 1:
        parser.error("--task requires exactly one domain")
    directory = (
        (
            args.output_dir
            or runner.RESULTS_ROOT
            / (
                datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + uuid.uuid4().hex[:8]
            )
        )
        .expanduser()
        .resolve()
    )
    if not args.dry and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    # Validate both domains before starting any episodes.
    for domain in args.domains:
        run_args = argparse.Namespace(**vars(args), domain=domain)
        runner.selected_scenario_ids(run_args)
        if not args.dry:
            runner.validate_preparation(domain=domain)
    if not args.dry:
        write_json(
            directory / "evaluation.json",
            {"agent": args.agent, "domains": args.domains},
        )
    for domain in args.domains:
        run_args = argparse.Namespace(**vars(args), domain=domain)
        run_args.output_dir = directory / domain
        runner.execute(run_args)
        if not args.dry:
            report(run_args.output_dir, write=not options.no_write)
    if not args.dry:
        report(directory, write=not options.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
