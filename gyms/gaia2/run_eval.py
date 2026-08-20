"""Dry-run or submit Gaia2 execution evaluation jobs to MLSpace."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gyms.gaia2.experiments import (  # noqa: E402
    BASE_IMAGE,
    DECOMPOSER_STAGING_ROOT,
    DOMAIN,
    HF_HOME,
    INSTANCE_TYPES_BY_NUM_GPUS,
    PROJECT_VENV,
    SPLIT,
    Experiment,
    collect_experiments,
    completion_marker,
    job_description,
    run_name,
)
from gyms.gaia2.run import positive_int, validate_preparation  # noqa: E402
from gyms.gaia2.staging import git, stage_revision  # noqa: E402

_TAG_RE = re.compile(r"[#@]\S+")
_AUTHOR_RE = re.compile(r"[A-Za-z0-9_.-]+")


def normalize_job_desc(description: str) -> str:
    return " ".join(_TAG_RE.sub("", description).split())


def author_name(value: str) -> str:
    if not _AUTHOR_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "author name may contain only letters, numbers, '.', '_' and '-'"
        )
    return value


def worktree_dirty(repo_root: Path) -> list[str]:
    return git(repo_root, "status", "--porcelain").splitlines()


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from mls.manager.job.redact import redact_payload as mls_redact_payload

        redacted = mls_redact_payload(payload)
    except ImportError:
        redacted = {**payload, "env_variables": dict(payload["env_variables"])}
    for key in redacted.get("env_variables", {}):
        if any(hint in key.upper() for hint in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted["env_variables"][key] = "<redacted>"
    return redacted


def build_job_desc(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None,
    author: str,
) -> str:
    return f"{job_description(experiment, num_repeats, limit)} #{author}"


def build_job_script(
    staged_workdir: Path,
    experiment: Experiment,
    num_repeats: int,
    limit: int | None,
    *,
    force: bool,
) -> str:
    command = [
        str(PROJECT_VENV / "bin" / "python"),
        str(staged_workdir / "gyms" / "gaia2" / "run.py"),
        "--workdir",
        str(staged_workdir),
        "--experiment",
        experiment.name,
        "--split",
        SPLIT,
        "--domain",
        DOMAIN,
        "--num-repeats",
        str(num_repeats),
        "--cuda-visible-devices",
        ",".join(str(index) for index in range(experiment.num_gpus)),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if force:
        command.append("--force")
    return shlex.join(command)


def build_payload(
    experiment: Experiment,
    staged_workdir: Path,
    *,
    num_repeats: int,
    limit: int | None,
    author: str,
    base_image: str,
    priority: str | None,
    force: bool,
    judge_environment: Mapping[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "script": build_job_script(
            staged_workdir,
            experiment,
            num_repeats,
            limit,
            force=force,
        ),
        "job_desc": build_job_desc(experiment, num_repeats, limit, author),
        "env_variables": {
            "WORKDIR": str(staged_workdir),
            "HF_HOME": str(HF_HOME),
            "PROJECT_VENV": str(PROJECT_VENV),
            "PYTHONPATH": os.pathsep.join(
                (str(staged_workdir), str(staged_workdir / "src"))
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            **judge_environment,
        },
        "instance_type": INSTANCE_TYPES_BY_NUM_GPUS[experiment.num_gpus],
        "type": "binary_exp",
        "shm_size_class": "large",
        "base_image": base_image,
        "n_workers": 1,
        "processes_per_worker": 1,
    }
    if priority:
        payload["priority_class"] = priority
    return payload


def print_parameter_table(
    experiments: Sequence[Experiment], num_repeats: int, limit: int | None
) -> None:
    print("\nSelected jobs:")
    print("| # | Run | Agent | Split/domain | GPUs |")
    print("| ---: | --- | --- | --- | ---: |")
    for index, experiment in enumerate(experiments, start=1):
        identity = run_name(experiment, num_repeats)
        if limit is not None:
            identity += f"/smoke_{limit}"
        print(
            f"| {index} | `{identity}` | {experiment.kind} | "
            f"{SPLIT}/{DOMAIN} | {experiment.num_gpus} |"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--filter", action="append")
    parser.add_argument("--split", choices=(SPLIT,), default=SPLIT)
    parser.add_argument("--domain", choices=(DOMAIN,), default=DOMAIN)
    parser.add_argument("--num-repeats", type=positive_int, default=3)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--dry", "--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--priority", choices=("low", "medium", "high"))
    parser.add_argument("--base-image", default=BASE_IMAGE)
    parser.add_argument(
        "--author-name", "--author_name", type=author_name, required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.experiment and not args.filter:
        parser.error("at least one --experiment or --filter is required")
    try:
        experiments = collect_experiments(
            tuple(args.experiment or ()),
            tuple(args.filter or ()),
            default_all=False,
        )
    except ValueError as error:
        parser.error(str(error))
    if not experiments:
        parser.error("experiment selectors matched no registered experiments")

    selected_count = len(experiments)
    candidates: list[Experiment] = []
    skipped_completed = 0
    for experiment in experiments:
        marker = completion_marker(experiment, args.num_repeats, args.limit)
        if marker.is_file() and not args.force:
            skipped_completed += 1
            print(f"Skip (completed): {marker}")
        else:
            candidates.append(experiment)
    experiments = candidates
    if not experiments:
        summary = {
            "selected": selected_count,
            "planned": 0,
            "split": SPLIT,
            "domain": DOMAIN,
            "num_repeats": args.num_repeats,
            "limit": args.limit,
            "skipped_completed": skipped_completed,
            "skipped_in_progress": 0,
            "launched": 0,
            "jobs": [],
        }
        print("__GAIA2_EVAL_JOBS_JSON__")
        print(json.dumps(summary, sort_keys=True))
        return 0

    repo_root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if args.dry:
        staged_workdir = repo_root
    else:
        for experiment in experiments:
            validate_preparation(experiment)
        dirty = worktree_dirty(repo_root)
        if dirty:
            raise RuntimeError(
                "Worktree is dirty; commit before submission:\n" + "\n".join(dirty)
            )
        commit = git(repo_root, "rev-parse", "HEAD")
        staged_workdir = DECOMPOSER_STAGING_ROOT / commit
        stage_revision(repo_root, commit, staged_workdir)

    judge_url = os.environ.get("LLM_PROXY_URL", "")
    judge_key = os.environ.get("LLM_PROXY_MASTER_KEY", "")
    if not args.dry and (not judge_url or not judge_key):
        raise RuntimeError("LLM_PROXY_URL and LLM_PROXY_MASTER_KEY are required")
    judge_environment = {
        "LLM_PROXY_URL": judge_url or "<not-set>",
        "LLM_PROXY_MASTER_KEY": judge_key or "<not-set>",
    }

    from mls.manager.job.utils import (
        get_in_progress_jobs,
        run_job_with_retry,
        training_job_api_from_profile,
    )

    client, options = training_job_api_from_profile(args.profile)
    in_progress = {
        normalize_job_desc(job.get("job_desc", ""))
        for job in get_in_progress_jobs(client_profile=args.profile)
    }
    planned: list[tuple[Experiment, dict[str, Any]]] = []
    skipped_in_progress = 0
    for experiment in experiments:
        payload = build_payload(
            experiment,
            staged_workdir,
            num_repeats=args.num_repeats,
            limit=args.limit,
            author=args.author_name,
            base_image=args.base_image,
            priority=args.priority,
            force=args.force,
            judge_environment=judge_environment,
        )
        payload["region"] = options["region"]
        if normalize_job_desc(payload["job_desc"]) in in_progress:
            skipped_in_progress += 1
            print(f"Skip (already queued/running): {payload['job_desc']}")
            continue
        planned.append((experiment, payload))

    if args.dry:
        print_parameter_table(
            [experiment for experiment, _ in planned],
            args.num_repeats,
            args.limit,
        )

    launched: list[dict[str, Any]] = []
    for experiment, payload in planned:
        print(f"Would launch [{experiment.num_gpus} GPU]: {payload['job_desc']}")
        print(payload["script"])
        if args.dry:
            print(json.dumps(redact_payload(payload), indent=2, sort_keys=True))
            continue
        result = run_job_with_retry(client, payload, profile=args.profile)
        job_name = result.get("job_name") if isinstance(result, dict) else None
        if job_name:
            launched.append(
                {
                    "job_name": job_name,
                    "experiment": experiment.name,
                    "num_gpus": experiment.num_gpus,
                }
            )

    summary = {
        "selected": selected_count,
        "planned": len(planned),
        "split": SPLIT,
        "domain": DOMAIN,
        "num_repeats": args.num_repeats,
        "limit": args.limit,
        "skipped_completed": skipped_completed,
        "skipped_in_progress": skipped_in_progress,
        "launched": len(launched),
        "jobs": launched,
    }
    print("__GAIA2_EVAL_JOBS_JSON__")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
