"""Dry-run or submit unified Workplace Assistant MLSpace evaluation jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gyms.workplace_assistant.experiments import (  # noqa: E402
    ARTIFACTS_ROOT,
    BASE_IMAGE,
    HF_HOME,
    INSTANCE_TYPES_BY_NUM_GPUS,
    PROJECT_VENV,
    SPLITS,
    STAGING_ROOT,
    Experiment,
    collect_experiments,
    completion_marker,
    gym_venv,
    job_description,
    run_name,
)
from gyms.workplace_assistant.run import positive_int, validate_preparation  # noqa: E402

_TAG_RE = re.compile(r"[#@]\S+")
_AUTHOR_RE = re.compile(r"[A-Za-z0-9_.-]+")
_PROXY_ENV_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
)
_SENSITIVE_PROXY_ENV_VARIABLES = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
}


def git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def normalize_job_desc(description: str) -> str:
    return " ".join(_TAG_RE.sub("", description).split())


def tracked_dirty(repo_root: Path) -> list[str]:
    return git(
        repo_root, "status", "--porcelain", "--untracked-files=no"
    ).splitlines()


def stage_repo(repo_root: Path, target: Path) -> None:
    marker = target / ".stage_complete"
    if marker.is_file():
        return
    if target.exists():
        raise RuntimeError(f"Incomplete staging directory exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    temporary.mkdir()
    subprocess.run(["cp", "-a", str(repo_root / ".git"), str(temporary)], check=True)
    subprocess.run(
        ["git", "-C", str(temporary), "reset", "--hard", "HEAD"], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(temporary),
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        check=True,
    )
    (temporary / ".stage_complete").write_text(
        git(repo_root, "rev-parse", "HEAD") + "\n"
    )
    os.rename(temporary, target)


def configured_proxy_environment(environ: Mapping[str, str]) -> dict[str, str]:
    proxy_env = {
        name: environ[name] for name in _PROXY_ENV_VARIABLES if environ.get(name)
    }
    if not (proxy_env.get("HTTPS_PROXY") or proxy_env.get("https_proxy")):
        raise RuntimeError(
            "HTTPS_PROXY or https_proxy is not set; the Decomposer job cannot reach OpenRouter"
        )
    return proxy_env


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from mls.manager.job.redact import redact_payload as mls_redact_payload

        redacted = mls_redact_payload(payload)
    except ImportError:
        redacted = {**payload, "env_variables": dict(payload["env_variables"])}
    for key in redacted.get("env_variables", {}):
        if key in _SENSITIVE_PROXY_ENV_VARIABLES or any(
            hint in key.upper() for hint in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            redacted["env_variables"][key] = "<redacted>"
    return redacted


def author_name(value: str) -> str:
    if not _AUTHOR_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "author name may contain only letters, numbers, '.', '_' and '-'"
        )
    return value


def build_job_desc(
    experiment: Experiment,
    split: str,
    num_repeats: int,
    limit: int | None,
    author: str,
) -> str:
    return f"{job_description(experiment, split, num_repeats, limit)} #{author}"


def build_job_script(
    staged_workdir: Path,
    experiment: Experiment,
    split: str,
    num_repeats: int,
    limit: int | None,
    *,
    force: bool,
) -> str:
    command = [
        str(PROJECT_VENV / "bin" / "python"),
        str(staged_workdir / "gyms" / "workplace_assistant" / "run.py"),
        "--workdir",
        str(staged_workdir),
        "--experiment",
        experiment.name,
        "--split",
        split,
        "--num-repeats",
        str(num_repeats),
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
    split: str,
    num_repeats: int,
    limit: int | None,
    author: str,
    base_image: str,
    priority: str | None,
    force: bool,
    proxy_env: Mapping[str, str],
    openrouter_key: str,
) -> dict[str, Any]:
    env_variables = {
        "WORKDIR": str(staged_workdir),
        "ARTIFACTS_ROOT": str(ARTIFACTS_ROOT),
        "HF_HOME": str(HF_HOME),
        "PROJECT_VENV": str(PROJECT_VENV),
        "GYM_VENV": str(gym_venv(staged_workdir)),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if experiment.kind == "decomposer":
        env_variables.update(proxy_env)
        env_variables["OPENROUTER_API_KEY_DECOMPOSER"] = openrouter_key
    payload: dict[str, Any] = {
        "script": build_job_script(
            staged_workdir,
            experiment,
            split,
            num_repeats,
            limit,
            force=force,
        ),
        "job_desc": build_job_desc(
            experiment, split, num_repeats, limit, author
        ),
        "env_variables": env_variables,
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
    experiments: Sequence[Experiment],
    split: str,
    num_repeats: int,
    limit: int | None,
) -> None:
    print("\nSelected jobs:")
    print("| # | Run | Agent | Split | GPUs |")
    print("| ---: | --- | --- | --- | ---: |")
    for index, experiment in enumerate(experiments, start=1):
        identity = run_name(experiment, num_repeats)
        if limit is not None:
            identity += f"/smoke_{limit}"
        print(
            f"| {index} | `{identity}` | {experiment.kind} | {split} | "
            f"{experiment.num_gpus} |"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--filter", action="append")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--num-repeats", type=positive_int, default=1)
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

    repo_root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
    selected_count = len(experiments)
    candidates: list[Experiment] = []
    skipped_completed = 0
    for experiment in experiments:
        marker = completion_marker(
            experiment, args.split, args.num_repeats, args.limit
        )
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
            "split": args.split,
            "num_repeats": args.num_repeats,
            "limit": args.limit,
            "skipped_completed": skipped_completed,
            "skipped_in_progress": 0,
            "launched": 0,
            "jobs": [],
        }
        print("__WORKPLACE_ASSISTANT_EVAL_JOBS_JSON__")
        print(json.dumps(summary, sort_keys=True))
        return 0

    if args.dry:
        staged_workdir = repo_root
    else:
        for experiment in experiments:
            validate_preparation(repo_root, experiment, args.split)
        dirty = tracked_dirty(repo_root)
        if dirty:
            raise RuntimeError(
                "Tracked worktree is dirty; commit before submission:\n"
                + "\n".join(dirty)
            )
        commit = git(repo_root, "rev-parse", "HEAD")
        staged_workdir = STAGING_ROOT / commit
        stage_repo(repo_root, staged_workdir)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY_DECOMPOSER", "")
    needs_openrouter = any(
        experiment.kind == "decomposer" for experiment in experiments
    )
    if needs_openrouter and not args.dry:
        if not openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY_DECOMPOSER is not set")
        proxy_env = configured_proxy_environment(os.environ)
    else:
        proxy_env = {
            name: os.environ[name]
            for name in _PROXY_ENV_VARIABLES
            if os.environ.get(name)
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
            split=args.split,
            num_repeats=args.num_repeats,
            limit=args.limit,
            author=args.author_name,
            base_image=args.base_image,
            priority=args.priority,
            force=args.force,
            proxy_env=proxy_env,
            openrouter_key=openrouter_key or "<not-set>",
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
            args.split,
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
                    "kind": experiment.kind,
                    "run_name": run_name(experiment, args.num_repeats),
                    "split": args.split,
                }
            )
        print("result", result)

    summary = {
        "selected": selected_count,
        "planned": len(planned),
        "split": args.split,
        "num_repeats": args.num_repeats,
        "limit": args.limit,
        "skipped_completed": skipped_completed,
        "skipped_in_progress": skipped_in_progress,
        "launched": len(launched),
        "jobs": launched,
    }
    print("__WORKPLACE_ASSISTANT_EVAL_JOBS_JSON__")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
