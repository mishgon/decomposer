"""Dry-run or submit Gaia2 evaluation jobs to MLSpace."""

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from decomposer.prompts import DECOMPOSER_PROMPT_PROFILES  # noqa: E402

from gyms.gaia2.experiments import (  # noqa: E402
    BASE_IMAGE,
    DECOMPOSER_STAGING_ROOT,
    DOMAIN,
    DOMAINS,
    HF_HOME,
    INSTANCE_TYPES_BY_NUM_GPUS,
    PARTITIONS,
    PROJECT_VENV,
    SPLIT,
    Gaia2Domain,
    Experiment,
    collect_experiments,
    completion_marker,
    get_domain_spec,
    job_description,
    run_name,
    trace_completion_marker,
    trace_run_name,
)
from gyms.gaia2.run import (  # noqa: E402
    nonnegative_int,
    positive_int,
    run_identity,
    select_prompt_profile,
    validate_run_identity,
    validate_preparation,
)
from gyms.gaia2.staging import git, stage_revision  # noqa: E402

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


def configured_proxy_environment(environ: Mapping[str, str]) -> dict[str, str]:
    proxy_env = {
        name: environ[name] for name in _PROXY_ENV_VARIABLES if environ.get(name)
    }
    if not (proxy_env.get("HTTPS_PROXY") or proxy_env.get("https_proxy")):
        raise RuntimeError(
            "HTTPS_PROXY or https_proxy is not set; the Gaia2 job cannot reach OpenRouter"
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


def build_job_desc(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None,
    author: str,
    *,
    purpose: str = "evaluation",
    partition: str = "full",
    rollout_offset: int = 0,
    prompt_profile: str | None = None,
    domain: Gaia2Domain = DOMAIN,
) -> str:
    spec = get_domain_spec(domain)
    if purpose == "trace-generation":
        identity = trace_run_name(
            experiment,
            num_repeats,
            rollout_offset,
            prompt_profile=prompt_profile,
        )
        if limit is not None:
            identity += f"-smoke-{limit}"
        return (
            f"gaia2-trace {spec.split_manifest_name} {partition} "
            f"{experiment.kind}-agent {identity} #{author}"
        )
    description = job_description(
        experiment,
        num_repeats,
        limit,
        partition=partition,
        domain=spec.name,
    )
    if prompt_profile is not None:
        description += f" prompt-{prompt_profile}"
    return f"{description} #{author}"


def build_job_script(
    staged_workdir: Path,
    experiment: Experiment,
    num_repeats: int,
    limit: int | None,
    *,
    force: bool,
    purpose: str = "evaluation",
    partition: str = "full",
    concurrency: int | None = None,
    rollout_offset: int = 0,
    prompt_profile: str | None = None,
    domain: Gaia2Domain = DOMAIN,
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
        domain,
        "--purpose",
        purpose,
        "--partition",
        partition,
        "--num-repeats",
        str(num_repeats),
    ]
    if concurrency is not None:
        command.extend(["--concurrency", str(concurrency)])
    if prompt_profile is not None:
        command.extend(["--prompt-profile", prompt_profile])
    if purpose == "trace-generation" or rollout_offset:
        command.extend(["--rollout-offset", str(rollout_offset)])
    if experiment.num_gpus:
        command.extend(
            [
                "--cuda-visible-devices",
                ",".join(str(index) for index in range(experiment.num_gpus)),
            ]
        )
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
    proxy_environment: Mapping[str, str],
    openrouter_key: str,
    purpose: str = "evaluation",
    partition: str = "full",
    concurrency: int | None = None,
    rollout_offset: int = 0,
    prompt_profile: str | None = None,
    domain: Gaia2Domain = DOMAIN,
) -> dict[str, Any]:
    if experiment.num_gpus == 0:
        raise ValueError(
            f"{experiment.name} is a local-only remote experiment and cannot be "
            "submitted through the GPU MLSpace launcher"
        )
    env_variables = {
        "WORKDIR": str(staged_workdir),
        "HF_HOME": str(HF_HOME),
        "PROJECT_VENV": str(PROJECT_VENV),
        "PYTHONPATH": os.pathsep.join(
            (str(staged_workdir), str(staged_workdir / "src"))
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        **judge_environment,
    }
    if experiment.requires_openrouter:
        env_variables.update(proxy_environment)
        env_variables["OPENROUTER_API_KEY_DECOMPOSER"] = openrouter_key
    payload: dict[str, Any] = {
        "script": build_job_script(
            staged_workdir,
            experiment,
            num_repeats,
            limit,
            force=force,
            purpose=purpose,
            partition=partition,
            concurrency=concurrency,
            rollout_offset=rollout_offset,
            prompt_profile=prompt_profile,
            domain=domain,
        ),
        "job_desc": build_job_desc(
            experiment,
            num_repeats,
            limit,
            author,
            purpose=purpose,
            partition=partition,
            rollout_offset=rollout_offset,
            prompt_profile=prompt_profile,
            domain=domain,
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
    num_repeats: int,
    limit: int | None,
    *,
    purpose: str = "evaluation",
    partition: str = "full",
    rollout_offset: int = 0,
    prompt_profile: str | None = None,
    domain: Gaia2Domain = DOMAIN,
) -> None:
    spec = get_domain_spec(domain)
    print("\nSelected jobs:")
    print("| # | Run | Agent | Split/domain | GPUs |")
    print("| ---: | --- | --- | --- | ---: |")
    for index, experiment in enumerate(experiments, start=1):
        identity = (
            trace_run_name(
                experiment,
                num_repeats,
                rollout_offset,
                prompt_profile=prompt_profile,
            )
            if purpose == "trace-generation"
            else run_name(experiment, num_repeats, prompt_profile=prompt_profile)
        )
        if limit is not None:
            identity += f"/smoke_{limit}"
        print(
            f"| {index} | `{identity}` | {experiment.kind} | "
            f"{SPLIT}/{spec.name}/{partition} | {experiment.num_gpus} |"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--filter", action="append")
    parser.add_argument("--split", choices=(SPLIT,), default=SPLIT)
    parser.add_argument("--domain", choices=DOMAINS, default=DOMAIN)
    parser.add_argument(
        "--purpose",
        choices=("evaluation", "trace-generation"),
        default="evaluation",
    )
    parser.add_argument("--partition", choices=PARTITIONS, default="full")
    parser.add_argument("--num-repeats", type=positive_int, default=3)
    parser.add_argument("--concurrency", type=positive_int)
    parser.add_argument("--prompt-profile", choices=DECOMPOSER_PROMPT_PROFILES)
    parser.add_argument("--rollout-offset", type=nonnegative_int, default=0)
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
    spec = get_domain_spec(args.domain)
    if args.purpose == "trace-generation":
        if not spec.supports_trace_generation:
            parser.error(f"{spec.name} does not support trace generation")
        if args.partition != "train":
            parser.error("trace generation is restricted to --partition train")
    if args.purpose == "evaluation" and args.partition not in ("full", "test"):
        parser.error("evaluation supports only --partition full or test")
    if args.purpose == "evaluation" and args.rollout_offset:
        parser.error("--rollout-offset is only valid for trace generation")
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
    if args.prompt_profile is not None and any(
        experiment.kind != "decomposer" for experiment in experiments
    ):
        parser.error("--prompt-profile is only valid for Decomposer experiments")

    selected_count = len(experiments)
    candidates: list[Experiment] = []
    skipped_completed = 0
    for experiment in experiments:
        marker = (
            trace_completion_marker(
                experiment,
                args.num_repeats,
                args.rollout_offset,
                args.partition,
                args.limit,
                prompt_profile=args.prompt_profile,
                domain=spec.name,
            )
            if args.purpose == "trace-generation"
            else completion_marker(
                experiment,
                args.num_repeats,
                args.limit,
                partition=args.partition,
                prompt_profile=args.prompt_profile,
                domain=spec.name,
            )
        )
        if marker.is_file() and not args.force:
            selected_experiment = select_prompt_profile(
                experiment, args.prompt_profile
            )
            validate_run_identity(
                marker,
                run_identity(
                    selected_experiment,
                    domain=spec.name,
                    purpose=args.purpose,
                    partition=args.partition,
                    num_repeats=args.num_repeats,
                    concurrency=args.concurrency or experiment.concurrency,
                    limit=args.limit,
                    rollout_offset=args.rollout_offset,
                ),
                require_complete=True,
            )
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
            "domain": spec.name,
            "purpose": args.purpose,
            "partition": args.partition,
            "num_repeats": args.num_repeats,
            "concurrency": args.concurrency,
            "rollout_offset": args.rollout_offset,
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
            validate_preparation(
                experiment, partition=args.partition, domain=spec.name
            )
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
    openrouter_key = os.environ.get("OPENROUTER_API_KEY_DECOMPOSER", "")
    needs_openrouter = any(experiment.requires_openrouter for experiment in experiments)
    if needs_openrouter and not args.dry:
        if not openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY_DECOMPOSER is required")
        proxy_environment = configured_proxy_environment(os.environ)
    else:
        proxy_environment = {
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
            num_repeats=args.num_repeats,
            limit=args.limit,
            author=args.author_name,
            base_image=args.base_image,
            priority=args.priority,
            force=args.force,
            judge_environment=judge_environment,
            proxy_environment=proxy_environment,
            openrouter_key=openrouter_key or "<not-set>",
            purpose=args.purpose,
            partition=args.partition,
            concurrency=args.concurrency,
            rollout_offset=args.rollout_offset,
            prompt_profile=args.prompt_profile,
            domain=spec.name,
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
            purpose=args.purpose,
            partition=args.partition,
            rollout_offset=args.rollout_offset,
            prompt_profile=args.prompt_profile,
            domain=spec.name,
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
                    "prompt_profile": args.prompt_profile,
                    "manager_prompt_addendum_profile": getattr(
                        experiment, "manager_prompt_addendum_profile", None
                    ),
                }
            )

    summary = {
        "selected": selected_count,
        "planned": len(planned),
        "split": SPLIT,
        "domain": spec.name,
        "purpose": args.purpose,
        "partition": args.partition,
        "num_repeats": args.num_repeats,
        "concurrency": args.concurrency,
        "prompt_profile": args.prompt_profile,
        "rollout_offset": args.rollout_offset,
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
