from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from gyms.toolathlon.mlspace_experiments import (
    INSTANCE_TYPES_BY_NUM_GPUS,
    collect_experiments,
)


ARTIFACTS_ROOT = Path(
    "/mnt/shared_ru.ml.SZ-5_000264/matrosov/decomposer-toolathlon-bench-artifacts"
)
STAGING_ROOT = ARTIFACTS_ROOT / "code"
SERVICES_ROOT = ARTIFACTS_ROOT / "inference"
HF_HOME = Path("/home/jovyan/.cache/huggingface")
MLSPACE_PYTHON = Path("/home/jovyan/.mlspace/envs/decomposer_jobs/bin/python")
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
TUNNEL_KEY = ARTIFACTS_ROOT / "secrets" / "hertz_tunnel"
KNOWN_HOSTS = ARTIFACTS_ROOT / "secrets" / "known_hosts"
HERTZ_HOST = "135.106.169.8"
HERTZ_PORT = 44444
_TAG_RE = re.compile(r"[#@]\S+")


def normalize_description(value: str) -> str:
    return " ".join(_TAG_RE.sub("", value).split())


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit Toolathlon benchmark Qwen inference services to MLSpace."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--author-name", required=True)
    parser.add_argument("--telegram-nick", required=True)
    parser.add_argument("--base-image", default=BASE_IMAGE)
    parser.add_argument("--dry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from mls.manager.job.redact import redact_payload
        from mls.manager.job.staging import git_dirty_tracked, stage_repo
        from mls.manager.job.utils import (
            get_in_progress_jobs,
            run_job_with_retry,
            training_job_api_from_profile,
        )
    except ImportError as error:
        print(f"Run this launcher in the configured decomposer_jobs env: {error}")
        return 2

    repo = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
    commit = git(repo, "rev-parse", "HEAD")
    if args.dry:
        staged_repo = repo
    else:
        dirty = git_dirty_tracked(str(repo))
        if dirty:
            print("Working tree is dirty; commit or stash before submission.")
            print("\n".join(dirty))
            return 1
        staged_repo = STAGING_ROOT / commit
        stage_repo(str(repo), str(staged_repo))
        if not TUNNEL_KEY.is_file() or not KNOWN_HOSTS.is_file():
            print(f"Missing dedicated tunnel credentials under {TUNNEL_KEY.parent}")
            return 1

    client, extra = training_job_api_from_profile(args.profile)
    active = {
        normalize_description(job.get("job_desc", ""))
        for job in get_in_progress_jobs(client_profile=args.profile)
    }
    launched = []
    for experiment in collect_experiments(pilot=args.pilot):
        output_dir = SERVICES_ROOT / experiment.name
        command = [
            str(MLSPACE_PYTHON),
            "-m",
            "gyms.toolathlon.mlspace_serve",
            "--gpu-count",
            str(experiment.num_gpus),
            "--remote-port-start",
            str(experiment.remote_port_start),
            "--hertz-host",
            HERTZ_HOST,
            "--hertz-port",
            str(HERTZ_PORT),
            "--ssh-key",
            str(TUNNEL_KEY),
            "--known-hosts",
            str(KNOWN_HOSTS),
            "--output-dir",
            str(output_dir),
        ]
        script = (
            f"cd {shlex.quote(str(staged_repo))} && "
            f"export PYTHONPATH={shlex.quote(str(staged_repo))}:${{PYTHONPATH:-}} && "
            f"exec {shlex.join(command)}"
        )
        description = (
            f"{experiment.description} #{args.author_name} @{args.telegram_nick}"
        )
        if normalize_description(description) in active:
            print(f"Skip (already Pending/Running): {description}")
            continue
        payload = {
            "script": script,
            "job_desc": description,
            "env_variables": {
                "HF_HOME": str(HF_HOME),
                "WORKDIR": str(staged_repo),
            },
            "instance_type": INSTANCE_TYPES_BY_NUM_GPUS[experiment.num_gpus],
            "region": extra["region"],
            "type": "binary_exp",
            "shm_size_class": "large",
            "base_image": args.base_image,
            "n_workers": 1,
            "processes_per_worker": 1,
            "priority_class": "high",
        }
        print(f"Would launch [{experiment.num_gpus} GPU]: {description}")
        if args.dry:
            print(json.dumps(redact_payload(payload), indent=2))
            continue
        result = run_job_with_retry(client, payload, profile=args.profile)
        launched.append(result)
        print("result", result)

    print("__TOOLATHLON_BENCH_INFERENCE_JOBS_JSON__")
    print(json.dumps({"jobs": launched, "launched": len(launched)}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
