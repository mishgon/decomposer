"""Dry-run or submit Decomposer SFT experiments to MLSpace."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mls.manager.job.dedup import in_progress_descs, normalize_job_desc
from mls.manager.job.redact import redact_payload
from mls.manager.job.staging import git_commit_hash, git_dirty_tracked, git_toplevel, stage_repo
from mls.manager.job.utils import run_job_with_retry, training_job_api_from_profile
from mls.manager.job.uv_env import uv_lock_hash

from .experiments import (
    INSTANCE_TYPES_BY_NUM_GPUS,
    build_train_command,
    collect_experiments,
    has_experiment_artifacts,
)
from .qwen35_fast_runtime import (
    DEFAULT_BUNDLE_DIR,
    PROFILE_NAME as QWEN35_FAST_PROFILE,
    runtime_pythonpath_entries,
    validate_runtime_bundle,
)


ARTIFACTS_ROOT = Path("/home/jovyan/decomposer-artifacts/training/sft/jobs")
ARTIFACTS_ROOT_SANITY = Path(
    "/home/jovyan/decomposer-artifacts/training/sft/jobs_sanity"
)
STAGING_ROOT = Path("/home/jovyan/decomposer-artifacts/code")
VENV_ROOT = Path("/home/jovyan/decomposer-artifacts/venvs/sft")
TRITON_CACHE_ROOT = Path("/home/jovyan/decomposer-artifacts/cache/triton/sft")
HF_HOME = Path("/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface")
CLEARML_CONFIG_FILE = Path(
    "/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.secrets/clearml.conf"
)
BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
_CHECKPOINT_NAME = re.compile(r"^checkpoint-(\d+)$")


def _ensure_training_venv(repo_root: str, venv: Path) -> None:
    """Sync base + train dependencies; binary overlays are prepared separately."""
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is required to create the shared training venv.")
    venv.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(venv),
        "UV_LINK_MODE": "copy",
    }
    subprocess.run(
        [
            uv,
            "sync",
            "--locked",
            "--no-install-project",
            "--no-default-groups",
            "--group",
            "train",
            "--no-progress",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )


def _validate_clearml_config(path: Path = CLEARML_CONFIG_FILE) -> None:
    """Require a private regular file without reading or copying credentials."""
    if not path.is_file():
        raise FileNotFoundError(f"ClearML config does not exist: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"ClearML config {path} has mode {mode:04o}; expected 0600 or stricter."
        )


def _latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if (
            match is not None
            and path.is_dir()
            and (path / "trainer_state.json").is_file()
        ):
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def _require_latest_checkpoint(output_dir: Path) -> Path:
    checkpoint = _latest_checkpoint(output_dir)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No complete checkpoint-N directory exists under {output_dir}."
        )
    return checkpoint


def _build_job_script(
    command: list[str],
    *,
    workdir: Path,
    venv: Path,
    triton_cache: Path,
    output_dir: Path,
    runtime_bundle: Path | None = None,
) -> str:
    if not command or command[0] != "torchrun":
        raise ValueError("SFT job commands must start with torchrun.")
    command = [str(venv / "bin" / "torchrun"), *command[1:]]
    console_log = output_dir / "console.log"
    logged_command = (
        f"{shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(console_log))}"
    )
    pythonpath_entries = ["$WORKDIR/src", "$WORKDIR"]
    if runtime_bundle is not None:
        pythonpath_entries = [
            *(str(path) for path in runtime_pythonpath_entries(runtime_bundle)),
            *pythonpath_entries,
        ]
    pythonpath = ":".join(pythonpath_entries)
    exports = [
        f"cd {shlex.quote(str(workdir))}",
        f"export VIRTUAL_ENV={shlex.quote(str(venv))}",
        'export PATH="$VIRTUAL_ENV/bin:$PATH"',
        f'export PYTHONPATH="{pythonpath}:${{PYTHONPATH:-}}"',
        f"export TRITON_CACHE_DIR={shlex.quote(str(triton_cache))}",
    ]
    if runtime_bundle is not None:
        exports.extend(
            [
                "export FLA_TILELANG=0",
                (
                    "export DECOMPOSER_SFT_RUNTIME_PROFILE="
                    f"{shlex.quote(QWEN35_FAST_PROFILE)}"
                ),
                (
                    "export DECOMPOSER_SFT_RUNTIME_BUNDLE="
                    f"{shlex.quote(str(runtime_bundle))}"
                ),
            ]
        )
    return " && ".join(
        [
            *exports,
            f"mkdir -p {shlex.quote(str(output_dir))}",
            f"bash -o pipefail -c {shlex.quote(logged_command)}",
        ]
    )


def _archive_output_dir(
    output_dir: Path,
    *,
    archive_root: Path,
    commit: str,
    timestamp: datetime | None = None,
) -> Path | None:
    """Atomically preserve non-empty output before a forced fresh run."""
    if not output_dir.exists():
        return None
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Training output is not a directory: {output_dir}")
    if not any(output_dir.iterdir()):
        return None
    timestamp = timestamp or datetime.now(UTC)
    label = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{commit[:12]}"
    target = archive_root / label
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Training archive already exists: {target}")
    output_dir.rename(target)
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--base-image", default=BASE_IMAGE)
    parser.add_argument("--author-name", default="sukhorukov")
    parser.add_argument("--dry", "--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--filter")
    parser.add_argument("--priority", choices=("low", "medium", "high"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_clearml_config()
    repo_root = git_toplevel(os.getcwd())
    commit = git_commit_hash(repo_root)
    venv = VENV_ROOT / uv_lock_hash(repo_root)

    if args.dry:
        workdir = Path(repo_root)
    else:
        dirty = git_dirty_tracked(repo_root)
        if dirty:
            print(
                "Tracked working tree is dirty; commit or stash before submission:",
                file=sys.stderr,
            )
            print("\n".join(dirty), file=sys.stderr)
            return 1
        _ensure_training_venv(repo_root, venv)
        workdir = STAGING_ROOT / commit
        stage_repo(repo_root, str(workdir))

    client, options = training_job_api_from_profile(args.profile)
    queued_descriptions = in_progress_descs(args.profile)
    artifacts_root = ARTIFACTS_ROOT_SANITY if args.sanity_check else ARTIFACTS_ROOT
    launched: list[dict[str, object]] = []
    selected = collect_experiments(args.filter)

    for experiment in selected:
        output_dir = artifacts_root / experiment.name
        if has_experiment_artifacts(experiment, output_dir) and not args.force:
            print(f"Skip (completed): {experiment.name}")
            continue
        if experiment.num_gpus not in INSTANCE_TYPES_BY_NUM_GPUS:
            raise ValueError(f"No instance type for {experiment.num_gpus} GPUs.")

        resume_from_checkpoint = None
        if args.resume_latest:
            resume_from_checkpoint = _require_latest_checkpoint(output_dir)
            print(f"Resume {experiment.name}: {resume_from_checkpoint}")

        runtime_bundle = None
        if experiment.runtime_profile is not None:
            if experiment.runtime_profile != QWEN35_FAST_PROFILE:
                raise ValueError(
                    f"Unsupported runtime profile: {experiment.runtime_profile!r}."
                )
            validate_runtime_bundle(DEFAULT_BUNDLE_DIR)
            runtime_bundle = DEFAULT_BUNDLE_DIR

        command = build_train_command(
            experiment,
            workdir=workdir,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        triton_cache = TRITON_CACHE_ROOT / experiment.name
        script = _build_job_script(
            command,
            workdir=workdir,
            venv=venv,
            triton_cache=triton_cache,
            output_dir=output_dir,
            runtime_bundle=runtime_bundle,
        )
        job_desc = f"🏋️ {experiment.description} #{args.author_name}"
        if normalize_job_desc(job_desc) in queued_descriptions:
            print(f"Skip (already queued): {experiment.name}")
            continue

        if args.force and not args.resume_latest and output_dir.exists():
            archive_root = artifacts_root / "_archive" / experiment.name
            if args.dry:
                if output_dir.is_dir() and any(output_dir.iterdir()):
                    print(f"Would archive existing output under {archive_root}")
            else:
                archived = _archive_output_dir(
                    output_dir,
                    archive_root=archive_root,
                    commit=commit,
                )
                if archived is not None:
                    print(f"Archived existing output: {archived}")

        payload = {
            "script": script,
            "job_desc": job_desc,
            "env_variables": {
                "WORKDIR": str(workdir),
                "CLEARML_CONFIG_FILE": str(CLEARML_CONFIG_FILE),
                "HF_HOME": str(HF_HOME),
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
                **(
                    {
                        "PYTORCH_CUDA_ALLOC_CONF": (
                            experiment.pytorch_cuda_alloc_conf
                        )
                    }
                    if experiment.pytorch_cuda_alloc_conf
                    else {}
                ),
            },
            "instance_type": INSTANCE_TYPES_BY_NUM_GPUS[experiment.num_gpus],
            "region": options["region"],
            "type": "binary_exp",
            "shm_size_class": "large",
            "base_image": args.base_image,
            "n_workers": 1,
            "processes_per_worker": 1,
            **({"priority_class": args.priority} if args.priority else {}),
        }
        print(f"Would launch [{experiment.num_gpus} GPU]: {job_desc}")
        print(script)
        if args.dry:
            print(json.dumps(redact_payload(payload), indent=2, ensure_ascii=False))
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
        print("result", result)

    summary = {
        "selected": len(selected),
        "launched": len(launched),
        "jobs": launched,
    }
    print("__SFT_TRAINING_JOBS_JSON__")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
