"""Discover SFT checkpoints and dry-run or submit an explicit eval entrypoint.

This branch does not yet contain a benchmark evaluation adapter. A fresh
artifacts root therefore exits successfully with "nothing to evaluate". Once
an adapter exists, pass its module through ``--entrypoint``; it must accept
``--model`` and ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

from mls.manager.job.dedup import in_progress_descs, normalize_job_desc
from mls.manager.job.redact import redact_payload
from mls.manager.job.staging import git_commit_hash, git_dirty_tracked, git_toplevel, stage_repo
from mls.manager.job.utils import run_job_with_retry, training_job_api_from_profile
from mls.manager.job.uv_env import uv_lock_hash

from .experiments import INSTANCE_TYPES_BY_NUM_GPUS
from .run_train_jobs import (
    BASE_IMAGE,
    HF_HOME,
    STAGING_ROOT,
    VENV_ROOT,
    _ensure_training_venv,
)


ARTIFACTS_ROOT = Path("/home/jovyan/decomposer-artifacts/training/sft/jobs")


def discover_models(root: Path) -> list[Path]:
    candidates = [path for path in root.rglob("final") if path.is_dir()]
    candidates.extend(path for path in root.rglob("checkpoint-*") if path.is_dir())
    return sorted(
        path
        for path in candidates
        if (path / "model.safetensors").is_file()
        or (path / "pytorch_model_fsdp_0").is_dir()
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--entrypoint", help="Python module implementing evaluation.")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--base-image", default=BASE_IMAGE)
    parser.add_argument("--author-name", default="sukhorukov")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--dry", "--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--priority", choices=("low", "medium", "high"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args, extra = _build_parser().parse_known_args(argv)
    models = discover_models(args.checkpoints_dir.resolve())
    if not models:
        print(f"No trained SFT models under {args.checkpoints_dir}; nothing to evaluate.")
        return 0
    if not args.entrypoint:
        print(
            "SFT models exist, but this branch has no evaluation adapter; "
            "pass --entrypoint once one is implemented.",
            file=sys.stderr,
        )
        return 2
    if args.num_gpus not in INSTANCE_TYPES_BY_NUM_GPUS:
        raise ValueError(f"No instance type for {args.num_gpus} GPUs.")

    repo_root = git_toplevel(os.getcwd())
    commit = git_commit_hash(repo_root)
    venv = VENV_ROOT / uv_lock_hash(repo_root)
    if args.dry:
        workdir = Path(repo_root)
    else:
        dirty = git_dirty_tracked(repo_root)
        if dirty:
            print("Tracked working tree is dirty; commit or stash first.", file=sys.stderr)
            return 1
        _ensure_training_venv(repo_root, venv)
        workdir = STAGING_ROOT / commit
        stage_repo(repo_root, str(workdir))

    client, options = training_job_api_from_profile(args.profile)
    queued_descriptions = in_progress_descs(args.profile)
    launched: list[dict[str, str]] = []
    for model in models:
        output_dir = model / "evaluation"
        marker = output_dir / "summary.json"
        if marker.is_file() and not args.force:
            print(f"Skip (completed): {model}")
            continue
        inner = [
            "python",
            "-m",
            args.entrypoint,
            "--model",
            str(model),
            "--output-dir",
            str(output_dir),
            *extra,
        ]
        script = " && ".join(
            [
                f"cd {shlex.quote(str(workdir))}",
                f"export VIRTUAL_ENV={shlex.quote(str(venv))}",
                'export PATH="$VIRTUAL_ENV/bin:$PATH"',
                'export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"',
                shlex.join(inner),
            ]
        )
        job_desc = f"📐 SFT eval {model.parent.name}/{model.name} #{args.author_name}"
        if normalize_job_desc(job_desc) in queued_descriptions:
            print(f"Skip (already queued): {model}")
            continue
        payload = {
            "script": script,
            "job_desc": job_desc,
            "env_variables": {
                "WORKDIR": str(workdir),
                "HF_HOME": str(HF_HOME),
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "instance_type": INSTANCE_TYPES_BY_NUM_GPUS[args.num_gpus],
            "region": options["region"],
            "type": "binary_exp",
            "shm_size_class": "large",
            "base_image": args.base_image,
            "n_workers": 1,
            "processes_per_worker": 1,
            **({"priority_class": args.priority} if args.priority else {}),
        }
        print(f"Would launch [{args.num_gpus} GPU]: {job_desc}")
        print(script)
        if args.dry:
            print(json.dumps(redact_payload(payload), indent=2, ensure_ascii=False))
            continue
        result = run_job_with_retry(client, payload, profile=args.profile)
        job_name = result.get("job_name") if isinstance(result, dict) else None
        if job_name:
            launched.append({"job_name": job_name, "model": str(model)})
        print("result", result)

    print("__SFT_EVAL_JOBS_JSON__")
    print(json.dumps({"launched": len(launched), "jobs": launched}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
