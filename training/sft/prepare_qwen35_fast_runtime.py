"""Build or verify the reusable Qwen3.5 H100 training runtime bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .qwen35_fast_runtime import (
    DEFAULT_BUNDLE_DIR,
    HF_FA2_REPOSITORY,
    HF_FA2_REVISION,
    build_bundle_manifest,
    validate_runtime_bundle,
)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def _build_environment(cuda_home: Path, *, max_jobs: int) -> dict[str, str]:
    target = cuda_home / "targets/x86_64-linux"
    if not (cuda_home / "bin/nvcc").is_file():
        raise FileNotFoundError(f"CUDA nvcc does not exist under {cuda_home}.")
    if not (target / "include").is_dir() or not (target / "lib").is_dir():
        raise FileNotFoundError(
            f"CUDA target headers or libraries do not exist under {target}."
        )
    path = os.environ.get("PATH", "")
    cpath = os.environ.get("CPATH", "")
    library_path = os.environ.get("LIBRARY_PATH", "")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    return {
        **os.environ,
        "CUDA_HOME": str(cuda_home),
        "PATH": f"{cuda_home / 'bin'}:{path}",
        "CPATH": f"{target / 'include'}{':' + cpath if cpath else ''}",
        "LIBRARY_PATH": (
            f"{target / 'lib'}{':' + library_path if library_path else ''}"
        ),
        "LD_LIBRARY_PATH": (
            f"{target / 'lib'}{':' + ld_library_path if ld_library_path else ''}"
        ),
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "MAX_JOBS": str(max_jobs),
        "TORCH_CUDA_ARCH_LIST": "9.0",
    }


def _install_overlay(
    uv: str,
    target: Path,
    requirement: str,
    *,
    env: dict[str, str] | None = None,
    build: bool = False,
) -> None:
    command = [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--target",
        str(target),
        "--no-deps",
    ]
    if build:
        command.append("--no-build-isolation")
    command.append(requirement)
    _run(command, env=env)


def _prefetch_hf_kernel(cache_dir: Path | None) -> str:
    if cache_dir is not None:
        os.environ["HF_HOME"] = str(cache_dir)
    from kernels import get_kernel

    module = get_kernel(
        HF_FA2_REPOSITORY,
        revision=HF_FA2_REVISION,
    )
    return str(module.__file__)


def build_runtime_bundle(
    bundle_dir: Path,
    *,
    cuda_home: Path,
    max_jobs: int,
    hf_cache_dir: Path | None,
    prefetch_hf: bool,
) -> dict:
    bundle_dir = bundle_dir.resolve()
    if bundle_dir.exists():
        raise FileExistsError(
            f"Runtime bundle already exists: {bundle_dir}. Verify it or choose a new path."
        )
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is required to prepare the runtime bundle.")
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{bundle_dir.name}.tmp.", dir=bundle_dir.parent)
    )
    try:
        build_env = _build_environment(cuda_home.resolve(), max_jobs=max_jobs)
        _run([str(cuda_home.resolve() / "bin/nvcc"), "--version"], env=build_env)
        _install_overlay(uv, temporary / "triton", "triton==3.7.1")
        _install_overlay(
            uv,
            temporary / "causal",
            "causal-conv1d==1.7.0",
            env=build_env,
            build=True,
        )
        if prefetch_hf:
            location = _prefetch_hf_kernel(hf_cache_dir)
            print(f"Prefetched HF FlashAttention kernel to {location}", flush=True)
        manifest = build_bundle_manifest(temporary)
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.rename(bundle_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_runtime_bundle(bundle_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--max-jobs", type=int, default=4)
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--no-prefetch-hf", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.max_jobs <= 0:
        raise ValueError("--max-jobs must be positive.")
    if args.verify_only:
        manifest = validate_runtime_bundle(args.bundle_dir)
    else:
        if args.cuda_home is None:
            raise ValueError("--cuda-home is required when building a runtime bundle.")
        manifest = build_runtime_bundle(
            args.bundle_dir,
            cuda_home=args.cuda_home,
            max_jobs=args.max_jobs,
            hf_cache_dir=args.hf_cache_dir,
            prefetch_hf=not args.no_prefetch_hf,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
