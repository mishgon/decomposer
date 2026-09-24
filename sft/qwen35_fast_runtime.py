"""Pinned runtime identity for accelerated Qwen3.5 SFT jobs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

PROFILE_NAME = "qwen35-hf-fa2-fla-v1"
DEFAULT_BUNDLE_DIR = Path(
    "/home/jovyan/decomposer-artifacts/kernels/sft/qwen35-hf-fa2-fla-v1"
)
BUNDLE_FORMAT_VERSION = 1
HF_FA2_REPOSITORY = "kernels-community/flash-attn2"
HF_FA2_REVISION = "c269cc539ad0c1fc0899abd4b05ecc1303d6c4b1"
HF_FA2_IMPLEMENTATION = f"{HF_FA2_REPOSITORY}@{HF_FA2_REVISION}"

EXPECTED_RUNTIME = {
    "python_abi": "cp312",
    "torch": "2.11.0+cu129",
    "torch_cuda": "12.9",
    "gpu_arch": "9.0",
    "packages": {
        "kernels": "0.15.2",
        "flash-linear-attention": "0.5.2",
        "fla-core": "0.5.2",
        "causal-conv1d": "1.7.0",
        "triton": "3.7.1",
        "transformers": "5.14.1",
    },
}

JsonObject = dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_pythonpath_entries(bundle_dir: str | Path) -> tuple[Path, Path]:
    root = Path(bundle_dir).resolve()
    return root / "triton", root / "causal"


def _require_relative_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Runtime bundle file paths must be non-empty strings.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Runtime bundle file path is not portable: {value!r}.")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Runtime bundle file does not exist: {path}")
    return path


def validate_runtime_bundle(
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    *,
    expected_profile: str = PROFILE_NAME,
    verify_checksums: bool = True,
) -> JsonObject:
    """Validate the immutable bundle metadata and critical file checksums."""
    root = Path(bundle_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Qwen3.5 fast-runtime manifest does not exist: {manifest_path}"
        )
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("Runtime bundle manifest must be a JSON object.")
    if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("Runtime bundle manifest format version does not match.")
    if manifest.get("profile") != expected_profile:
        raise ValueError(
            f"Runtime bundle profile is {manifest.get('profile')!r}, "
            f"expected {expected_profile!r}."
        )
    identity = manifest.get("runtime")
    if identity != EXPECTED_RUNTIME:
        raise ValueError(
            "Runtime bundle identity does not match the checked-in runtime spec."
        )
    overlays = manifest.get("overlays")
    if overlays != {"triton": "triton", "causal_conv1d": "causal"}:
        raise ValueError("Runtime bundle overlay layout does not match.")
    for overlay in runtime_pythonpath_entries(root):
        if not overlay.is_dir():
            raise FileNotFoundError(f"Runtime bundle overlay does not exist: {overlay}")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Runtime bundle manifest has no critical file checksums.")
    seen: set[str] = set()
    for metadata in files:
        if not isinstance(metadata, Mapping):
            raise ValueError("Runtime bundle file metadata must be objects.")
        relative = metadata.get("path")
        path = _require_relative_file(root, relative)
        assert isinstance(relative, str)
        if relative in seen:
            raise ValueError(f"Duplicate runtime bundle file metadata: {relative}")
        seen.add(relative)
        if metadata.get("bytes") != path.stat().st_size:
            raise ValueError(f"Runtime bundle file size changed: {path}")
        if verify_checksums and metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"Runtime bundle file checksum changed: {path}")
    return manifest


def critical_bundle_files(bundle_dir: str | Path) -> list[Path]:
    """Resolve the files whose content pins the reusable binary overlays."""
    root = Path(bundle_dir).resolve()
    causal_extensions = sorted(root.glob("causal/causal_conv1d_cuda*.so"))
    if len(causal_extensions) != 1:
        raise ValueError(
            "The runtime bundle must contain exactly one causal_conv1d CUDA extension."
        )
    files = [
        root / "triton/triton/__init__.py",
        root / "triton/triton-3.7.1.dist-info/METADATA",
        root / "triton/triton/backends/nvidia/bin/ptxas",
        root / "causal/causal_conv1d/__init__.py",
        root / "causal/causal_conv1d-1.7.0.dist-info/METADATA",
        causal_extensions[0],
    ]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Runtime bundle is missing critical files: "
            + ", ".join(str(path) for path in missing)
        )
    return files


def build_bundle_manifest(bundle_dir: str | Path) -> JsonObject:
    root = Path(bundle_dir).resolve()
    files = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in critical_bundle_files(root)
    ]
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        "profile": PROFILE_NAME,
        "runtime": deepcopy(EXPECTED_RUNTIME),
        "hf_flash_attention": {
            "repository": HF_FA2_REPOSITORY,
            "revision": HF_FA2_REVISION,
            "attention_implementation": HF_FA2_IMPLEMENTATION,
        },
        "overlays": {"triton": "triton", "causal_conv1d": "causal"},
        "files": files,
    }
