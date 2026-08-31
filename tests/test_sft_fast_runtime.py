from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.sft.qwen35_fast_runtime import (
    EXPECTED_RUNTIME,
    HF_FA2_IMPLEMENTATION,
    PROFILE_NAME,
    build_bundle_manifest,
    runtime_pythonpath_entries,
    validate_runtime_bundle,
)


def _fixture_bundle(root: Path) -> dict:
    files = {
        "triton/triton/__init__.py": b'__version__ = "3.7.1"\n',
        "triton/triton-3.7.1.dist-info/METADATA": b"Version: 3.7.1\n",
        "triton/triton/backends/nvidia/bin/ptxas": b"ptxas",
        "causal/causal_conv1d/__init__.py": b"from .interface import *\n",
        "causal/causal_conv1d-1.7.0.dist-info/METADATA": b"Version: 1.7.0\n",
        "causal/causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so": b"extension",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = build_bundle_manifest(root)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_runtime_bundle_manifest_is_portable_and_checksum_validated(
    tmp_path: Path,
) -> None:
    manifest = _fixture_bundle(tmp_path)
    assert manifest["profile"] == PROFILE_NAME
    assert manifest["runtime"] == EXPECTED_RUNTIME
    assert (
        manifest["hf_flash_attention"]["attention_implementation"]
        == HF_FA2_IMPLEMENTATION
    )
    assert validate_runtime_bundle(tmp_path) == manifest
    assert runtime_pythonpath_entries(tmp_path) == (
        tmp_path / "triton",
        tmp_path / "causal",
    )
    assert all(not Path(item["path"]).is_absolute() for item in manifest["files"])


def test_runtime_bundle_rejects_critical_file_mutation(tmp_path: Path) -> None:
    _fixture_bundle(tmp_path)
    (tmp_path / "causal/causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so").write_bytes(
        b"changed!!"
    )
    with pytest.raises(ValueError, match="checksum changed"):
        validate_runtime_bundle(tmp_path)


def test_runtime_bundle_can_skip_repeated_hashing_inside_torchrun(tmp_path: Path) -> None:
    _fixture_bundle(tmp_path)
    extension = (
        tmp_path / "causal/causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so"
    )
    replacement = b"same-size"
    assert len(replacement) == extension.stat().st_size
    extension.write_bytes(replacement)
    assert validate_runtime_bundle(tmp_path, verify_checksums=False)["profile"] == PROFILE_NAME


def test_runtime_bundle_rejects_changed_identity(tmp_path: Path) -> None:
    manifest = _fixture_bundle(tmp_path)
    manifest["runtime"]["packages"]["triton"] = "3.6.0"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity does not match"):
        validate_runtime_bundle(tmp_path)
