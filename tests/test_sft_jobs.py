from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

# The MLSpace launcher imports the `mls` submitter, which lives in its own
# environment (see sft/requirements-mlspace.txt).
pytest.importorskip("mls")

from sft.experiments import (
    INSTANCE_TYPES_BY_NUM_GPUS,
    ExperimentConfig,
    build_train_command,
    collect_experiments,
    has_experiment_artifacts,
    has_training_artifacts,
)
from sft.run_train_jobs import (
    _archive_output_dir,
    _build_job_script,
    _latest_checkpoint,
    _require_latest_checkpoint,
    _validate_clearml_config,
)


def _experiment(**overrides: object) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        config_path="sft/configs/smoke.yaml",
        description="SFT smoke",
        **overrides,
    )


def test_sft_experiments_are_unique() -> None:
    experiments = collect_experiments()
    assert len({experiment.name for experiment in experiments}) == len(experiments)
    assert INSTANCE_TYPES_BY_NUM_GPUS[2] == "a100plus.2gpu.80vG.24C.488G"
    assert INSTANCE_TYPES_BY_NUM_GPUS[4] == "a100plus.4gpu.80vG.48C.976G"


def test_build_train_command_uses_torchrun_and_explicit_liger_mode() -> None:
    native = _experiment()
    liger = _experiment(use_liger_kernel=True)
    native_command = build_train_command(
        native, workdir="/staged", output_dir="/artifacts/native"
    )
    liger_command = build_train_command(
        liger, workdir="/staged", output_dir="/artifacts/liger"
    )
    assert native_command[:3] == ["torchrun", "--standalone", "--nproc-per-node=2"]
    assert "/staged/sft/configs/smoke.yaml" in native_command
    assert "--clearml" in native_command
    assert "--no-use-liger-kernel" in native_command
    assert "--use-liger-kernel" in liger_command


def test_build_train_command_can_resume_an_explicit_checkpoint() -> None:
    native = _experiment()
    command = build_train_command(
        native,
        workdir="/staged",
        output_dir="/artifacts/native",
        resume_from_checkpoint="/artifacts/native/checkpoint-458",
    )
    assert command[-2:] == [
        "--resume-from-checkpoint",
        "/artifacts/native/checkpoint-458",
    ]


def test_job_script_captures_console_log_without_hiding_training_failure() -> None:
    script = _build_job_script(
        ["torchrun", "--standalone", "train.py"],
        workdir=Path("/staged"),
        venv=Path("/venv"),
        triton_cache=Path("/cache/triton"),
        output_dir=Path("/artifacts/run"),
    )
    assert "mkdir -p /artifacts/run" in script
    assert 'PYTHONPATH="$WORKDIR/src:$WORKDIR:' in script
    assert "bash -o pipefail -c" in script
    assert "/venv/bin/torchrun --standalone train.py" in script
    assert "tee -a /artifacts/run/console.log" in script


def test_job_script_adds_qwen35_runtime_overlays_and_fail_closed_environment() -> None:
    script = _build_job_script(
        ["torchrun", "--standalone", "train.py"],
        workdir=Path("/staged"),
        venv=Path("/venv"),
        triton_cache=Path("/cache/triton"),
        output_dir=Path("/artifacts/run"),
        runtime_bundle=Path("/runtime/qwen35"),
    )
    assert 'PYTHONPATH="/runtime/qwen35/triton:/runtime/qwen35/causal:' in script
    assert "export FLA_TILELANG=0" in script
    assert "DECOMPOSER_SFT_RUNTIME_PROFILE=qwen35-hf-fa2-fla-v1" in script
    assert "DECOMPOSER_SFT_RUNTIME_BUNDLE=/runtime/qwen35" in script


def test_force_archive_is_atomic_and_recoverable(tmp_path: Path) -> None:
    output = tmp_path / "jobs" / "experiment"
    output.mkdir(parents=True)
    (output / "training_summary.json").write_text("{}")
    archived = _archive_output_dir(
        output,
        archive_root=tmp_path / "jobs" / "_archive" / "experiment",
        commit="1234567890abcdef",
        timestamp=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
    )
    assert archived == (
        tmp_path
        / "jobs/_archive/experiment/20260831T120000Z-1234567890ab"
    )
    assert not output.exists()
    assert (archived / "training_summary.json").is_file()


def test_force_archive_ignores_missing_or_empty_output(tmp_path: Path) -> None:
    output = tmp_path / "empty"
    assert (
        _archive_output_dir(
            output,
            archive_root=tmp_path / "archive",
            commit="abc",
        )
        is None
    )
    output.mkdir()
    assert (
        _archive_output_dir(
            output,
            archive_root=tmp_path / "archive",
            commit="abc",
        )
        is None
    )


def test_clearml_config_must_be_private(tmp_path: Path) -> None:
    config = tmp_path / "clearml.conf"
    config.write_text("api {}")
    config.chmod(0o600)
    _validate_clearml_config(config)
    config.chmod(0o644)
    with pytest.raises(PermissionError, match="expected 0600 or stricter"):
        _validate_clearml_config(config)


def test_latest_checkpoint_is_numeric_and_requires_trainer_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    for step in (9, 100):
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}")
    (output / "checkpoint-1000").mkdir()
    (output / "checkpoint-invalid").mkdir()

    assert _latest_checkpoint(output) == output / "checkpoint-100"
    assert _latest_checkpoint(tmp_path / "missing") is None
    with pytest.raises(FileNotFoundError, match="No complete checkpoint-N"):
        _require_latest_checkpoint(tmp_path / "missing")


def test_training_completion_requires_summary_and_final_weights(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "final").mkdir(parents=True)
    (output / "training_summary.json").write_text("{}")
    assert not has_training_artifacts(output)
    (output / "final" / "model.safetensors").write_bytes(b"weights")
    assert has_training_artifacts(output)


def test_training_completion_accepts_sharded_safetensors_index(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "final").mkdir(parents=True)
    (output / "training_summary.json").write_text("{}")
    (output / "final" / "model.safetensors.index.json").write_text("{}")
    assert has_training_artifacts(output)


def test_benchmark_experiment_completion_uses_benchmark_summary(tmp_path: Path) -> None:
    smoke = _experiment(benchmark=True)
    output = tmp_path / "smoke"
    output.mkdir()
    assert not has_experiment_artifacts(smoke, output)
    (output / "benchmark_summary.json").write_text("{}")
    assert has_experiment_artifacts(smoke, output)
