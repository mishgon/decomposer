from __future__ import annotations

import json
from pathlib import Path

import pytest

from gyms.gaia2 import prepare
from gyms.gaia2.dataset import (
    validate_materialized_dataset,
    validate_materialized_filesystem,
    write_filesystem_manifest,
    write_materialized_dataset,
)
from gyms.gaia2.experiments import (
    DATASET_REVISION,
    DEEPSEEK_GEMMA_EXPERIMENT,
    DEEPSEEK_QWEN_EXPERIMENT,
    DECOMPOSER_EXPERIMENT,
    DOMAIN,
    INSTANCE_TYPES_BY_NUM_GPUS,
    SCENARIO_COUNT,
    SIMPLE_EXPERIMENT,
    SPLIT,
    collect_experiments,
    filesystem_dir,
    output_dir,
)
from gyms.gaia2.run import (
    _base_environment,
    _runtime_configs,
    are_command,
    decomposer_vllm_commands,
    simple_vllm_command,
    validate_result,
)
from gyms.gaia2.run_eval import build_payload, normalize_job_desc, redact_payload


def _rows():
    for index in range(SCENARIO_COUNT):
        scenario_id = f"scenario_{index:03d}"
        yield {
            "scenario_id": scenario_id,
            "split": SPLIT,
            "data": json.dumps(
                {"metadata": {"definition": {"scenario_id": scenario_id}}}
            ),
        }


def test_dataset_materialization_is_revisioned_and_integrity_checked(tmp_path) -> None:
    revision_root = tmp_path / DATASET_REVISION
    manifest = write_materialized_dataset(
        _rows(), revision_root, gaia2_revision="gaia-commit"
    )

    assert manifest["rows"] == SCENARIO_COUNT
    assert manifest["domain"] == DOMAIN
    assert manifest["split"] == SPLIT
    scenario = revision_root / SPLIT / DOMAIN / "scenario_000.json"
    assert scenario.is_file()
    assert (
        validate_materialized_dataset(revision_root)["aggregate_sha256"]
        == manifest["aggregate_sha256"]
    )

    scenario.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or is missing"):
        validate_materialized_dataset(revision_root)


def test_filesystem_mirror_manifest_detects_asset_changes(tmp_path) -> None:
    directory = tmp_path / "demo_filesystem" / "Documents"
    directory.mkdir(parents=True)
    asset = directory / "example.txt"
    asset.write_text("original", encoding="utf-8")

    manifest = write_filesystem_manifest(tmp_path)

    assert manifest["file_count"] == 1
    assert manifest["total_bytes"] == len("original")
    assert (
        validate_materialized_filesystem(tmp_path)["aggregate_sha256"]
        == manifest["aggregate_sha256"]
    )

    asset.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum changed"):
        validate_materialized_filesystem(tmp_path)


def test_experiment_registry_contains_local_and_openrouter_profiles() -> None:
    assert collect_experiments() == [
        DECOMPOSER_EXPERIMENT,
        DEEPSEEK_GEMMA_EXPERIMENT,
        DEEPSEEK_QWEN_EXPERIMENT,
        SIMPLE_EXPERIMENT,
    ]
    assert INSTANCE_TYPES_BY_NUM_GPUS == {
        1: "a100plus.1gpu.80vG.12C.182G",
        2: "a100plus.2gpu.80vG.24C.364G",
    }
    assert output_dir(SIMPLE_EXPERIMENT, 3).parts[-3:] == (
        SPLIT,
        DOMAIN,
        "gemma4-e4b-it-thinking-n3",
    )


def test_vllm_commands_use_current_e4b_thinking_profiles() -> None:
    simple = simple_vllm_command(SIMPLE_EXPERIMENT)
    manager, worker = decomposer_vllm_commands(DECOMPOSER_EXPERIMENT)

    assert "gemma4" in simple
    assert '{"enable_thinking":true}' in simple
    assert '{"enable_thinking":false}' in manager
    assert '{"enable_thinking":true}' in worker
    assert str(DECOMPOSER_EXPERIMENT.manager_checkpoint) in manager
    assert str(DECOMPOSER_EXPERIMENT.worker_checkpoint) in worker


def test_openrouter_decomposer_starts_only_the_configured_worker() -> None:
    gemma_manager, gemma_worker = decomposer_vllm_commands(
        DEEPSEEK_GEMMA_EXPERIMENT
    )
    qwen_manager, qwen_worker = decomposer_vllm_commands(DEEPSEEK_QWEN_EXPERIMENT)

    assert gemma_manager is None
    assert qwen_manager is None
    assert "gemma4" in gemma_worker
    assert "--language-model-only" in gemma_worker
    assert "qwen3_xml" in qwen_worker
    assert "--reasoning-parser" not in qwen_worker
    assert "--language-model-only" not in qwen_worker
    assert "--trust-remote-code" in qwen_worker
    assert qwen_worker[qwen_worker.index("--gdn-prefill-backend") + 1] == "triton"


def test_openrouter_runtime_uses_teacher_responses_api(tmp_path) -> None:
    _, plugin_path = _runtime_configs(
        Path(__file__).resolve().parents[2],
        tmp_path,
        DEEPSEEK_GEMMA_EXPERIMENT,
    )
    plugin = json.loads(plugin_path.read_text(encoding="utf-8"))
    service = plugin["service_configuration"]

    assert service["decomposer_system_prompt_profile"] == "teacher"
    assert service["manager"] == {
        "model": "deepseek/deepseek-v4-flash-0731",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY_DECOMPOSER",
        "temperature": 1.0,
        "top_p": 1.0,
        "use_responses_api": True,
        "reasoning": {"effort": "high"},
        "timeout": 3300,
        "max_retries": 2,
    }
    assert "path" not in plugin["model_configuration"]["manager"]


def test_openrouter_preparation_hashes_only_the_local_worker(monkeypatch) -> None:
    calls = []

    def fake_validate_checkpoint(path, *, full_hashes):
        calls.append((path, full_hashes))
        return {"path": str(path)}

    monkeypatch.setattr(prepare, "validate_checkpoint", fake_validate_checkpoint)

    models = prepare.experiment_models(
        DEEPSEEK_QWEN_EXPERIMENT,
        full_hashes=False,
    )

    assert calls == [(DEEPSEEK_QWEN_EXPERIMENT.worker_checkpoint, False)]
    assert models["manager"] == {
        "backend": "openrouter",
        "model": "deepseek/deepseek-v4-flash-0731",
    }


def test_are_commands_use_local_execution_dataset(tmp_path) -> None:
    benchmark = tmp_path / "are-benchmark"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    simple = are_command(
        SIMPLE_EXPERIMENT,
        benchmark=benchmark,
        dataset_root=dataset,
        output=output,
        judge_endpoint="https://judge.test/v1",
        num_repeats=3,
        limit=2,
        plugin_config=None,
    )
    decomposer = are_command(
        DECOMPOSER_EXPERIMENT,
        benchmark=benchmark,
        dataset_root=dataset,
        output=output,
        judge_endpoint="https://judge.test/v1",
        num_repeats=3,
        limit=None,
        plugin_config=tmp_path / "plugin.json",
    )

    assert simple[simple.index("-d") + 1] == str(dataset)
    assert "--hf-dataset" not in simple
    assert simple[simple.index("--num_runs") + 1] == "3"
    assert simple[simple.index("--limit") + 1] == "2"
    assert "native_tools" in simple
    assert "gyms.gaia2.plugin:create_plugin" in decomposer
    assert "thread" in decomposer


def test_runtime_environment_separates_judge_and_local_vllm_credentials(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "inherited-key")

    environment = _base_environment(
        tmp_path / "decomposer",
        tmp_path / "gaia2",
        tmp_path / "output",
        "judge-key",
    )

    assert "VLLM_API_KEY" not in environment
    assert environment["OPENAI_API_KEY"] == "judge-key"
    assert environment["ARE_JUDGE_SAMPLING_PARAMS"] == '{"temperature":0.0}'
    assert environment["DEMO_FS_PATH"] == str(filesystem_dir())


def test_result_validation_keeps_failed_rollouts_in_fixed_denominator(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    rows = [
        {"task_id": "a", "score": 1.0, "metadata": {"has_exception": False}},
        {
            "task_id": "a",
            "score": 0.0,
            "metadata": {"has_exception": True, "exception_type": "RemoteError"},
        },
        {"task_id": "a", "score": 0.0, "metadata": {"has_exception": False}},
    ]
    (output / "output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    metrics = validate_result(output, num_repeats=3, limit=1)

    assert metrics["rollout_rows"] == 3
    assert metrics["passed_rollouts"] == 1
    assert metrics["failed_rollouts"] == 2
    assert metrics["fixed_denominator_score"] == pytest.approx(1 / 3)
    assert metrics["exception_types"] == {"RemoteError": 1}


def test_mlspace_payload_uses_registry_gpu_type_and_redactable_judge_key(
    tmp_path,
) -> None:
    payload = build_payload(
        DECOMPOSER_EXPERIMENT,
        tmp_path / "staged",
        num_repeats=3,
        limit=None,
        author="sukhorukov",
        base_image="image",
        priority=None,
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "secret",
        },
        proxy_environment={},
        openrouter_key="<not-set>",
    )

    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[2]
    assert "--num-repeats 3" in payload["script"]
    assert "--cuda-visible-devices 0,1" in payload["script"]
    assert payload["env_variables"]["LLM_PROXY_MASTER_KEY"] == "secret"
    assert normalize_job_desc(payload["job_desc"]) == normalize_job_desc(
        payload["job_desc"] + " @someone"
    )


def test_openrouter_mlspace_payload_uses_one_gpu_and_redacts_credentials(
    tmp_path,
) -> None:
    payload = build_payload(
        DEEPSEEK_QWEN_EXPERIMENT,
        tmp_path / "staged",
        num_repeats=1,
        limit=1,
        author="sukhorukov",
        base_image="image",
        priority="high",
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "judge-secret",
        },
        proxy_environment={"HTTPS_PROXY": "http://user:pass@proxy.test"},
        openrouter_key="openrouter-secret",
    )

    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert "--cuda-visible-devices 0" in payload["script"]
    assert payload["env_variables"]["OPENROUTER_API_KEY_DECOMPOSER"] == (
        "openrouter-secret"
    )
    redacted = redact_payload(payload)
    assert redacted["env_variables"]["OPENROUTER_API_KEY_DECOMPOSER"] == "<redacted>"
    assert redacted["env_variables"]["HTTPS_PROXY"] == "<redacted>"
