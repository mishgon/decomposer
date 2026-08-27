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
    BASE_IMAGE,
    DATASET_REVISION,
    DEEPSEEK_GEMMA_EXPERIMENT,
    DEEPSEEK_QWEN_EXPERIMENT,
    DECOMPOSER_EXPERIMENT,
    DOMAIN,
    INSTANCE_TYPES_BY_NUM_GPUS,
    QWEN35_BASE_DECOMPOSER_EXPERIMENT,
    QWEN35_MIXED_SFT_EXPERIMENT,
    QWEN35_SFT_EXPERIMENT,
    SCENARIO_COUNT,
    SIMPLE_DEEPSEEK_EXPERIMENT,
    SIMPLE_EXPERIMENT,
    SIMPLE_QWEN_2B_EXPERIMENT,
    SIMPLE_QWEN_9B_EXPERIMENT,
    SIMPLE_QWEN_EXPERIMENT,
    SPLIT,
    collect_experiments,
    filesystem_dir,
    output_dir,
)
from gyms.gaia2.run import (
    _base_environment,
    _dry_plan,
    _runtime_configs,
    are_command,
    decomposer_vllm_commands,
    openrouter_proxy_command,
    selected_cuda_devices,
    simple_vllm_command,
    simple_sampling_parameters,
    subagent_environment,
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
        QWEN35_SFT_EXPERIMENT,
        QWEN35_MIXED_SFT_EXPERIMENT,
        QWEN35_BASE_DECOMPOSER_EXPERIMENT,
        DEEPSEEK_QWEN_EXPERIMENT,
        SIMPLE_EXPERIMENT,
        SIMPLE_QWEN_EXPERIMENT,
        SIMPLE_QWEN_2B_EXPERIMENT,
        SIMPLE_QWEN_9B_EXPERIMENT,
        SIMPLE_DEEPSEEK_EXPERIMENT,
    ]
    assert BASE_IMAGE.endswith("py3.12-torch2.7.0:0.0.42")
    assert INSTANCE_TYPES_BY_NUM_GPUS == {
        1: "a100plus.1gpu.80vG.12C.182G",
        2: "a100plus.2gpu.80vG.24C.364G",
    }
    assert all(
        not experiment.manager_parallel_tool_calls
        for experiment in (
            DECOMPOSER_EXPERIMENT,
            DEEPSEEK_GEMMA_EXPERIMENT,
            QWEN35_SFT_EXPERIMENT,
            QWEN35_MIXED_SFT_EXPERIMENT,
            QWEN35_BASE_DECOMPOSER_EXPERIMENT,
            DEEPSEEK_QWEN_EXPERIMENT,
        )
    )
    assert output_dir(SIMPLE_EXPERIMENT, 3).parts[-3:] == (
        SPLIT,
        DOMAIN,
        "gemma4-e4b-it-thinking-n3",
    )
    assert output_dir(SIMPLE_QWEN_EXPERIMENT, 3).parts[-1] == (
        "qwen35-4b-non-thinking-n3"
    )
    assert output_dir(SIMPLE_QWEN_2B_EXPERIMENT, 3).parts[-1] == (
        "qwen35-2b-base-non-thinking-n3"
    )
    assert output_dir(SIMPLE_QWEN_9B_EXPERIMENT, 3).parts[-1] == (
        "qwen35-9b-base-non-thinking-n3"
    )
    assert output_dir(SIMPLE_DEEPSEEK_EXPERIMENT, 3).parts[-1] == (
        "deepseek-v4-flash-0731-n3"
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


def test_simple_qwen_uses_qwen_runtime_and_sampling_profile() -> None:
    for experiment in (
        SIMPLE_QWEN_EXPERIMENT,
        SIMPLE_QWEN_2B_EXPERIMENT,
        SIMPLE_QWEN_9B_EXPERIMENT,
    ):
        command = simple_vllm_command(experiment)

        assert str(experiment.checkpoint) in command
        assert command[command.index("--max-model-len") + 1] == "131072"
        assert command[command.index("--tool-call-parser") + 1] == "qwen3_xml"
        assert "--reasoning-parser" not in command
        assert "--language-model-only" not in command
        assert "--trust-remote-code" in command
        assert command[command.index("--gdn-prefill-backend") + 1] == "triton"
        assert '{"enable_thinking":false}' in command
        assert simple_sampling_parameters(experiment) == {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 4096,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }


def test_simple_deepseek_uses_local_credential_proxy_without_gpu() -> None:
    experiment = SIMPLE_DEEPSEEK_EXPERIMENT

    assert experiment.requires_openrouter
    assert experiment.num_gpus == 0
    assert selected_cuda_devices(experiment, None) == ()
    assert experiment.remote_extra_body == {"reasoning": {"effort": "high"}}
    assert simple_sampling_parameters(experiment) == {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 4096,
    }
    with pytest.raises(ValueError, match="do not start a local vLLM"):
        simple_vllm_command(experiment)

    command = openrouter_proxy_command(experiment)
    assert "gyms.gaia2.openrouter_proxy" in command
    assert "OPENROUTER_API_KEY_DECOMPOSER" in command
    assert not any("sk-or-" in item for item in command)
    assert '{"reasoning":{"effort":"high"}}' in command

    plan = _dry_plan(Path.cwd(), experiment, Path("/tmp/deepseek"), (), 3, None)
    assert plan["gpu_assignments"] == {}
    assert len(plan["services"]) == 1
    assert "openrouter_proxy" in plan["services"][0]


def test_openrouter_decomposer_starts_only_the_configured_worker() -> None:
    gemma_manager, gemma_worker = decomposer_vllm_commands(DEEPSEEK_GEMMA_EXPERIMENT)
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


@pytest.mark.parametrize(
    "experiment",
    [QWEN35_SFT_EXPERIMENT, QWEN35_MIXED_SFT_EXPERIMENT],
)
def test_qwen_sft_decomposer_uses_qwen_manager_and_worker_profiles(
    experiment,
) -> None:
    manager, worker = decomposer_vllm_commands(experiment)

    assert manager is not None
    assert str(experiment.manager_checkpoint) in manager
    assert str(experiment.worker_checkpoint) in worker
    for command in (manager, worker):
        assert "qwen3_xml" in command
        assert "--reasoning-parser" not in command
        assert '{"enable_thinking":false}' in command
        assert command[command.index("--gdn-prefill-backend") + 1] == "triton"
    assert "--language-model-only" in manager
    assert "--trust-remote-code" not in manager
    assert "--language-model-only" not in worker
    assert "--trust-remote-code" in worker


def test_untuned_qwen_decomposer_matches_sft_two_gpu_topology() -> None:
    experiment = QWEN35_BASE_DECOMPOSER_EXPERIMENT

    assert experiment.name == ("qwen35-4b-base-non-thinking-qwen35-4b-non-thinking")
    assert experiment.num_gpus == 2
    assert experiment.prompt_profile == "student"
    assert experiment.manager_parallel_tool_calls is False
    assert experiment.manager_checkpoint == experiment.worker_checkpoint
    assert experiment.manager_port != experiment.worker_port
    assert (
        experiment.temperature,
        experiment.top_p,
        experiment.top_k,
        experiment.min_p,
        experiment.presence_penalty,
        experiment.repetition_penalty,
    ) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)

    manager, worker = decomposer_vllm_commands(experiment)
    assert manager is not None
    for command, port in (
        (manager, experiment.manager_port),
        (worker, experiment.worker_port),
    ):
        assert str(experiment.worker_checkpoint) in command
        assert command[command.index("--port") + 1] == str(port)
        assert command[command.index("--max-model-len") + 1] == "131072"
        assert command[command.index("--tool-call-parser") + 1] == "qwen3_xml"
        assert command[command.index("--gdn-prefill-backend") + 1] == "triton"
        assert "--reasoning-parser" not in command
        assert "--language-model-only" not in command
        assert "--trust-remote-code" in command
        assert '{"enable_thinking":false}' in command

    plan = _dry_plan(Path.cwd(), experiment, Path("/tmp/output"), ("0", "1"), 3, None)
    assert plan["gpu_assignments"] == {
        "manager_vllm": "0",
        "worker_vllm": "1",
    }
    vllm_services = [service for service in plan["services"] if "vllm serve" in service]
    assert len(vllm_services) == 2


def test_qwen_worker_uses_official_non_thinking_sampling() -> None:
    experiment = DEEPSEEK_QWEN_EXPERIMENT
    assert (
        experiment.temperature,
        experiment.top_p,
        experiment.top_k,
        experiment.min_p,
        experiment.presence_penalty,
        experiment.repetition_penalty,
    ) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)
    assert subagent_environment(experiment) == {
        "GAIA2_SUBAGENT_MODEL": "Qwen/Qwen3.5-4B",
        "GAIA2_SUBAGENT_ENDPOINT": "http://127.0.0.1:8031/v1",
        "GAIA2_SUBAGENT_API_KEY": "EMPTY",
        "GAIA2_SUBAGENT_TEMPERATURE": "0.7",
        "GAIA2_SUBAGENT_TOP_P": "0.8",
        "GAIA2_SUBAGENT_TOP_K": "20",
        "GAIA2_SUBAGENT_MAX_COMPLETION_TOKENS": "4096",
        "GAIA2_SUBAGENT_THINKING": "0",
        "GAIA2_SUBAGENT_MIN_P": "0.0",
        "GAIA2_SUBAGENT_PRESENCE_PENALTY": "1.5",
        "GAIA2_SUBAGENT_REPETITION_PENALTY": "1.0",
    }


@pytest.mark.parametrize(
    ("experiment", "served_name"),
    [
        (
            QWEN35_SFT_EXPERIMENT,
            "decomposer/qwen35-4b-sft-workplace-v1-3765-32k",
        ),
        (
            QWEN35_MIXED_SFT_EXPERIMENT,
            "decomposer/qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k",
        ),
    ],
)
def test_qwen_sft_manager_uses_official_non_thinking_sampling(
    tmp_path,
    experiment,
    served_name: str,
) -> None:
    service_path, _ = _runtime_configs(
        Path(__file__).resolve().parents[2],
        tmp_path,
        experiment,
    )
    manager = json.loads(service_path.read_text(encoding="utf-8"))["manager"]

    assert manager == {
        "model": served_name,
        "base_url": "http://127.0.0.1:8026/v1",
        "api_key": "EMPTY",
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "max_completion_tokens": 4096,
        "use_responses_api": False,
        "parallel_tool_calls": False,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_gemma_worker_sampling_is_unchanged() -> None:
    assert (
        DECOMPOSER_EXPERIMENT.temperature,
        DECOMPOSER_EXPERIMENT.top_p,
        DECOMPOSER_EXPERIMENT.top_k,
    ) == (1.0, 0.95, 64)
    environment = subagent_environment(DECOMPOSER_EXPERIMENT)
    assert "GAIA2_SUBAGENT_MIN_P" not in environment
    assert "GAIA2_SUBAGENT_PRESENCE_PENALTY" not in environment
    assert "GAIA2_SUBAGENT_REPETITION_PENALTY" not in environment


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
        "parallel_tool_calls": False,
    }
    assert "path" not in plugin["model_configuration"]["manager"]
    assert plugin["model_configuration"]["manager"]["parallel_tool_calls"] is False


def test_decomposer_dry_plan_records_sequential_manager_tool_calls(tmp_path) -> None:
    plan = _dry_plan(
        Path(__file__).resolve().parents[2],
        DEEPSEEK_QWEN_EXPERIMENT,
        tmp_path,
        ("0",),
        1,
        1,
    )

    assert plan["manager_parallel_tool_calls"] is False


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


@pytest.mark.parametrize(
    "experiment",
    [QWEN35_SFT_EXPERIMENT, QWEN35_MIXED_SFT_EXPERIMENT],
)
def test_qwen_sft_preparation_hashes_manager_and_worker(
    monkeypatch,
    experiment,
) -> None:
    calls = []

    def fake_validate_checkpoint(path, *, full_hashes):
        calls.append((path, full_hashes))
        return {"path": str(path)}

    monkeypatch.setattr(prepare, "validate_checkpoint", fake_validate_checkpoint)

    models = prepare.experiment_models(
        experiment,
        full_hashes=False,
    )

    assert calls == [
        (experiment.worker_checkpoint, False),
        (experiment.manager_checkpoint, False),
    ]
    assert models == {
        "worker": {"path": str(experiment.worker_checkpoint)},
        "manager": {"path": str(experiment.manager_checkpoint)},
    }


def test_simple_qwen_preparation_hashes_policy_only(monkeypatch) -> None:
    calls = []

    def fake_validate_checkpoint(path, *, full_hashes):
        calls.append((path, full_hashes))
        return {"path": str(path)}

    monkeypatch.setattr(prepare, "validate_checkpoint", fake_validate_checkpoint)

    models = prepare.experiment_models(
        SIMPLE_QWEN_EXPERIMENT,
        full_hashes=False,
    )

    assert calls == [(SIMPLE_QWEN_EXPERIMENT.checkpoint, False)]
    assert models == {"policy": {"path": str(SIMPLE_QWEN_EXPERIMENT.checkpoint)}}


def test_simple_deepseek_preparation_records_remote_policy_without_hashing(
    monkeypatch,
) -> None:
    def unexpected_validate_checkpoint(path, *, full_hashes):
        raise AssertionError(f"Unexpected checkpoint validation: {path}")

    monkeypatch.setattr(prepare, "validate_checkpoint", unexpected_validate_checkpoint)

    assert prepare.experiment_models(
        SIMPLE_DEEPSEEK_EXPERIMENT,
        full_hashes=False,
    ) == {
        "policy": {
            "backend": "openrouter",
            "model": "deepseek/deepseek-v4-flash-0731",
        }
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
        QWEN35_MIXED_SFT_EXPERIMENT,
        tmp_path / "staged",
        num_repeats=3,
        limit=None,
        author="sukhorukov",
        base_image="image",
        priority="high",
        force=False,
        judge_environment={
            "LLM_PROXY_URL": "https://judge.test/v1",
            "LLM_PROXY_MASTER_KEY": "secret",
        },
        proxy_environment={},
        openrouter_key="<not-set>",
    )

    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[2]
    assert payload["priority_class"] == "high"
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


def test_remote_simple_agent_is_local_only_for_mlspace_launcher(tmp_path) -> None:
    with pytest.raises(ValueError, match="local-only remote experiment"):
        build_payload(
            SIMPLE_DEEPSEEK_EXPERIMENT,
            tmp_path / "staged",
            num_repeats=3,
            limit=None,
            author="sukhorukov",
            base_image="image",
            priority="high",
            force=False,
            judge_environment={
                "LLM_PROXY_URL": "https://judge.test/v1",
                "LLM_PROXY_MASTER_KEY": "judge-secret",
            },
            proxy_environment={"HTTPS_PROXY": "http://proxy.test"},
            openrouter_key="openrouter-secret",
        )
