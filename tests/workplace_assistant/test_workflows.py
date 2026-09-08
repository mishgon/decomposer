from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gyms.workplace_assistant import experiments, run_eval
from gyms.workplace_assistant import migrate_call_limit_artifacts as migration_module
from gyms.workplace_assistant import prepare as prepare_module
from gyms.workplace_assistant import run as run_module
from gyms.workplace_assistant.experiments import (
    DECOMPOSER_EXPERIMENTS,
    INSTANCE_TYPES_BY_NUM_GPUS,
    SIMPLE_EXPERIMENTS,
    WORKPLACE_E4B_SFT_FINAL,
    WORKPLACE_E4B_SFT_MODEL_ID,
    WORKPLACE_E4B_SFT_VLLM,
    WORKPLACE_QWEN35_4B_FILTERED_SFT_FINAL,
    WORKPLACE_QWEN35_4B_FILTERED_SFT_MODEL_ID,
    WORKPLACE_QWEN35_4B_FINAL_MIXED_SFT_FINAL,
    WORKPLACE_QWEN35_4B_FINAL_MIXED_SFT_MODEL_ID,
    WORKPLACE_QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_FINAL,
    WORKPLACE_QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_MODEL_ID,
    WORKPLACE_QWEN35_4B_GAIA2_SFT_FINAL,
    WORKPLACE_QWEN35_4B_GAIA2_SFT_MODEL_ID,
    WORKPLACE_QWEN35_4B_MIXED_SFT_FINAL,
    WORKPLACE_QWEN35_4B_MIXED_SFT_MODEL_ID,
    WORKPLACE_QWEN35_4B_SFT_FINAL,
    WORKPLACE_QWEN35_4B_SFT_MODEL_ID,
    WORKPLACE_QWEN35_4B_TOOLATHLON_ONLY_SFT_FINAL,
    WORKPLACE_QWEN35_4B_TOOLATHLON_ONLY_SFT_MODEL_ID,
    WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID,
    DecomposerExperiment,
    SimpleExperiment,
    collect_experiments,
    completion_marker,
    decomposer_prompt_profile,
    get_experiment,
    job_description,
    models_for_experiment,
    output_dir,
    run_name,
)
from gyms.qwen_sampling import qwen35_general_sampling


def test_registry_is_global_and_unique() -> None:
    assert len(DECOMPOSER_EXPERIMENTS) == 35
    assert len(SIMPLE_EXPERIMENTS) == 32
    assert len(experiments.EXPERIMENTS) == 67
    assert experiments.BASE_IMAGE.endswith("py3.12-torch2.7.0:0.0.42")
    assert {experiment.kind for experiment in experiments.ALL_EXPERIMENTS} == {
        "decomposer",
        "simple",
    }


def test_port_layout_offsets_all_coordinated_endpoints() -> None:
    ports = run_module.WorkplacePortLayout(12000)
    assert ports.simple_vllm == 20000
    assert ports.langgraph == 14024
    assert ports.gym_head == 23000
    assert (ports.gym_component_low, ports.gym_component_high) == (23001, 23999)
    assert run_module.nonnegative_int("0") == 0
    assert run_module.nonnegative_int("53536") == 53536
    with pytest.raises(run_module.argparse.ArgumentTypeError, match="TCP port range"):
        run_module.nonnegative_int("53537")
    with pytest.raises(run_module.argparse.ArgumentTypeError, match="non-negative"):
        run_module.nonnegative_int("-1")


def test_port_offset_threads_through_simple_commands() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment(
        "gemma4-e4b-thinking-simple-text-defaults"
    )
    assert isinstance(experiment, SimpleExperiment)
    ports = run_module.WorkplacePortLayout(12000)
    vllm = run_module.simple_vllm_command(experiment, ports)
    assert vllm[vllm.index("--port") + 1] == "20000"
    start = run_module.gym_start_command(
        repo_root,
        experiment,
        purpose="evaluation",
        gym_bin=Path("/gym"),
        component_root=Path("/components"),
        logs=Path("/logs"),
        ports=ports,
    )
    assert start[start.index("--model-url") + 1] == "http://127.0.0.1:20000/v1"
    assert "+head_server.port=23000" in start
    assert "+port_range_low=23001" in start
    assert "+port_range_high=23999" in start
    evaluation = run_module.gym_eval_command(
        experiment,
        gym_bin=Path("/gym"),
        split="validation",
        output=Path("/output.jsonl"),
        num_repeats=1,
        limit=None,
        resume=False,
        ports=ports,
    )
    assert "+head_server.port=23000" in evaluation


def test_port_offset_materializes_decomposer_runtime_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment(
        "gemma4-26b-a4b-thinking-gemma4-e4b-thinking-text-defaults"
    )
    assert isinstance(experiment, DecomposerExperiment)
    ports = run_module.WorkplacePortLayout(24000)
    config_path, metadata = run_module.runtime_decomposer_config(
        repo_root,
        experiment,
        tmp_path,
        ports,
        materialize=True,
    )
    config = yaml.safe_load(config_path.read_text())
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    agent = config["decomposer"]["responses_api_agents"]["decomposer_agent"]
    assert policy["base_url"] == "http://127.0.0.1:32023/v1"
    assert {item["url"] for item in agent["subagent_types"]} == {
        "http://127.0.0.1:26024"
    }
    assert metadata["sha256"] == run_module.sha256_file(config_path)
    assert len(metadata["rewrites"]) == 2
    tracked = repo_root / "gyms" / "workplace_assistant" / "configs"
    source = tracked / experiment.gym_config_filename
    assert yaml.safe_load(source.read_text())["policy_model"][
        "responses_api_models"
    ]["vllm_model"]["base_url"] == "http://127.0.0.1:8023/v1"


def test_port_offset_rewrites_qwen_proxy_and_legacy_configs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ports = run_module.WorkplacePortLayout(12000)
    qwen = get_experiment(
        "qwen36-35b-a3b-non-thinking-teacher-"
        "qwen35-4b-non-thinking-text-defaults"
    )
    assert isinstance(qwen, DecomposerExperiment)
    path, _ = run_module.runtime_decomposer_config(
        repo_root, qwen, tmp_path / "qwen", ports, materialize=True
    )
    config = yaml.safe_load(path.read_text())
    proxy = config["policy_model"]["responses_api_models"]["openai_model"]
    assert proxy["openai_base_url"] == "http://127.0.0.1:20142/v1"
    command = run_module.remote_manager_proxy_command(qwen, ports)
    assert command[command.index("--port") + 1] == "20142"

    for experiment in DECOMPOSER_EXPERIMENTS:
        runtime_path, metadata = run_module.runtime_decomposer_config(
            repo_root,
            experiment,
            tmp_path / experiment.name,
            ports,
            materialize=False,
        )
        assert runtime_path.name == "workplace_assistant.runtime.yaml"
        assert metadata["rewrites"]


def test_nonzero_offset_uses_repository_graph_and_model_url_mapping() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("deepseek-v4-flash-0731-gemma4-all")
    assert isinstance(experiment, DecomposerExperiment)
    ports = run_module.WorkplacePortLayout(12000)
    command, directory = run_module.langgraph_command(repo_root, experiment, ports)
    assert directory == repo_root / "gyms" / "workplace_assistant" / "subagents"
    assert command[command.index("--port") + 1] == "14024"
    environment = run_module._base_environment(
        repo_root, experiment, "offset-test", ports
    )
    urls = json.loads(environment[run_module.SUBAGENT_MODEL_URLS_ENV])
    assert urls == {
        "google/gemma-4-E2B-it": "http://127.0.0.1:20020/v1",
        "google/gemma-4-E4B-it": "http://127.0.0.1:20021/v1",
        "google/gemma-4-12B-it": "http://127.0.0.1:20022/v1",
        "google/gemma-4-26B-A4B-it": "http://127.0.0.1:20023/v1",
    }


def test_qwen36_teacher_uses_internal_proxy_and_concurrency_override() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("qwen36-35b-a3b-teacher-qwen35-4b-non-thinking")
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.manager_backend == "llm_proxy"
    assert experiment.manager_model_id == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert experiment.manager_reasoning_mode == "service_default"
    assert experiment.remote_manager_extra_body == {}
    assert experiment.requires_llm_proxy
    assert experiment.requires_remote_manager
    assert not experiment.requires_openrouter
    assert experiment.num_gpus == 1
    assert experiment.concurrency == 16
    assert [model.model_id for model in models_for_experiment(experiment)] == [
        "Qwen/Qwen3.5-4B"
    ]
    assert prepare_module.components_for_experiments((experiment,)) == (
        "resources_servers/workplace_assistant",
        "responses_api_agents/decomposer_agent",
        "responses_api_models/openai_model",
    )

    proxy = run_module.remote_manager_proxy_command(experiment)
    assert "gyms.remote_model_proxy" in proxy
    assert proxy[proxy.index("--upstream-url-env") + 1] == "LLM_PROXY_URL"
    assert proxy[proxy.index("--api-key-env") + 1] == "LLM_PROXY_MASTER_KEY"
    assert "--no-verify-tls" in proxy

    plan = run_module._dry_plan(
        repo_root,
        experiment,
        "trace-generation",
        "validation",
        3,
        None,
        Path("/tmp/qwen36-workplace"),
        ("0",),
        16,
    )
    assert plan["decomposer_system_prompt_profile"] == "teacher"
    assert "gyms.remote_model_proxy" in plan["services"][0]
    assert "--concurrency 16" in plan["gym_eval"]

    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="trace-generation",
        split="validation",
        num_repeats=3,
        limit=None,
        author="alice",
        base_image=experiments.BASE_IMAGE,
        priority="high",
        force=False,
        proxy_env={},
        openrouter_key="<not-set>",
        llm_proxy_environment={
            "LLM_PROXY_URL": "https://internal/v1",
            "LLM_PROXY_MASTER_KEY": "secret",
        },
        concurrency=16,
    )
    assert payload["env_variables"]["LLM_PROXY_URL"] == "https://internal/v1"
    assert payload["env_variables"]["LLM_PROXY_MASTER_KEY"] == "secret"
    assert "--concurrency 16" in payload["script"]
    assert payload["priority_class"] == "high"


def test_qwen36_text_defaults_force_proxy_sampling_and_teacher_prompt(tmp_path) -> None:
    experiment = get_experiment(
        "qwen36-35b-a3b-non-thinking-teacher-"
        "qwen35-4b-non-thinking-text-defaults"
    )
    assert isinstance(experiment, DecomposerExperiment)
    expected_proxy_body = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "include_reasoning": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert experiment.manager_reasoning_mode == "non_thinking"
    assert experiment.evaluation_prompt_profile == "teacher"
    assert experiment.max_model_len == 131072
    assert experiment.max_output_tokens is None
    assert experiment.manager_max_model_calls == 100
    assert experiment.subagent_max_model_calls == 100
    assert experiment.remote_manager_extra_body == expected_proxy_body

    proxy = run_module.remote_manager_proxy_command(experiment)
    assert (
        json.loads(proxy[proxy.index("--extra-body-json") + 1])
        == expected_proxy_body
    )
    assert proxy[proxy.index("--upstream-url-env") + 1] == "LLM_PROXY_URL"
    assert proxy[proxy.index("--api-key-env") + 1] == "LLM_PROXY_MASTER_KEY"

    plan = run_module._dry_plan(
        Path(__file__).resolve().parents[2],
        experiment,
        "evaluation",
        "validation",
        3,
        None,
        tmp_path,
        ("0",),
    )
    assert plan["decomposer_system_prompt_profile"] == "teacher"
    assert plan["runtime_configuration"]["max_output_tokens"] is None
    assert "max_output_tokens" not in plan["gym_start"]
    assert [model.model_id for model in models_for_experiment(experiment)] == [
        "Qwen/Qwen3.5-4B"
    ]
    worker = run_module.decomposer_vllm_command(
        models_for_experiment(experiment)[0], experiment
    )
    assert "--language-model-only" in worker
    assert '{"enable_thinking":false}' in worker


def test_qwen36_thinking_text_defaults_preserve_reasoning(tmp_path) -> None:
    experiment = get_experiment(
        "qwen36-35b-a3b-thinking-teacher-"
        "qwen35-4b-non-thinking-text-defaults"
    )
    assert isinstance(experiment, DecomposerExperiment)
    expected_proxy_body = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "include_reasoning": True,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    }
    assert experiment.manager_backend == "llm_proxy"
    assert experiment.manager_model_id == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert experiment.manager_reasoning_mode == "thinking"
    assert experiment.evaluation_prompt_profile == "teacher"
    assert experiment.max_model_len == 131072
    assert experiment.max_output_tokens is None
    assert experiment.manager_max_model_calls == 100
    assert experiment.subagent_max_model_calls == 100
    assert experiment.subagent_recursion_limit == 1000
    assert experiment.num_gpus == 1
    assert experiment.concurrency == 16
    assert experiment.remote_manager_extra_body == expected_proxy_body

    ports = run_module.WorkplacePortLayout(12000)
    proxy = run_module.remote_manager_proxy_command(experiment, ports)
    assert proxy[proxy.index("--port") + 1] == "20142"
    assert proxy[proxy.index("--upstream-url-env") + 1] == "LLM_PROXY_URL"
    assert proxy[proxy.index("--api-key-env") + 1] == "LLM_PROXY_MASTER_KEY"
    assert proxy[proxy.index("--response-tool-parser") + 1] == "qwen3_xml"
    assert "--no-verify-tls" in proxy
    assert (
        json.loads(proxy[proxy.index("--extra-body-json") + 1])
        == expected_proxy_body
    )

    repo_root = Path(__file__).resolve().parents[2]
    runtime_config, _ = run_module.runtime_decomposer_config(
        repo_root, experiment, tmp_path / "runtime", ports, materialize=True
    )
    config = yaml.safe_load(runtime_config.read_text())
    assert config["responses_create_params"] == {
        "temperature": 1.0,
        "top_p": 0.95,
    }
    assert "presence_penalty" not in config["responses_create_params"]
    policy = config["policy_model"]["responses_api_models"]["openai_model"]
    assert policy["openai_base_url"] == "http://127.0.0.1:20142/v1"
    assert policy["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "include_reasoning": True,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    }
    assert [model.model_id for model in models_for_experiment(experiment)] == [
        "Qwen/Qwen3.5-4B"
    ]
    worker = run_module.decomposer_vllm_command(
        models_for_experiment(experiment)[0], experiment, ports
    )
    assert "--language-model-only" in worker
    assert '{"enable_thinking":false}' in worker

    plan = run_module._dry_plan(
        repo_root,
        experiment,
        "evaluation",
        "validation",
        3,
        None,
        tmp_path / "dry",
        ("0",),
        16,
        ports=ports,
    )
    assert plan["decomposer_system_prompt_profile"] == "teacher"
    assert plan["gpu_assignments"] == {"subagent_vllm_20025": "0"}
    assert "gyms.remote_model_proxy" in plan["services"][0]
    assert "--concurrency 16" in plan["gym_eval"]


def test_gemma_text_defaults_use_pinned_thinking_manager_and_worker(tmp_path) -> None:
    experiment = get_experiment(
        "gemma4-26b-a4b-thinking-gemma4-e4b-thinking-text-defaults"
    )
    assert isinstance(experiment, DecomposerExperiment)
    models = models_for_experiment(experiment)
    assert [model.model_id for model in models] == [
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-E4B-it",
    ]
    assert models[0].snapshot.name == "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
    assert all(model.thinking for model in models)
    assert experiment.max_model_len == 131072
    assert experiment.max_output_tokens is None
    for model in models:
        command = run_module.decomposer_vllm_command(model, experiment)
        assert command[command.index("--max-model-len") + 1] == "131072"
        assert "--language-model-only" in command
        assert '{"enable_thinking":true,"preserve_thinking":true}' in command

    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "gyms"
            / "workplace_assistant"
            / "configs"
            / experiment.gym_config_filename
        ).read_text()
    )
    assert config["responses_create_params"] == {
        "temperature": 1.0,
        "top_p": 0.95,
    }
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert policy["chat_template_kwargs"]["enable_thinking"] is True
    assert policy["extra_body"]["top_k"] == 64
    command, directory = run_module.langgraph_command(
        Path(__file__).resolve().parents[2], experiment
    )
    assert directory == (
        Path(__file__).resolve().parents[2]
        / "gyms"
        / "workplace_assistant"
        / "subagents"
    )
    assert "langgraph.json" in " ".join(command)


def test_requested_gemma_simple_profiles_use_matched_128k_defaults() -> None:
    names = [
        f"gemma4-{size}-it-{mode}"
        for size in ("e2b", "e4b", "31b", "26b-a4b")
        for mode in ("non-thinking", "thinking")
    ]
    profiles = [get_experiment(name) for name in names]
    assert all(isinstance(profile, SimpleExperiment) for profile in profiles)
    assert experiments.GEMMA4_E2B_BASE.name == (
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    )
    assert experiments.GEMMA4_31B_BASE.name == (
        "842da3794eaa0b77d5f08bae87a17459d91ff475"
    )
    for profile in profiles:
        assert isinstance(profile, SimpleExperiment)
        assert profile.num_gpus == 1
        assert profile.max_model_len == 131072
        assert profile.max_output_tokens is None
        assert profile.max_steps == 100
        assert (profile.temperature, profile.top_p, profile.top_k) == (
            1.0,
            0.95,
            64,
        )
        command = run_module.simple_vllm_command(profile)
        template_kwargs = json.loads(
            command[command.index("--default-chat-template-kwargs") + 1]
        )
        assert template_kwargs == {
            "enable_thinking": profile.thinking,
            **({"preserve_thinking": True} if profile.thinking else {}),
        }


def test_shared_gemma26_teacher_manager_and_worker_use_one_server(tmp_path) -> None:
    experiment = get_experiment(
        "gemma4-26b-a4b-thinking-teacher-"
        "gemma4-26b-a4b-non-thinking-text-defaults"
    )
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.manager_backend == "local_vllm"
    assert experiment.evaluation_prompt_profile == "teacher"
    assert experiment.num_gpus == 1
    assert experiment.max_model_len == 131072
    models = models_for_experiment(experiment)
    assert len(models) == 1
    assert models[0].model_id == "google/gemma-4-26B-A4B-it"
    assert models[0].thinking is True

    server = run_module.decomposer_vllm_command(models[0], experiment)
    assert json.loads(
        server[server.index("--default-chat-template-kwargs") + 1]
    ) == {"enable_thinking": True, "preserve_thinking": True}

    config_path, _ = run_module.runtime_decomposer_config(
        Path.cwd(),
        experiment,
        tmp_path,
        run_module.WorkplacePortLayout(12000),
        materialize=True,
    )
    config = yaml.safe_load(config_path.read_text())
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert policy["base_url"] == "http://127.0.0.1:20023/v1"
    assert policy["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    subagent = config["decomposer"]["responses_api_agents"]["decomposer_agent"]
    assert subagent["decomposer_system_prompt_profile"] == "teacher"
    assert subagent["subagent_types"][0]["assistant_id"] == (
        "gemma_4_26b_a4b_non_thinking"
    )


def test_deepseek_gemma26_teacher_profile_is_matched_and_single_gpu() -> None:
    experiment = get_experiment(
        "deepseek-v4-flash-0731-teacher-gemma4-26b-a4b-non-thinking"
    )
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.requires_openrouter is True
    assert experiment.evaluation_prompt_profile == "teacher"
    assert experiment.num_gpus == 1
    assert experiment.max_model_len == 131072
    models = models_for_experiment(experiment)
    assert [model.model_id for model in models] == [
        "google/gemma-4-26B-A4B-it"
    ]
    assert models[0].thinking is True
    config = yaml.safe_load(
        (
            Path.cwd()
            / "gyms"
            / "workplace_assistant"
            / "configs"
            / experiment.gym_config_filename
        ).read_text()
    )
    agent = config["decomposer"]["responses_api_agents"]["decomposer_agent"]
    assert agent["decomposer_system_prompt_profile"] == "teacher"
    assert agent["subagent_types"][0]["assistant_id"] == (
        "gemma_4_26b_a4b_non_thinking"
    )


def test_reasoning_policy_identity_distinguishes_proxy_capture_only() -> None:
    local = get_experiment("gemma4-e2b-it-thinking")
    proxy = get_experiment(
        "qwen36-35b-a3b-thinking-teacher-"
        "qwen35-4b-non-thinking-text-defaults"
    )
    assert run_module.runtime_configuration(local)[
        "structured_reasoning_policy"
    ] == "capture_replay_v2_template_preserved"
    assert run_module.runtime_configuration(proxy)[
        "structured_reasoning_policy"
    ] == "capture_only_upstream_no_replay_v1"


def test_every_local_gemma_thinking_path_enables_template_replay() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for experiment in experiments.ALL_EXPERIMENTS:
        if isinstance(experiment, SimpleExperiment):
            if experiment.thinking:
                command = run_module.simple_vllm_command(experiment)
                kwargs = json.loads(
                    command[command.index("--default-chat-template-kwargs") + 1]
                )
                assert kwargs["preserve_thinking"] is True
            continue

        for model in models_for_experiment(experiment):
            if model.thinking:
                command = run_module.decomposer_vllm_command(model, experiment)
                kwargs = json.loads(
                    command[command.index("--default-chat-template-kwargs") + 1]
                )
                assert kwargs["preserve_thinking"] is True

        config = yaml.safe_load(
            (
                repo_root
                / "gyms"
                / "workplace_assistant"
                / "configs"
                / experiment.gym_config_filename
            ).read_text()
        )
        policy_models = config.get("policy_model", {}).get(
            "responses_api_models", {}
        )
        for policy in policy_models.values():
            kwargs = policy.get("chat_template_kwargs", {})
            if kwargs.get("enable_thinking") is True:
                assert kwargs["preserve_thinking"] is True


def test_text_default_simple_profiles_use_matched_workplace_limits() -> None:
    gemma = get_experiment("gemma4-e4b-thinking-simple-text-defaults")
    qwen = get_experiment("qwen35-4b-non-thinking-simple-general-text-defaults")
    assert isinstance(gemma, SimpleExperiment)
    assert isinstance(qwen, SimpleExperiment)
    for experiment in (gemma, qwen):
        assert experiment.max_model_len == 131072
        assert experiment.max_output_tokens is None
        assert experiment.max_steps == 100
        assert "--language-model-only" in run_module.simple_vllm_command(experiment)
    assert (gemma.temperature, gemma.top_p, gemma.top_k) == (1.0, 0.95, 64)
    assert (
        qwen.temperature,
        qwen.top_p,
        qwen.top_k,
        qwen.min_p,
        qwen.presence_penalty,
        qwen.repetition_penalty,
    ) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)


def test_all_simple_profiles_use_provider_output_length_by_default() -> None:
    for experiment in SIMPLE_EXPERIMENTS:
        assert experiment.max_output_tokens is None
        command = run_module.gym_eval_command(
            experiment,
            gym_bin=Path("/gym"),
            split="validation",
            output=Path("/rollouts.jsonl"),
            num_repeats=1,
            limit=None,
            resume=False,
        )
        assert "--max-output-tokens" not in command

    limited = replace(SIMPLE_EXPERIMENTS[0], max_output_tokens=4096)
    command = run_module.gym_eval_command(
        limited,
        gym_bin=Path("/gym"),
        split="validation",
        output=Path("/rollouts.jsonl"),
        num_repeats=1,
        limit=None,
        resume=False,
    )
    assert command[command.index("--max-output-tokens") + 1] == "4096"
    with pytest.raises(ValueError, match="max_output_tokens"):
        replace(SIMPLE_EXPERIMENTS[0], max_output_tokens=0)


def test_workplace_prompt_override_is_propagated_and_output_isolated() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment(
        "qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-"
        "non-thinking-qwen35-4b-non-thinking"
    )
    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="evaluation",
        split="validation",
        num_repeats=3,
        limit=None,
        author="alice",
        base_image=experiments.BASE_IMAGE,
        priority="high",
        force=False,
        proxy_env={},
        openrouter_key="unused",
        prompt_profile="teacher",
    )
    assert "--prompt-profile teacher" in payload["script"]
    assert "prompt-teacher" in payload["job_desc"]
    path = output_dir(
        experiment,
        "validation",
        3,
        purpose="evaluation",
        prompt_profile="teacher",
    )
    assert path.name.endswith("-prompt-teacher")


def test_all_qwen_simple_profiles_use_official_mode_specific_sampling() -> None:
    qwen_experiments = [
        experiment
        for experiment in SIMPLE_EXPERIMENTS
        if experiment.name.startswith(("q35-", "qwen35-"))
    ]
    assert qwen_experiments

    for experiment in qwen_experiments:
        expected = qwen35_general_sampling(thinking=experiment.thinking)
        assert (
            experiment.temperature,
            experiment.top_p,
            experiment.top_k,
            experiment.min_p,
            experiment.presence_penalty,
            experiment.repetition_penalty,
        ) == (
            expected.temperature,
            expected.top_p,
            expected.top_k,
            expected.min_p,
            expected.presence_penalty,
            expected.repetition_penalty,
        )
        assert experiment.extra_body == expected.extra_body | {
            "presence_penalty": expected.presence_penalty
        }
        assert run_module.hydra_flow_mapping(experiment.extra_body) == (
            "{top_k:20,min_p:0.0,presence_penalty:1.5,repetition_penalty:1.0}"
        )
        command = run_module.gym_eval_command(
            experiment,
            gym_bin=Path("/gym"),
            split="validation",
            output=Path("/tmp/qwen-sampling-test"),
            num_repeats=1,
            limit=1,
            resume=False,
        )
        assert command[command.index("--temperature") + 1] == str(expected.temperature)
        assert command[command.index("--top-p") + 1] == str(expected.top_p)
        assert "+rollout_failure_policy=score_zero" in command


def test_gemma_simple_profiles_keep_existing_sampling() -> None:
    for experiment in SIMPLE_EXPERIMENTS:
        if not experiment.name.startswith("gemma4-"):
            continue
        assert (experiment.temperature, experiment.top_p, experiment.top_k) == (
            1.0,
            0.95,
            64,
        )
        assert experiment.presence_penalty == 0.0


def test_qwen35_9b_simple_profile_uses_cached_non_thinking_checkpoint() -> None:
    experiment = get_experiment("qwen35-9b-base-non-thinking")
    assert isinstance(experiment, SimpleExperiment)
    assert experiment.checkpoint == experiments.QWEN35_9B_BASE
    assert experiment.checkpoint.is_dir()
    assert experiment.thinking is False
    assert experiment.num_gpus == 1
    command = run_module.simple_vllm_command(experiment)
    assert str(experiments.QWEN35_9B_BASE) in command
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":false}'
    )


def test_simple_profiles_default_to_calls100_and_use_isolated_output() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("qwen35-4b-base-non-thinking")
    assert isinstance(experiment, SimpleExperiment)
    assert all(item.max_steps == 100 for item in SIMPLE_EXPERIMENTS)
    assert experiment.max_steps == 100
    assert experiment.checkpoint == experiments.QWEN35_4B_BASE
    assert experiment.thinking is False
    with pytest.raises(ValueError, match="Unknown Workplace Assistant experiment"):
        get_experiment("qwen35-4b-base-non-thinking-maxsteps100")

    start = run_module.gym_start_command(
        repo_root,
        experiment,
        purpose="evaluation",
        gym_bin=Path("/gym"),
        component_root=Path("/components"),
        logs=Path("/logs"),
    )
    assert any(argument.endswith("simple_agent.max_steps=100") for argument in start)

    path = output_dir(experiment, "validation", 3, purpose="evaluation")
    assert path.name == "qwen35-4b-base-non-thinking-calls100-n3"

    plan = run_module._dry_plan(
        repo_root,
        experiment,
        "evaluation",
        "validation",
        3,
        None,
        path,
        ("0",),
    )
    assert plan["simple_agent_max_steps"] == 100
    assert "simple_agent.max_steps=100" in plan["gym_start"]

    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="evaluation",
        split="validation",
        num_repeats=3,
        limit=None,
        author="sukhorukov",
        base_image=experiments.BASE_IMAGE,
        priority="high",
        force=False,
        proxy_env={},
        openrouter_key="",
    )
    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert payload["priority_class"] == "high"
    assert payload["job_desc"] == (
        "workplace-assistant-validation simple-agent "
        "qwen35-4b-base-non-thinking-calls100-n3 #sukhorukov"
    )


def test_simple_output_identity_includes_max_steps(tmp_path: Path) -> None:
    experiment = get_experiment("qwen35-4b-base-non-thinking")
    (tmp_path / "run_status.json").write_text(
        json.dumps(
            {
                "experiment": experiment.name,
                "kind": "simple",
                "purpose": "evaluation",
                "split": "validation",
                "num_repeats": 3,
                "limit": None,
                "simple_agent_max_steps": 6,
            }
        )
    )
    with pytest.raises(RuntimeError, match="simple_agent_max_steps=6"):
        run_module.validate_existing_attempt_identity(
            tmp_path,
            experiment,
            purpose="evaluation",
            split="validation",
            num_repeats=3,
            limit=None,
            force=False,
        )


def test_port_offset_isolates_default_output_and_resume_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = get_experiment("qwen35-4b-base-non-thinking")
    assert isinstance(experiment, SimpleExperiment)
    monkeypatch.setattr(experiments, "RESULTS_ROOT", tmp_path / "results")
    args = run_module.build_parser().parse_args(
        [
            "--experiment",
            experiment.name,
            "--purpose",
            "evaluation",
            "--split",
            "validation",
            "--num-repeats",
            "3",
            "--port-offset",
            "12000",
        ]
    )
    directory = run_module.selected_output_dir(experiment, args)
    assert directory.name.endswith("-port-offset-12000")
    identity = run_module.local_run_name(
        experiment,
        3,
        prompt_profile=None,
        ports=run_module.WorkplacePortLayout(12000),
    )
    monkeypatch.setattr(run_module, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    environment = run_module._base_environment(
        Path(__file__).resolve().parents[2],
        experiment,
        identity,
        run_module.WorkplacePortLayout(12000),
    )
    assert Path(environment["VLLM_CACHE_ROOT"]).parts[-2:] == (identity, "vllm")

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "run_status.json").write_text(
        json.dumps(
            {
                "experiment": experiment.name,
                "kind": "simple",
                "purpose": "evaluation",
                "split": "validation",
                "num_repeats": 3,
                "limit": None,
                "simple_agent_max_steps": experiment.max_steps,
                "runtime_configuration": run_module.runtime_configuration(
                    experiment
                ),
            }
        )
    )
    run_module.validate_existing_attempt_identity(
        legacy,
        experiment,
        purpose="evaluation",
        split="validation",
        num_repeats=3,
        limit=None,
        force=False,
    )
    with pytest.raises(RuntimeError, match="port_offset=0"):
        run_module.validate_existing_attempt_identity(
            legacy,
            experiment,
            purpose="evaluation",
            split="validation",
            num_repeats=3,
            limit=None,
            force=False,
            ports=run_module.WorkplacePortLayout(12000),
        )


def test_port_offset_dry_plans_are_disjoint(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cases = (
        (
            get_experiment("qwen35-4b-non-thinking-simple-general-text-defaults"),
            run_module.WorkplacePortLayout(0),
            ("0",),
        ),
        (
            get_experiment("gemma4-e4b-thinking-simple-text-defaults"),
            run_module.WorkplacePortLayout(12000),
            ("1",),
        ),
        (
            get_experiment(
                "gemma4-26b-a4b-thinking-gemma4-e4b-thinking-text-defaults"
            ),
            run_module.WorkplacePortLayout(24000),
            ("2", "3"),
        ),
    )
    occupied: list[set[int]] = []
    for index, (experiment, ports, devices) in enumerate(cases):
        plan = run_module._dry_plan(
            repo_root,
            experiment,
            "evaluation",
            "validation",
            1,
            1,
            tmp_path / str(index),
            devices,
            ports=ports,
        )
        layout = plan["port_layout"]
        current = {
            layout["simple_agent_vllm"],
            layout["langgraph"],
            layout["gym_head"],
            *layout["model_servers"].values(),
        }
        if layout["manager_proxy"] is not None:
            current.add(layout["manager_proxy"])
        current.update(
            range(
                layout["gym_component_range"][0],
                layout["gym_component_range"][1] + 1,
            )
        )
        assert all(current.isdisjoint(previous) for previous in occupied)
        occupied.append(current)
        if plan["runtime_gym_config"] is not None and ports.offset:
            assert not Path(plan["runtime_gym_config"]["path"]).exists()


def test_simple_execute_does_not_read_decomposer_proxy_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PreflightComplete(RuntimeError):
        pass

    experiment = get_experiment("qwen35-4b-base-non-thinking")
    output = tmp_path / "simple-run"
    args = run_module.build_parser().parse_args(
        [
            "--experiment",
            experiment.name,
            "--purpose",
            "evaluation",
            "--split",
            "validation",
            "--num-repeats",
            "3",
            "--output-dir",
            str(output),
        ]
    )
    monkeypatch.setattr(run_module, "validate_preparation", lambda *_: {})

    def stop_after_preflight(*_: object) -> dict[str, str]:
        raise PreflightComplete

    monkeypatch.setattr(run_module, "_base_environment", stop_after_preflight)
    with pytest.raises(PreflightComplete):
        run_module.execute(Path(__file__).resolve().parents[2], args)

    status = json.loads((output / "run_status.json").read_text())
    assert status["experiment"] == experiment.name
    assert status["simple_agent_max_steps"] == 100


def test_deepseek_simple_profile_is_remote_and_does_not_start_vllm() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("deepseek-v4-flash-0731")
    assert isinstance(experiment, SimpleExperiment)
    assert experiment.requires_openrouter
    assert experiment.checkpoint is None
    assert experiment.model_id == "deepseek/deepseek-v4-flash-0731"
    assert experiment.reasoning_effort == "high"
    assert experiment.concurrency == 8
    assert prepare_module.components_for_experiments((experiment,)) == (
        "resources_servers/workplace_assistant",
        "responses_api_agents/simple_agent",
        "responses_api_models/openai_model",
    )
    with pytest.raises(ValueError, match="does not use a local vLLM server"):
        run_module.simple_vllm_command(experiment)

    start = run_module.gym_start_command(
        repo_root,
        experiment,
        purpose="evaluation",
        gym_bin=Path("/gym"),
        component_root=Path("/components"),
        logs=Path("/logs"),
    )
    assert start[start.index("--model-type") + 1] == "openai_model"
    assert start[start.index("--model") + 1] == experiment.model_id
    assert start[start.index("--model-url") + 1] == experiment.base_url
    assert "--model-api-key" not in start
    assert "++policy_api_key=${oc.env:OPENROUTER_API_KEY_DECOMPOSER}" in start
    assert any(argument.endswith('{reasoning:{effort:"high"}}') for argument in start)

    plan = run_module._dry_plan(
        repo_root,
        experiment,
        "evaluation",
        "validation",
        3,
        None,
        Path("/tmp/workplace-deepseek-simple"),
        ("0",),
    )
    assert plan["services"] == []
    assert plan["gpu_assignments"] == {}
    assert "policy_vllm" not in plan["gym_start"]

    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="evaluation",
        split="validation",
        num_repeats=3,
        limit=None,
        author="alice",
        base_image=experiments.BASE_IMAGE,
        priority="high",
        force=False,
        proxy_env={"HTTPS_PROXY": "http://proxy"},
        openrouter_key="secret",
    )
    assert payload["priority_class"] == "high"
    assert payload["env_variables"]["OPENROUTER_API_KEY_DECOMPOSER"] == "secret"
    assert payload["job_desc"] == (
        "workplace-assistant-validation simple-agent "
        "deepseek-v4-flash-0731-calls100-n3 #alice"
    )


def test_decomposer_base_profiles_use_student_prompt() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for experiment in DECOMPOSER_EXPERIMENTS:
        config = yaml.safe_load(
            (
                repo_root
                / "gyms"
                / "workplace_assistant"
                / "configs"
                / experiment.gym_config_filename
            ).read_text()
        )
        agent = config["decomposer"]["responses_api_agents"]["decomposer_agent"]
        assert agent["decomposer_system_prompt_profile"] == (
            experiment.evaluation_prompt_profile
        )


def test_decomposer_call_limits_reach_runtime_and_identity() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("deepseek-v4-flash-0731-gemma4-e4b-thinking")
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.manager_max_model_calls == 100
    assert experiment.subagent_max_model_calls == 100
    assert experiment.subagent_recursion_limit == 1000

    start = run_module.gym_start_command(
        repo_root,
        experiment,
        purpose="evaluation",
        gym_bin=Path("/gym"),
        component_root=Path("/components"),
        logs=Path("/logs"),
    )
    assert any(argument.endswith("manager_max_model_calls=100") for argument in start)
    assert any(argument.endswith("subagent_recursion_limit=1000") for argument in start)

    plan = run_module._dry_plan(
        repo_root,
        experiment,
        "evaluation",
        "validation",
        3,
        None,
        Path("/tmp/workplace-decomposer-calls100"),
        ("0",),
    )
    assert plan["decomposer_manager_max_model_calls"] == 100
    assert plan["decomposer_subagent_max_model_calls"] == 100
    assert plan["decomposer_subagent_recursion_limit"] == 1000

    environment = run_module._base_environment(repo_root, experiment, "test-run")
    assert environment["DECOMPOSER_SUBAGENT_MAX_MODEL_CALLS"] == "100"


def test_selectors_form_a_deduplicated_registry_ordered_union() -> None:
    selected = collect_experiments(
        names=("gemma4-e2b-it-thinking",),
        filters=("gemma4-e2b",),
        default_all=False,
    )
    assert [experiment.name for experiment in selected] == [
        "gemma4-e2b-it-non-thinking",
        "gemma4-e2b-it-thinking",
    ]
    with pytest.raises(ValueError, match="Unknown Workplace Assistant experiments"):
        collect_experiments(names=("missing",), default_all=False)


def test_split_repeat_and_smoke_paths_are_isolated() -> None:
    simple = get_experiment("qwen35-2b-base-non-thinking")
    decomposer = get_experiment("glm-5-2-gemma4-26b-a4b-non-thinking")
    assert run_name(simple) == f"{simple.name}-calls100"
    assert run_name(simple, 5) == f"{simple.name}-calls100-n5"
    assert output_dir(simple, "train", 5, purpose="evaluation").parts[-2:] == (
        "train",
        f"{simple.name}-calls100-n5",
    )
    teacher = output_dir(decomposer, "train", 3, purpose="trace-generation")
    student = output_dir(decomposer, "train", 3, purpose="evaluation")
    decomposer_identity = (
        f"{decomposer.name}-managercalls100-subagentcalls100-n3"
    )
    assert teacher.parts[-2:] == ("train", decomposer_identity)
    assert student.parts[-3:] == (
        "train",
        "evaluation",
        decomposer_identity,
    )
    assert (
        completion_marker(
            decomposer,
            "validation",
            5,
            2,
            purpose="evaluation",
        ).name
        == ".eval_done.json"
    )


def test_run_purpose_controls_prompt_and_preserves_teacher_job_identity() -> None:
    decomposer = get_experiment("glm-5-2-gemma4-26b-a4b-non-thinking")
    simple = get_experiment("gemma4-e2b-it-non-thinking")
    assert decomposer_prompt_profile("trace-generation") == "teacher"
    assert decomposer_prompt_profile("evaluation") == "student"
    assert job_description(
        decomposer,
        "train",
        3,
        purpose="trace-generation",
    ) == (
        "workplace-assistant-train decomposer-agent "
        "glm-5-2-gemma4-26b-a4b-non-thinking-"
        "managercalls100-subagentcalls100-n3"
    )
    assert "train-evaluation" in job_description(
        decomposer,
        "train",
        3,
        purpose="evaluation",
    )
    with pytest.raises(ValueError, match="only supported for Decomposer"):
        output_dir(simple, "train", purpose="trace-generation")


def test_live_allocation_instance_types_are_single_source_of_truth() -> None:
    assert INSTANCE_TYPES_BY_NUM_GPUS == {
        1: "a100plus.1gpu.80vG.12C.244G",
        2: "a100plus.2gpu.80vG.24C.488G",
        3: "a100plus.3gpu.80vG.36C.546G",
    }


def test_decomposer_model_placement_and_single_model_remapping() -> None:
    full = get_experiment("glm-5-2-gemma4-all")
    assert isinstance(full, DecomposerExperiment)
    selected = models_for_experiment(full)
    assert {model.gpu for model in selected} == {0, 1, 2}
    assert sum(
        model.gpu_memory_utilization for model in selected if model.gpu == 0
    ) == pytest.approx(0.9)

    one = get_experiment("glm-5-2-gemma4-26b-a4b-non-thinking")
    assert isinstance(one, DecomposerExperiment)
    selected = models_for_experiment(one)
    assert len(selected) == 1
    assert selected[0].gpu == 0
    assert selected[0].startup_wave == 0


def test_deepseek_e4b_thinking_profile_is_single_type_and_single_gpu() -> None:
    experiment = get_experiment("deepseek-v4-flash-0731-gemma4-e4b-thinking")
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.num_gpus == 1
    selected = models_for_experiment(experiment)
    assert [model.model_id for model in selected] == ["google/gemma-4-E4B-it"]
    assert selected[0].gpu == 0

    repo_root = Path(__file__).resolve().parents[2]
    config = (
        repo_root
        / "gyms"
        / "workplace_assistant"
        / "configs"
        / experiment.gym_config_filename
    ).read_text()
    assert "openai_model: deepseek/deepseek-v4-flash-0731" in config
    assert "effort: high" in config
    assert config.count("subagent_type_id:") == 1
    assert "subagent_type_id: gemma_4_4b_thinking" in config
    assert "assistant_id: gemma_4_4b_thinking" in config

    command = run_module.decomposer_vllm_command(selected[0], experiment)
    assert command[command.index("--served-model-name") + 1] == (
        "google/gemma-4-E4B-it"
    )
    assert command[command.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":true,"preserve_thinking":true}'
    )

    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="trace-generation",
        split="train",
        num_repeats=3,
        limit=None,
        author="alice",
        base_image=experiments.BASE_IMAGE,
        priority=None,
        force=False,
        proxy_env={"HTTPS_PROXY": "http://proxy"},
        openrouter_key="secret",
    )
    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert "--purpose trace-generation" in payload["script"]
    assert "--num-repeats 3" in payload["script"]
    assert run_name(experiment, 3) in payload["job_desc"]
    assert payload["job_desc"] == (
        "workplace-assistant-train decomposer-agent "
        "deepseek-v4-flash-0731-gemma4-e4b-thinking-"
        "managercalls100-subagentcalls100-n3 #alice"
    )


def test_deepseek_qwen_profile_uses_128k_context() -> None:
    experiment = get_experiment("deepseek-v4-flash-0731-qwen35-4b-non-thinking")
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.num_gpus == 1
    assert experiment.max_model_len == 131072

    selected = models_for_experiment(experiment)
    assert [model.model_id for model in selected] == ["Qwen/Qwen3.5-4B"]
    command = run_module.decomposer_vllm_command(selected[0], experiment)
    assert command[command.index("--max-model-len") + 1] == "131072"


@pytest.mark.parametrize(
    ("name", "manager_model_id", "manager_checkpoint"),
    [
        (
            "qwen35-4b-sft-workplace-v1-3765-32k-non-thinking-"
            "qwen35-4b-non-thinking",
            WORKPLACE_QWEN35_4B_SFT_MODEL_ID,
            WORKPLACE_QWEN35_4B_SFT_FINAL,
        ),
        (
            "qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-"
            "qwen35-4b-non-thinking",
            WORKPLACE_QWEN35_4B_MIXED_SFT_MODEL_ID,
            WORKPLACE_QWEN35_4B_MIXED_SFT_FINAL,
        ),
        (
            "qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-"
            "qwen35-4b-non-thinking",
            WORKPLACE_QWEN35_4B_FINAL_MIXED_SFT_MODEL_ID,
            WORKPLACE_QWEN35_4B_FINAL_MIXED_SFT_FINAL,
        ),
        (
            "qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-"
            "non-thinking-qwen35-4b-non-thinking",
            WORKPLACE_QWEN35_4B_FILTERED_SFT_MODEL_ID,
            WORKPLACE_QWEN35_4B_FILTERED_SFT_FINAL,
        ),
        (
            "qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-"
            "non-thinking-qwen35-4b-non-thinking",
            WORKPLACE_QWEN35_4B_GAIA2_SFT_MODEL_ID,
            WORKPLACE_QWEN35_4B_GAIA2_SFT_FINAL,
        ),
        (
            "qwen35-4b-sft-toolathlon-only-v1-493c24c4-teacher-prompt-"
            "filtered-32k-non-thinking-qwen35-4b-non-thinking",
            WORKPLACE_QWEN35_4B_TOOLATHLON_ONLY_SFT_MODEL_ID,
            WORKPLACE_QWEN35_4B_TOOLATHLON_ONLY_SFT_FINAL,
        ),
        (
            "qwen35-4b-sft-gaia2-execution-only-v1-110-n10-teacher-prompt-"
            "r1-balanced-32k-non-thinking-qwen35-4b-non-thinking",
            WORKPLACE_QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_MODEL_ID,
            WORKPLACE_QWEN35_4B_GAIA2_EXECUTION_ONLY_SFT_FINAL,
        ),
    ],
)
def test_sft_qwen_manager_and_base_worker_use_dedicated_gpus(
    name: str,
    manager_model_id: str,
    manager_checkpoint: Path,
) -> None:
    experiment = get_experiment(name)
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.manager_backend == "local_vllm"
    assert experiment.requires_openrouter is False
    assert experiment.num_gpus == 2
    assert experiment.max_model_len == 131072
    assert experiment.concurrency == 8
    assert experiment.subagent_graph == "qwen35"

    selected = models_for_experiment(experiment)
    assert [model.model_id for model in selected] == [
        manager_model_id,
        "Qwen/Qwen3.5-4B",
    ]
    assert [model.snapshot for model in selected] == [
        manager_checkpoint,
        experiments.QWEN35_4B_BASE,
    ]
    assert [model.gpu for model in selected] == [0, 1]
    assert [model.port for model in selected] == [8026, 8025]
    assert [model.gpu_memory_utilization for model in selected] == [0.9, 0.9]
    assert [model.thinking for model in selected] == [False, False]
    assert [model.dtype for model in selected] == ["bfloat16", "bfloat16"]

    for model in selected:
        command = run_module.decomposer_vllm_command(model, experiment)
        assert command[command.index("--dtype") + 1] == "bfloat16"
        assert command[command.index("--max-model-len") + 1] == "131072"
        assert command[command.index("--tool-call-parser") + 1] == "qwen3_xml"
        assert "--reasoning-parser" not in command
        assert command[command.index("--default-chat-template-kwargs") + 1] == (
            '{"enable_thinking":false}'
        )

    repo_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (
            repo_root
            / "gyms"
            / "workplace_assistant"
            / "configs"
            / experiment.gym_config_filename
        ).read_text()
    )
    assert config["responses_create_params"] == {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
    }
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert policy["base_url"] == "http://127.0.0.1:8026/v1"
    assert policy["model"] == manager_model_id
    assert policy["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert policy["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "include_reasoning": False,
    }
    agent = config["decomposer"]["responses_api_agents"]["decomposer_agent"]
    assert agent["decomposer_system_prompt_profile"] == "student"
    assert [item["assistant_id"] for item in agent["subagent_types"]] == [
        "qwen35_4b_non_thinking"
    ]

    plan = run_module._dry_plan(
        repo_root,
        experiment,
        "evaluation",
        "validation",
        3,
        None,
        Path("/tmp/workplace-qwen-sft-eval"),
        ("0", "1"),
    )
    assert plan["gpu_assignments"] == {
        "subagent_vllm_8026": "0",
        "subagent_vllm_8025": "1",
    }
    assert "--num-repeats 3" in plan["gym_eval"]
    assert "validation.decomposer.jsonl" in plan["gym_eval"]

    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="evaluation",
        split="validation",
        num_repeats=3,
        limit=None,
        author="alice",
        base_image=experiments.BASE_IMAGE,
        priority="high",
        force=False,
        proxy_env={},
        openrouter_key="",
    )
    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[2]
    assert payload["priority_class"] == "high"
    assert "--split validation" in payload["script"]
    assert "--num-repeats 3" in payload["script"]
    assert "OPENROUTER_API_KEY_DECOMPOSER" not in payload["env_variables"]
    assert payload["job_desc"] == (
        "workplace-assistant-validation-evaluation decomposer-agent "
        f"{run_name(experiment, 3)} "
        "#alice"
    )


def test_untuned_qwen_manager_matches_tuned_two_gpu_topology() -> None:
    name = "qwen35-4b-base-non-thinking-qwen35-4b-non-thinking"
    experiment = get_experiment(name)
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.manager_backend == "local_vllm"
    assert experiment.requires_openrouter is False
    assert experiment.num_gpus == 2
    assert experiment.max_model_len == 131072
    assert experiment.subagent_graph == "qwen35"

    selected = models_for_experiment(experiment)
    assert [model.model_id for model in selected] == [
        WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID,
        "Qwen/Qwen3.5-4B",
    ]
    assert [model.snapshot for model in selected] == [
        experiments.QWEN35_4B_BASE,
        experiments.QWEN35_4B_BASE,
    ]
    assert [model.gpu for model in selected] == [0, 1]
    assert [model.port for model in selected] == [8026, 8025]
    assert [model.thinking for model in selected] == [False, False]
    assert [model.dtype for model in selected] == ["bfloat16", "bfloat16"]

    for model in selected:
        command = run_module.decomposer_vllm_command(model, experiment)
        assert command[command.index("--dtype") + 1] == "bfloat16"
        assert command[command.index("--max-model-len") + 1] == "131072"
        assert command[command.index("--tool-call-parser") + 1] == "qwen3_xml"
        assert "--reasoning-parser" not in command
        assert command[command.index("--default-chat-template-kwargs") + 1] == (
            '{"enable_thinking":false}'
        )

    repo_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (
            repo_root
            / "gyms"
            / "workplace_assistant"
            / "configs"
            / experiment.gym_config_filename
        ).read_text()
    )
    assert config["responses_create_params"] == {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
    }
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert policy["base_url"] == "http://127.0.0.1:8026/v1"
    assert policy["model"] == WORKPLACE_QWEN35_4B_BASE_MANAGER_MODEL_ID
    assert policy["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert policy["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "include_reasoning": False,
    }
    agent = config["decomposer"]["responses_api_agents"]["decomposer_agent"]
    assert agent["decomposer_system_prompt_profile"] == "student"
    assert [item["assistant_id"] for item in agent["subagent_types"]] == [
        "qwen35_4b_non_thinking"
    ]

    plan = run_module._dry_plan(
        repo_root,
        experiment,
        "evaluation",
        "validation",
        3,
        None,
        Path("/tmp/workplace-qwen-base-manager-eval"),
        ("0", "1"),
    )
    assert plan["gpu_assignments"] == {
        "subagent_vllm_8026": "0",
        "subagent_vllm_8025": "1",
    }
    assert plan["decomposer_system_prompt_profile"] == "student"
    assert "--num-repeats 3" in plan["gym_eval"]
    assert "validation.decomposer.jsonl" in plan["gym_eval"]

    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="evaluation",
        split="validation",
        num_repeats=3,
        limit=None,
        author="alice",
        base_image=experiments.BASE_IMAGE,
        priority="medium",
        force=False,
        proxy_env={},
        openrouter_key="",
    )
    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[2]
    assert payload["priority_class"] == "medium"
    assert payload["base_image"].endswith("py3.12-torch2.7.0:0.0.42")
    assert "--purpose evaluation" in payload["script"]
    assert "--split validation" in payload["script"]
    assert "--num-repeats 3" in payload["script"]
    assert "OPENROUTER_API_KEY_DECOMPOSER" not in payload["env_variables"]
    assert payload["job_desc"] == (
        "workplace-assistant-validation-evaluation decomposer-agent "
        f"{run_name(experiment, 3)} "
        "#alice"
    )


def test_local_e4b_manager_shares_thinking_subagent_server() -> None:
    name = "gemma4-e4b-it-non-thinking-gemma4-e4b-thinking"
    experiment = get_experiment(name)
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.manager_backend == "local_vllm"
    assert experiment.requires_openrouter is False
    assert experiment.num_gpus == 1

    selected = models_for_experiment(experiment)
    assert [model.model_id for model in selected] == ["google/gemma-4-E4B-it"]
    assert selected[0].gpu == 0

    repo_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (
            repo_root
            / "gyms"
            / "workplace_assistant"
            / "configs"
            / experiment.gym_config_filename
        ).read_text()
    )
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert policy["base_url"] == "http://127.0.0.1:8021/v1"
    assert policy["model"] == "google/gemma-4-E4B-it"
    assert policy["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    agent = config["decomposer"]["responses_api_agents"]["decomposer_agent"]
    assert agent["decomposer_system_prompt_profile"] == "student"
    assert len(agent["subagent_types"]) == 1
    assert agent["subagent_types"][0]["assistant_id"] == "gemma_4_4b_thinking"

    server = run_module.decomposer_vllm_command(selected[0], experiment)
    assert server[server.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":true,"preserve_thinking":true}'
    )
    assert prepare_module.components_for_experiments((experiment,)) == (
        "resources_servers/workplace_assistant",
        "responses_api_agents/decomposer_agent",
        "responses_api_models/vllm_model",
    )

    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="evaluation",
        split="validation",
        num_repeats=3,
        limit=None,
        author="alice",
        base_image=experiments.BASE_IMAGE,
        priority=None,
        force=False,
        proxy_env={},
        openrouter_key="",
    )
    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert "--purpose evaluation" in payload["script"]
    assert "--split validation" in payload["script"]
    assert "--num-repeats 3" in payload["script"]
    assert "--limit" not in payload["script"]
    assert "OPENROUTER_API_KEY_DECOMPOSER" not in payload["env_variables"]
    assert "HTTPS_PROXY" not in payload["env_variables"]
    assert payload["job_desc"] == (
        "workplace-assistant-validation-evaluation decomposer-agent "
        f"{run_name(experiment, 3)} #alice"
    )


def test_sft_e4b_manager_and_vanilla_subagent_use_dedicated_gpus() -> None:
    name = "gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking"
    experiment = get_experiment(name)
    assert isinstance(experiment, DecomposerExperiment)
    assert experiment.manager_backend == "local_vllm"
    assert experiment.num_gpus == 2

    selected = models_for_experiment(experiment)
    assert [model.model_id for model in selected] == [
        WORKPLACE_E4B_SFT_MODEL_ID,
        "google/gemma-4-E4B-it",
    ]
    assert [model.snapshot for model in selected] == [
        WORKPLACE_E4B_SFT_VLLM,
        experiments.GEMMA4_E4B_BASE,
    ]
    assert WORKPLACE_E4B_SFT_VLLM == WORKPLACE_E4B_SFT_FINAL.with_name("final-vllm")
    assert [model.gpu for model in selected] == [0, 1]
    assert [model.port for model in selected] == [8024, 8021]
    assert [model.gpu_memory_utilization for model in selected] == [0.9, 0.9]
    assert [model.thinking for model in selected] == [False, True]

    manager_server = run_module.decomposer_vllm_command(selected[0], experiment)
    subagent_server = run_module.decomposer_vllm_command(selected[1], experiment)
    assert manager_server[
        manager_server.index("--default-chat-template-kwargs") + 1
    ] == ('{"enable_thinking":false}')
    assert subagent_server[
        subagent_server.index("--default-chat-template-kwargs") + 1
    ] == ('{"enable_thinking":true,"preserve_thinking":true}')

    repo_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (
            repo_root
            / "gyms"
            / "workplace_assistant"
            / "configs"
            / experiment.gym_config_filename
        ).read_text()
    )
    policy = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert policy["base_url"] == "http://127.0.0.1:8024/v1"
    assert policy["model"] == WORKPLACE_E4B_SFT_MODEL_ID
    assert policy["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    agent = config["decomposer"]["responses_api_agents"]["decomposer_agent"]
    assert agent["decomposer_system_prompt_profile"] == "student"
    assert [item["assistant_id"] for item in agent["subagent_types"]] == [
        "gemma_4_4b_thinking"
    ]

    plan = run_module._dry_plan(
        repo_root,
        experiment,
        "evaluation",
        "validation",
        3,
        None,
        Path("/tmp/workplace-sft-eval"),
        ("0", "1"),
    )
    assert plan["gpu_assignments"] == {
        "subagent_vllm_8024": "0",
        "subagent_vllm_8021": "1",
    }
    assert "--num-repeats 3" in plan["gym_eval"]
    assert "validation.decomposer.jsonl" in plan["gym_eval"]

    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="evaluation",
        split="validation",
        num_repeats=3,
        limit=None,
        author="alice",
        base_image=experiments.BASE_IMAGE,
        priority=None,
        force=False,
        proxy_env={},
        openrouter_key="",
    )
    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[2]
    assert "--split validation" in payload["script"]
    assert "--num-repeats 3" in payload["script"]
    assert "OPENROUTER_API_KEY_DECOMPOSER" not in payload["env_variables"]
    assert payload["job_desc"] == (
        "workplace-assistant-validation-evaluation decomposer-agent "
        f"{run_name(experiment, 3)} #alice"
    )


def test_simple_command_uses_local_python_supervisor_not_external_shell() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("gemma4-e2b-it-non-thinking")
    assert isinstance(experiment, SimpleExperiment)
    vllm = run_module.simple_vllm_command(experiment)
    assert vllm[:2] == [str(experiments.PROJECT_VENV / "bin" / "vllm"), "serve"]
    assert "--language-model-only" in vllm
    assert "--tool-call-parser" in vllm

    start = run_module.gym_start_command(
        repo_root,
        experiment,
        purpose="evaluation",
        gym_bin=Path("/gym"),
        component_root=Path("/components"),
        logs=Path("/logs"),
    )
    assert start[:4] == ["/gym", "env", "start", "--environment"]
    assert "run_gym_baseline_job.sh" not in " ".join(start)
    assert "workplace_assistant" in start


def test_decomposer_gym_start_overrides_prompt_for_run_purpose() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("glm-5-2-gemma4-26b-a4b-non-thinking")
    commands = {
        purpose: run_module.gym_start_command(
            repo_root,
            experiment,
            purpose=purpose,
            gym_bin=Path("/gym"),
            component_root=Path("/components"),
            logs=Path("/logs"),
        )
        for purpose in ("trace-generation", "evaluation")
    }
    assert any(
        argument.endswith("decomposer_system_prompt_profile=teacher")
        for argument in commands["trace-generation"]
    )
    assert any(
        argument.endswith("decomposer_system_prompt_profile=student")
        for argument in commands["evaluation"]
    )


def test_agent_profiles_share_one_gym_eval_builder() -> None:
    simple = get_experiment("gemma4-e2b-it-non-thinking")
    decomposer = get_experiment("glm-5-2-gemma4-26b-a4b-non-thinking")
    simple_command = run_module.gym_eval_command(
        simple,
        gym_bin=Path("/gym"),
        split="train",
        output=Path("/out/simple.jsonl"),
        num_repeats=3,
        limit=2,
        resume=True,
    )
    decomposer_command = run_module.gym_eval_command(
        decomposer,
        gym_bin=Path("/gym"),
        split="validation",
        output=Path("/out/decomposer.jsonl"),
        num_repeats=1,
        limit=None,
        resume=False,
    )
    assert simple_command[:3] == decomposer_command[:3] == ["/gym", "eval", "run"]
    assert (
        simple_command[simple_command.index("--agent") + 1]
        == "workplace_assistant_simple_agent"
    )
    assert decomposer_command[decomposer_command.index("--agent") + 1] == "decomposer"
    assert "train.jsonl" in simple_command[simple_command.index("--input") + 1]
    assert (
        "validation.decomposer.jsonl"
        in decomposer_command[decomposer_command.index("--input") + 1]
    )
    assert "--resume" in simple_command
    assert "--limit" in simple_command


def test_decomposer_service_topology_has_subagent_vllm_and_langgraph() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("glm-5-2-gemma4-26b-a4b-non-thinking")
    assert isinstance(experiment, DecomposerExperiment)
    model_command = run_module.decomposer_vllm_command(
        models_for_experiment(experiment)[0], experiment
    )
    langgraph, cwd = run_module.langgraph_command(repo_root, experiment)
    assert "--served-model-name" in model_command
    assert "langgraph" in Path(langgraph[0]).name
    assert langgraph[langgraph.index("--port") + 1] == "2024"
    assert cwd.name == "subagents"


def test_local_cuda_devices_remap_decomposer_logical_slots() -> None:
    experiment = get_experiment("glm-5-2-gemma4-all")
    requested = run_module.parse_cuda_visible_devices("3, 5,7")
    devices = run_module.selected_cuda_devices(experiment, requested)
    assert devices == ("3", "5", "7")
    assert {
        model.model_id: devices[model.gpu]
        for model in models_for_experiment(experiment)
    } == {
        "google/gemma-4-E2B-it": "3",
        "google/gemma-4-E4B-it": "3",
        "google/gemma-4-12B-it": "5",
        "google/gemma-4-26B-A4B-it": "7",
    }

    one_gpu = get_experiment("deepseek-v4-flash-0731-gemma4-e4b-thinking")
    assert run_module.selected_cuda_devices(one_gpu, ("2",)) == ("2",)
    with pytest.raises(ValueError, match="requires 1 CUDA device"):
        run_module.selected_cuda_devices(one_gpu, ("2", "3"))
    with pytest.raises(run_module.argparse.ArgumentTypeError, match="must be unique"):
        run_module.parse_cuda_visible_devices("2,2")


def test_local_dry_plan_routes_every_output_and_reports_gpu(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("deepseek-v4-flash-0731-gemma4-e4b-thinking")
    plan = run_module._dry_plan(
        repo_root,
        experiment,
        "trace-generation",
        "train",
        3,
        1,
        tmp_path,
        ("2",),
    )
    assert plan["output_dir"] == str(tmp_path)
    assert plan["purpose"] == "trace-generation"
    assert plan["decomposer_system_prompt_profile"] == "teacher"
    assert plan["gpu_assignments"] == {"subagent_vllm_8021": "2"}
    assert str(tmp_path / "rollouts.jsonl") in plan["gym_eval"]
    assert str(tmp_path / "logs" / "gym_components") in plan["gym_start"]


def test_prepare_decomposer_dataset_sets_agent_ref(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "prepared.jsonl"
    source.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(3)))
    manifest = prepare_module.prepare_decomposer_dataset(source, destination, 3)
    assert manifest["rows"] == 3
    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert all(record["agent_ref"] == prepare_module.AGENT_REF for record in records)


def test_source_preparation_wraps_upstream_and_materializes_both_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments, "DATA_DIR", tmp_path)
    monkeypatch.setattr(prepare_module, "DATA_DIR", tmp_path)
    monkeypatch.setitem(experiments.SPLIT_ROWS, "train", 3)

    def fake_upstream(
        repo_root: Path, gym_python: Path, split: str, destination_dir: Path
    ) -> Path:
        del repo_root, gym_python
        output = destination_dir / f"{split}.jsonl"
        output.write_text(
            "".join(json.dumps({"id": index}) + "\n" for index in range(3))
        )
        return output

    monkeypatch.setattr(prepare_module, "_run_upstream_preparer", fake_upstream)
    raw, decomposer = prepare_module.prepare_source(
        tmp_path, Path("/python"), "train", reuse_source=False
    )
    assert raw["rows"] == decomposer["rows"] == 3
    assert Path(raw["path"]).name == "train.jsonl"
    assert Path(decomposer["path"]).name == "train.decomposer.jsonl"


def test_component_preparation_repairs_partial_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component = "resources_servers/workplace_assistant"
    component_dir = tmp_path / "external" / "Gym" / component
    component_dir.mkdir(parents=True)
    (component_dir / "requirements.txt").write_text("example==1\n")
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "external" / "Gym" / "uv.lock").write_text("")

    runtime_root = tmp_path / "runtime"
    python = runtime_root / component / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    (python.parent / "activate").write_text("")

    monkeypatch.setattr(prepare_module, "component_venv_root", lambda _: runtime_root)
    monkeypatch.setattr(prepare_module, "_uv_executable", lambda: Path("/uv"))
    commands: list[list[str]] = []
    version_probes = 0

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal version_probes
        commands.append(command)
        if command[0] == str(python):
            version_probes += 1
            if version_probes == 1:
                return subprocess.CompletedProcess(command, 1, "", "missing")
            return subprocess.CompletedProcess(command, 0, "2.49.2\n1.102.0\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prepare_module.subprocess, "run", fake_run)
    result = prepare_module.prepare_component_venvs(
        tmp_path,
        (component,),
        {"ray": "2.49.2", "openai": "1.102.0"},
        create=True,
    )

    assert result["components"][component]["python"] == str(python)
    install_commands = [
        command for command in commands if command[:3] == ["/uv", "pip", "install"]
    ]
    assert len(install_commands) == 2


def test_checkpoint_validation_requires_every_shard(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001.safetensors"}})
    )
    shard = tmp_path / "model-00001.safetensors"
    shard.write_bytes(b"weights")
    manifest = prepare_module.validate_checkpoint(tmp_path)
    assert manifest["safetensors_shards"] == [shard.name]
    shard.unlink()
    with pytest.raises(FileNotFoundError):
        prepare_module.validate_checkpoint(tmp_path)


def test_result_validation_uses_split_limit_and_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(experiments.SPLIT_ROWS, "validation", 5)
    rollouts = tmp_path / "rollouts.jsonl"
    rows = [
        {
            "_ng_task_index": task_index,
            "_ng_rollout_index": rollout_index,
            "reward": 0.0 if (task_index, rollout_index) == (1, 2) else 1.0,
            **(
                {
                    "_ng_rollout_error": {
                        "type": "http_500",
                        "status_code": 500,
                        "detail": "synthetic failure",
                    }
                }
                if (task_index, rollout_index) == (1, 2)
                else {}
            ),
        }
        for task_index in range(2)
        for rollout_index in range(3)
    ]
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "rollouts_aggregate_metrics.json").write_text("{}")
    result = run_module.validate_result("validation", 3, rollout_path=rollouts, limit=2)
    assert result["task_rows"] == 2
    assert result["rollout_rows"] == 6
    assert result["rollout_error_rows"] == 1
    assert result["rollout_error_types"] == {"http_500": 1}


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_result_validation_requires_exact_task_repeat_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setitem(experiments.SPLIT_ROWS, "validation", 2)
    rows = [
        {
            "_ng_task_index": task_index,
            "_ng_rollout_index": rollout_index,
            "reward": 1.0,
        }
        for task_index in range(2)
        for rollout_index in range(2)
    ]
    if mutation == "missing":
        rows.pop()
    else:
        rows[-1] = dict(rows[0])
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "rollouts_aggregate_metrics.json").write_text("{}")

    expected_error = "missing" if mutation == "missing" else "Duplicate"
    with pytest.raises(ValueError, match=expected_error):
        run_module.validate_result(
            "validation", 2, rollout_path=rollouts, limit=None
        )


def test_force_archives_previous_attempt(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "logs").mkdir(parents=True)
    (output / "logs" / "vllm.log").write_text("old")
    (output / "run_status.json").write_text(
        json.dumps({"started_at": "2026-08-18T12:00:00+00:00"})
    )
    archive = run_module.archive_attempt(output)
    assert archive is not None
    assert (archive / "logs" / "vllm.log").read_text() == "old"
    assert not (output / "run_status.json").exists()


def test_call_limit_artifact_migration_preserves_raw_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = migration_module.Migration(
        source="old-n3",
        destination="new-calls100-n3",
        source_experiment="old-profile",
        destination_experiment="canonical-profile",
        num_repeats=3,
        max_steps=100,
    )
    monkeypatch.setattr(migration_module, "MIGRATIONS", (spec,))
    source = tmp_path / spec.source
    source.mkdir()
    raw = source / "rollouts.jsonl"
    raw.write_bytes(b'{"reward": 1}\n')
    metadata = {
        "schema_version": 3,
        "state": "complete",
        "experiment": spec.source_experiment,
        "run_name": spec.source,
        "kind": "simple",
        "purpose": "evaluation",
        "split": "validation",
        "num_repeats": 3,
        "limit": None,
        "output_dir": str(source),
        "result": {"rollouts": str(raw)},
    }
    for filename in ("run_status.json", ".eval_done.json"):
        (source / filename).write_text(json.dumps(metadata))

    dry = migration_module.migrate(tmp_path, apply=False)
    assert dry["records"][0]["action"] == "move"
    assert source.is_dir()

    before = raw.read_bytes()
    applied = migration_module.migrate(tmp_path, apply=True)
    destination = tmp_path / spec.destination
    assert applied["applied"] is True
    assert not source.exists()
    assert (destination / "rollouts.jsonl").read_bytes() == before
    marker = json.loads((destination / ".eval_done.json").read_text())
    assert marker["experiment"] == spec.destination_experiment
    assert marker["run_name"] == spec.destination
    assert marker["simple_agent_max_steps"] == 100
    assert marker["result"]["rollouts"] == str(destination / "rollouts.jsonl")
    assert (tmp_path / migration_module.MIGRATION_MANIFEST).is_file()

    resumed = migration_module.migrate(tmp_path, apply=True)
    assert resumed == applied


def test_full_run_ignores_and_preserves_nested_smoke_output(tmp_path: Path) -> None:
    experiment = get_experiment("qwen36-35b-a3b-teacher-qwen35-4b-non-thinking")
    smoke = tmp_path / "smoke_1"
    smoke.mkdir(parents=True)
    (smoke / ".eval_done.json").write_text("{}")

    run_module.validate_existing_attempt_identity(
        tmp_path,
        experiment,
        purpose="trace-generation",
        split="validation",
        num_repeats=3,
        limit=None,
        force=False,
    )

    (tmp_path / "run_status.json").write_text(
        json.dumps({"started_at": "2026-08-29T12:00:00+00:00"})
    )
    archive = run_module.archive_attempt(tmp_path)
    assert archive is not None
    assert (archive / "run_status.json").is_file()
    assert (smoke / ".eval_done.json").is_file()


def test_run_commands_require_explicit_purpose() -> None:
    with pytest.raises(SystemExit):
        run_module.build_parser().parse_args(["--experiment", "example"])
    with pytest.raises(SystemExit):
        run_eval.build_parser().parse_args(
            ["--experiment", "example", "--author-name", "alice"]
        )


def test_legacy_teacher_output_without_call_identity_cannot_be_reused(
    tmp_path: Path,
) -> None:
    experiment = get_experiment("glm-5-2-gemma4-26b-a4b-non-thinking")
    (tmp_path / ".eval_done.json").write_text("{}")
    with pytest.raises(RuntimeError, match="purpose='trace-generation'"):
        run_module.validate_existing_attempt_identity(
            tmp_path,
            experiment,
            purpose="evaluation",
            split="train",
            num_repeats=1,
            limit=None,
            force=False,
        )
    with pytest.raises(RuntimeError, match="missing required identity field"):
        run_module.validate_existing_attempt_identity(
            tmp_path,
            experiment,
            purpose="trace-generation",
            split="train",
            num_repeats=1,
            limit=None,
            force=False,
        )
    run_module.validate_existing_attempt_identity(
        tmp_path,
        experiment,
        purpose="evaluation",
        split="train",
        num_repeats=1,
        limit=None,
        force=True,
    )


def test_existing_completion_marker_skips_without_new_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experiments, "RESULTS_ROOT", tmp_path)
    experiment = get_experiment("gemma4-e2b-it-non-thinking")
    marker = completion_marker(experiment, "train", purpose="evaluation")
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "experiment": experiment.name,
                "purpose": "evaluation",
                "split": "train",
                "num_repeats": 1,
                "limit": None,
                "simple_agent_max_steps": 100,
                "runtime_configuration": run_module.runtime_configuration(
                    experiment
                ),
            }
        )
    )
    args = run_module.build_parser().parse_args(
        [
            "--experiment",
            experiment.name,
            "--purpose",
            "evaluation",
            "--split",
            "train",
        ]
    )
    assert run_module.execute(Path(__file__).resolve().parents[2], args) == 0
    assert "Skip (completed)" in capsys.readouterr().out


def test_custom_output_completion_marker_is_isolated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment = get_experiment("gemma4-e2b-it-non-thinking")
    custom_output = tmp_path / "local" / experiment.name
    custom_output.mkdir(parents=True)
    (custom_output / ".eval_done.json").write_text(
        json.dumps(
            {
                "experiment": experiment.name,
                "purpose": "evaluation",
                "split": "train",
                "num_repeats": 1,
                "limit": None,
                "simple_agent_max_steps": 100,
                "runtime_configuration": run_module.runtime_configuration(
                    experiment
                ),
            }
        )
    )
    args = run_module.build_parser().parse_args(
        [
            "--experiment",
            experiment.name,
            "--purpose",
            "evaluation",
            "--split",
            "train",
            "--output-dir",
            str(custom_output),
        ]
    )
    assert run_module.execute(Path(__file__).resolve().parents[2], args) == 0
    assert f"Skip (completed): {custom_output / '.eval_done.json'}" in (
        capsys.readouterr().out
    )


def test_submission_rejects_capped_marker_for_uncapped_simple_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(experiments, "RESULTS_ROOT", tmp_path)
    experiment = get_experiment("gemma4-e2b-it-non-thinking")
    assert isinstance(experiment, SimpleExperiment)
    marker = completion_marker(experiment, "train", purpose="evaluation")
    marker.parent.mkdir(parents=True)
    capped_runtime = run_module.runtime_configuration(experiment)
    capped_runtime["max_output_tokens"] = 32768
    marker.write_text(
        json.dumps(
            {
                "experiment": experiment.name,
                "purpose": "evaluation",
                "split": "train",
                "num_repeats": 1,
                "limit": None,
                "simple_agent_max_steps": 100,
                "runtime_configuration": capped_runtime,
            }
        )
    )
    monkeypatch.setattr(
        run_eval.sys,
        "argv",
        [
            "run_eval",
            "--experiment",
            experiment.name,
            "--purpose",
            "evaluation",
            "--split",
            "train",
            "--author-name",
            "alice",
            "--dry",
        ],
    )

    with pytest.raises(RuntimeError, match="runtime_configuration"):
        run_eval.main()

    uncapped_runtime = run_module.runtime_configuration(experiment)
    marker.write_text(
        json.dumps(
            {
                "experiment": experiment.name,
                "purpose": "evaluation",
                "split": "train",
                "num_repeats": 1,
                "limit": None,
                "simple_agent_max_steps": 100,
                "runtime_configuration": uncapped_runtime,
            }
        )
    )
    assert run_eval.main() == 0
    assert f"Skip (completed): {marker}" in capsys.readouterr().out


def test_job_payload_uses_shared_runner_and_redacts_decomposer_secrets() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    experiment = get_experiment("glm-5-2-gemma4-26b-a4b-non-thinking")
    payload = run_eval.build_payload(
        experiment,
        repo_root,
        purpose="evaluation",
        split="train",
        num_repeats=1,
        limit=1,
        author="alice",
        base_image=experiments.BASE_IMAGE,
        priority=None,
        force=False,
        proxy_env={"HTTPS_PROXY": "http://user:password@proxy"},
        openrouter_key="secret",
    )
    assert "gyms/workplace_assistant/run.py" in payload["script"]
    assert "--purpose evaluation" in payload["script"]
    assert "--output-dir" not in payload["script"]
    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert "workplace-assistant-train-evaluation" in payload["job_desc"]
    assert payload["job_desc"].endswith("#alice")
    redacted = run_eval.redact_payload(payload)
    assert redacted["env_variables"]["OPENROUTER_API_KEY_DECOMPOSER"] == "<redacted>"
    assert redacted["env_variables"]["HTTPS_PROXY"] == "<redacted>"


def test_named_sft_specs_are_gym_owned() -> None:
    spec_root = Path(prepare_module.__file__).with_name("sft_specs")
    assert set(prepare_module.SFT_SPECS) == {
        "workplace-all-v3",
        "workplace-26b-nonthinking-v3",
        "workplace-deepseek-e4b-thinking-v1",
        "workplace-deepseek-e4b-thinking-v2-8k",
        "workplace-deepseek-e4b-thinking-v2-32k",
    }
    assert all(
        (spec_root / filename).is_file()
        for filename in prepare_module.SFT_SPECS.values()
    )
