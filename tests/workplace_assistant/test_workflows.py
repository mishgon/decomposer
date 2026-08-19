from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from gyms.workplace_assistant import experiments
from gyms.workplace_assistant import prepare as prepare_module
from gyms.workplace_assistant import run as run_module
from gyms.workplace_assistant import run_eval
from gyms.workplace_assistant.experiments import (
    DECOMPOSER_EXPERIMENTS,
    INSTANCE_TYPES_BY_NUM_GPUS,
    SIMPLE_EXPERIMENTS,
    WORKPLACE_E4B_SFT_FINAL,
    WORKPLACE_E4B_SFT_MODEL_ID,
    WORKPLACE_E4B_SFT_VLLM,
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


def test_registry_is_global_and_unique() -> None:
    assert len(DECOMPOSER_EXPERIMENTS) == 7
    assert len(SIMPLE_EXPERIMENTS) == 26
    assert len(experiments.EXPERIMENTS) == 33
    assert {experiment.kind for experiment in experiments.ALL_EXPERIMENTS} == {
        "decomposer",
        "simple",
    }


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
        assert agent["decomposer_system_prompt_profile"] == "student"


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
    assert run_name(simple) == simple.name
    assert run_name(simple, 5) == f"{simple.name}-n5"
    assert output_dir(simple, "train", 5, purpose="evaluation").parts[-2:] == (
        "train",
        f"{simple.name}-n5",
    )
    teacher = output_dir(decomposer, "train", 3, purpose="trace-generation")
    student = output_dir(decomposer, "train", 3, purpose="evaluation")
    assert teacher.parts[-2:] == ("train", f"{decomposer.name}-n3")
    assert student.parts[-3:] == (
        "train",
        "evaluation",
        f"{decomposer.name}-n3",
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
        "glm-5-2-gemma4-26b-a4b-non-thinking-n3"
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
    assert sum(model.gpu_memory_utilization for model in selected if model.gpu == 0) == pytest.approx(0.9)

    one = get_experiment("glm-5-2-gemma4-26b-a4b-non-thinking")
    assert isinstance(one, DecomposerExperiment)
    selected = models_for_experiment(one)
    assert len(selected) == 1
    assert selected[0].gpu == 0
    assert selected[0].startup_wave == 0


def test_deepseek_e4b_thinking_profile_is_single_type_and_single_gpu() -> None:
    experiment = get_experiment(
        "deepseek-v4-flash-0731-gemma4-e4b-thinking"
    )
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
        '{"enable_thinking":true}'
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
    assert "deepseek-v4-flash-0731-gemma4-e4b-thinking-n3" in payload[
        "job_desc"
    ]
    assert payload["job_desc"] == (
        "workplace-assistant-train decomposer-agent "
        "deepseek-v4-flash-0731-gemma4-e4b-thinking-n3 #alice"
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
        '{"enable_thinking":true}'
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
        f"workplace-assistant-validation-evaluation decomposer-agent {name}-n3 #alice"
    )


def test_sft_e4b_manager_and_vanilla_subagent_use_dedicated_gpus() -> None:
    name = (
        "gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-"
        "gemma4-e4b-thinking"
    )
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
    assert WORKPLACE_E4B_SFT_VLLM == WORKPLACE_E4B_SFT_FINAL.with_name(
        "final-vllm"
    )
    assert [model.gpu for model in selected] == [0, 1]
    assert [model.port for model in selected] == [8024, 8021]
    assert [model.gpu_memory_utilization for model in selected] == [0.9, 0.9]
    assert [model.thinking for model in selected] == [False, True]

    manager_server = run_module.decomposer_vllm_command(selected[0], experiment)
    subagent_server = run_module.decomposer_vllm_command(selected[1], experiment)
    assert manager_server[manager_server.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":false}'
    )
    assert subagent_server[subagent_server.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking":true}'
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
        f"workplace-assistant-validation-evaluation decomposer-agent {name}-n3 #alice"
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
    assert simple_command[simple_command.index("--agent") + 1] == "workplace_assistant_simple_agent"
    assert decomposer_command[decomposer_command.index("--agent") + 1] == "decomposer"
    assert "train.jsonl" in simple_command[simple_command.index("--input") + 1]
    assert "validation.decomposer.jsonl" in decomposer_command[decomposer_command.index("--input") + 1]
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
    with pytest.raises(
        run_module.argparse.ArgumentTypeError, match="must be unique"
    ):
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
        command
        for command in commands
        if command[:3] == ["/uv", "pip", "install"]
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
    rollouts.write_text(
        "".join(json.dumps({"id": index}) + "\n" for index in range(6))
    )
    (tmp_path / "rollouts_aggregate_metrics.json").write_text("{}")
    result = run_module.validate_result(
        "validation", 3, rollout_path=rollouts, limit=2
    )
    assert result["task_rows"] == 2
    assert result["rollout_rows"] == 6


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


def test_run_commands_require_explicit_purpose() -> None:
    with pytest.raises(SystemExit):
        run_module.build_parser().parse_args(["--experiment", "example"])
    with pytest.raises(SystemExit):
        run_eval.build_parser().parse_args(
            ["--experiment", "example", "--author-name", "alice"]
        )


def test_legacy_teacher_output_cannot_be_reused_for_student_evaluation(
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
    marker.write_text("{}")
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
    (custom_output / ".eval_done.json").write_text("{}")
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


def test_job_payload_uses_shared_runner_and_redacts_decomposer_secrets(
) -> None:
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
    }
    assert all((spec_root / filename).is_file() for filename in prepare_module.SFT_SPECS.values())
