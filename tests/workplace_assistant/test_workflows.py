from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gyms.workplace_assistant import experiments
from gyms.workplace_assistant import prepare as prepare_module
from gyms.workplace_assistant import run as run_module
from gyms.workplace_assistant import run_eval
from gyms.workplace_assistant.experiments import (
    DECOMPOSER_EXPERIMENTS,
    INSTANCE_TYPES_BY_NUM_GPUS,
    SIMPLE_EXPERIMENTS,
    DecomposerExperiment,
    SimpleExperiment,
    collect_experiments,
    completion_marker,
    get_experiment,
    models_for_experiment,
    output_dir,
    run_name,
)


def test_registry_is_global_and_unique() -> None:
    assert len(DECOMPOSER_EXPERIMENTS) == 5
    assert len(SIMPLE_EXPERIMENTS) == 26
    assert len(experiments.EXPERIMENTS) == 31
    assert {experiment.kind for experiment in experiments.ALL_EXPERIMENTS} == {
        "decomposer",
        "simple",
    }


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
    experiment = get_experiment("qwen35-2b-base-non-thinking")
    assert run_name(experiment) == experiment.name
    assert run_name(experiment, 5) == f"{experiment.name}-n5"
    assert output_dir(experiment, "train", 5).parts[-2:] == (
        "train",
        f"{experiment.name}-n5",
    )
    assert output_dir(experiment, "validation", 5, 2).name == "smoke_2"
    assert completion_marker(experiment, "validation", 5, 2).name == ".eval_done.json"


def test_live_allocation_instance_types_are_single_source_of_truth() -> None:
    assert INSTANCE_TYPES_BY_NUM_GPUS == {
        1: "a100plus.1gpu.80vG.12C.244G",
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
    assert "--num-repeats 3" in payload["script"]
    assert "deepseek-v4-flash-0731-gemma4-e4b-thinking-n3" in payload[
        "job_desc"
    ]


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
        gym_bin=Path("/gym"),
        component_root=Path("/components"),
        logs=Path("/logs"),
    )
    assert start[:4] == ["/gym", "env", "start", "--environment"]
    assert "run_gym_baseline_job.sh" not in " ".join(start)
    assert "workplace_assistant" in start


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
        "train",
        3,
        1,
        tmp_path,
        ("2",),
    )
    assert plan["output_dir"] == str(tmp_path)
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


def test_existing_completion_marker_skips_without_new_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experiments, "RESULTS_ROOT", tmp_path)
    experiment = get_experiment("gemma4-e2b-it-non-thinking")
    marker = completion_marker(experiment, "train")
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    args = run_module.build_parser().parse_args(
        ["--experiment", experiment.name, "--split", "train"]
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
    assert "--output-dir" not in payload["script"]
    assert payload["instance_type"] == INSTANCE_TYPES_BY_NUM_GPUS[1]
    assert payload["job_desc"].endswith("#alice")
    redacted = run_eval.redact_payload(payload)
    assert redacted["env_variables"]["OPENROUTER_API_KEY_DECOMPOSER"] == "<redacted>"
    assert redacted["env_variables"]["HTTPS_PROXY"] == "<redacted>"


def test_named_sft_specs_are_gym_owned() -> None:
    spec_root = Path(prepare_module.__file__).with_name("sft_specs")
    assert set(prepare_module.SFT_SPECS) == {
        "workplace-all-v3",
        "workplace-26b-nonthinking-v3",
    }
    assert all((spec_root / filename).is_file() for filename in prepare_module.SFT_SPECS.values())
